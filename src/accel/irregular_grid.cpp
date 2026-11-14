// IrregularGrid build: a dense power-of-2 local grid per top-level cell (sort-based, like the
// other grids), then HAGrid's merge and expand passes. Merge only remaps `local_entries` values
// (its shape never changes), so each round is a count -> scan -> compact loop. Expand only grows
// bounds, which is safe because the absorbed neighbor's references are a subset of the cell's own.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "irregular_grid.h"
#include "grid_common.h"

#include <algorithm>
#include <utility>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "../core/thrust_scratch.h"

namespace happyray
{
	namespace
	{

		// Power-of-two local resolutions allow repeated per-axis halving during merge.
		HAPPYRAY_HD inline int RoundUpPow2(int x)
		{
			int p = 1;
			while (p < x)
			{
				p *= 2;
			}
			return p;
		}

		HAPPYRAY_HD inline int Log2Ceil(int x)
		{
			int rounds = 0;
			while ((1 << rounds) < x)
			{
				++rounds;
			}
			return rounds;
		}

		// Sorted-list subset test, advancing through big unconditionally to guarantee termination.
		HAPPYRAY_HD inline bool
		IsSubsetOf(const uint32_t *small, uint32_t small_count, const uint32_t *big, uint32_t big_count)
		{
			if (small_count > big_count)
			{
				return false;
			}
			if (small_count == 0)
			{
				return true;
			}
			uint32_t i = 0;
			uint32_t j = 0;
			do
			{
				const uint32_t a = big[i];
				const uint32_t b = small[j];
				if (b < a)
				{
					return false;
				}
				j += (a == b);
				++i;
			} while (i < big_count && j < small_count);
			return j == small_count;
		}

		// Inputs are sorted and duplicate-free, matching HAGrid's count_union/merge_refs.
		HAPPYRAY_HD inline uint32_t
		CountSortedUnion(const uint32_t *a, uint32_t count_a, const uint32_t *b, uint32_t count_b)
		{
			uint32_t i = 0, j = 0, count = 0;
			while (i < count_a && j < count_b)
			{
				const uint32_t va = a[i];
				const uint32_t vb = b[j];
				i += (va <= vb);
				j += (va >= vb);
				++count;
			}
			return count + (count_a - i) + (count_b - j);
		}

		HAPPYRAY_HD inline uint32_t
		MergeSortedUnique(const uint32_t *a, uint32_t count_a, const uint32_t *b, uint32_t count_b, uint32_t *out)
		{
			uint32_t i = 0, j = 0, count = 0;
			while (i < count_a && j < count_b)
			{
				const uint32_t va = a[i];
				const uint32_t vb = b[j];
				out[count++]	  = va < vb ? va : vb;
				i += (va <= vb);
				j += (va >= vb);
			}
			while (i < count_a)
				out[count++] = a[i++];
			while (j < count_b)
				out[count++] = b[j++];
			return count;
		}

		// Axis-generic helpers.

		HAPPYRAY_HD inline IrregularCell
		MakeCell(const int min_arr[3], const int max_arr[3], uint32_t ref_start, uint32_t ref_end)
		{
			IrregularCell cell;
			cell.min		= Vec3i(min_arr[0], min_arr[1], min_arr[2]);
			cell.max		= Vec3i(max_arr[0], max_arr[1], max_arr[2]);
			cell.references = {ref_start, ref_end};
			return cell;
		}

		HAPPYRAY_HD inline void CellMinArr(const IrregularCell &cell, int out[3])
		{
			out[0] = cell.min.x;
			out[1] = cell.min.y;
			out[2] = cell.min.z;
		}

		HAPPYRAY_HD inline void CellMaxArr(const IrregularCell &cell, int out[3])
		{
			out[0] = cell.max.x;
			out[1] = cell.max.y;
			out[2] = cell.max.z;
		}

		// Phase A: dense power-of-2 local grid per top-level cell.

		// Everything except `entries_offset`, which needs every cell's voxel count first.
		struct ComputeCellInfoFunctor
		{
			Vec3f			 top_bounds_min;
			Vec3f			 top_cell_size;
			Vec2i			 top_resolution;
			const CellRange *top_cell_ranges;
			float			 second_level_density;
			uint32_t		 dense_cell_threshold;

			HAPPYRAY_HD IrregularCellInfo operator()(uint32_t c) const
			{
				const int	   cz  = static_cast<int>(c) / (top_resolution.x * top_resolution.y);
				const int	   rem = static_cast<int>(c) % (top_resolution.x * top_resolution.y);
				const int	   cy  = rem / top_resolution.x;
				const int	   cx  = rem % top_resolution.x;
				const Vec3f	   cell_min(top_bounds_min.x + cx * top_cell_size.x,
										top_bounds_min.y + cy * top_cell_size.y,
										top_bounds_min.z + cz * top_cell_size.z);
				const Vec3f	   cell_max		   = cell_min + top_cell_size;
				const uint32_t primitive_count = top_cell_ranges[c].end - top_cell_ranges[c].start;

				Vec3i res =
					(primitive_count <= dense_cell_threshold)
						? Vec3i(1, 1, 1)
						: ComputeGridResolution(BBox(cell_min, cell_max), primitive_count, second_level_density);
				res.x = RoundUpPow2(res.x);
				res.y = RoundUpPow2(res.y);
				res.z = RoundUpPow2(res.z);

				IrregularCellInfo info;
				info.bounds_min		= cell_min;
				info.cell_size		= Vec3f((cell_max.x - cell_min.x) / res.x,
											(cell_max.y - cell_min.y) / res.y,
											(cell_max.z - cell_min.z) / res.z);
				info.inv_cell_size	= Vec3f(1.0f / info.cell_size.x, 1.0f / info.cell_size.y, 1.0f / info.cell_size.z);
				info.resolution		= res;
				info.entries_offset = 0;
				return info;
			}
		};

		// `shift` = log2(global_resolution / resolution), so lattice conversion is a bit shift.
		struct SetGlobalScaleFunctor
		{
			Vec2i top_resolution;
			Vec3i global_resolution;

			HAPPYRAY_HD IrregularCellInfo operator()(uint32_t c, IrregularCellInfo info) const
			{
				const int cz  = static_cast<int>(c) / (top_resolution.x * top_resolution.y);
				const int rem = static_cast<int>(c) % (top_resolution.x * top_resolution.y);
				const int cy  = rem / top_resolution.x;
				const int cx  = rem % top_resolution.x;
				info.shift	  = Vec3i(Log2Ceil(global_resolution.x) - Log2Ceil(info.resolution.x),
									  Log2Ceil(global_resolution.y) - Log2Ceil(info.resolution.y),
									  Log2Ceil(global_resolution.z) - Log2Ceil(info.resolution.z));
				info.global_offset =
					Vec3i(cx * global_resolution.x, cy * global_resolution.y, cz * global_resolution.z);
				return info;
			}
		};

		struct ExtractNumVoxelsFunctor
		{
			HAPPYRAY_HD uint32_t operator()(const IrregularCellInfo &info) const
			{
				return static_cast<uint32_t>(info.resolution.x) * info.resolution.y * info.resolution.z;
			}
		};

		struct SetEntriesOffsetFunctor
		{
			HAPPYRAY_HD IrregularCellInfo operator()(IrregularCellInfo info, uint32_t offset) const
			{
				info.entries_offset = offset;
				return info;
			}
		};

		// Writes top-level cell id `c` into every slot of its range (used for pairs and voxels).
		struct ScatterOwnerFunctor
		{
			const uint32_t *range_start;
			const uint32_t *range_end;
			uint32_t	   *owner;

			HAPPYRAY_HD void operator()(uint32_t c) const
			{
				const uint32_t begin = range_start[c];
				const uint32_t end	 = range_end[c];
				for (uint32_t i = begin; i < end; ++i)
				{
					owner[i] = c;
				}
			}
		};

		struct ScatterCellRangeOwnerFunctor
		{
			const CellRange *ranges;
			uint32_t		*owner;

			HAPPYRAY_HD void operator()(uint32_t c) const
			{
				for (uint32_t i = ranges[c].start; i < ranges[c].end; ++i)
				{
					owner[i] = c;
				}
			}
		};

		// Counts/writes the owning top-level cell's finest voxels a top-level (cell, primitive) pair
		// overlaps (see detail::ForEachOverlappedCell). kHybrid pads unused slots with
		// `invalid_voxel`.
		template <typename P, CellOverlapMode Mode, bool Write> struct Fragments
		{
			SceneView				 scene;
			const uint32_t			*owning_top_cell;
			const uint32_t			*top_prim_indices;
			const IrregularCellInfo *infos;
			const uint32_t			*write_offsets	   = nullptr;
			const uint32_t			*counts			   = nullptr;
			uint32_t				 invalid_voxel	   = 0;
			uint32_t				*out_voxel_indices = nullptr;
			uint32_t				*out_prim_indices  = nullptr;

			HAPPYRAY_HD uint32_t operator()(uint32_t i) const
			{
				const IrregularCellInfo &info	   = infos[owning_top_cell[i]];
				const uint32_t			 prim	   = top_prim_indices[i];
				const P					 primitive = PrimitiveTraits<P>::Fetch(scene, prim);
				const BBox				 box	   = BBoxExtractor<P>::Get(primitive);
				if constexpr (!Write)
				{
					return detail::CountOverlappedCells<P, Mode>(primitive,
																 box,
																 info.bounds_min,
																 info.inv_cell_size,
																 info.cell_size,
																 info.resolution);
				}
				else
				{
					uint32_t	   offset = write_offsets[i];
					const uint32_t end	  = offset + counts[i];
					const uint32_t count  = detail::ForEachOverlappedCell<P, Mode>(
						primitive,
						box,
						info.bounds_min,
						info.inv_cell_size,
						info.cell_size,
						info.resolution,
						[&](const Vec3i &cell)
						{
							out_voxel_indices[offset] = info.entries_offset + FlattenCellIndex(cell, info.resolution);
							out_prim_indices[offset]  = prim;
							++offset;
						});
					while (offset < end)
					{
						out_voxel_indices[offset] = invalid_voxel;
						out_prim_indices[offset]  = prim;
						++offset;
					}
					return count;
				}
			}
		};

		// Unsorted (finest voxel, primitive) pairs of every top-level pair; padding uses key
		// `total_entries`.
		template <typename P, CellOverlapMode Mode, typename Policy>
		void BuildFinestPairs(const SceneView								 &scene,
							  const thrust::device_vector<uint32_t>			 &owning_top_cell,
							  const uint32_t								 *top_prim_indices,
							  const thrust::device_vector<IrregularCellInfo> &infos,
							  uint32_t										  total_top_level_pairs,
							  uint32_t										  total_entries,
							  thrust::device_vector<uint32_t>				 *pair_voxel_indices,
							  thrust::device_vector<uint32_t>				 *pair_prim_indices,
							  const Policy									 &policy)
		{
			const auto						first = thrust::counting_iterator<uint32_t>(0);
			thrust::device_vector<uint32_t> counts(total_top_level_pairs);
			thrust::transform(policy,
							  first,
							  first + total_top_level_pairs,
							  counts.begin(),
							  Fragments<P, Mode, false>{scene,
														thrust::raw_pointer_cast(owning_top_cell.data()),
														top_prim_indices,
														thrust::raw_pointer_cast(infos.data())});

			thrust::device_vector<uint32_t> write_offsets(total_top_level_pairs);
			thrust::exclusive_scan(policy, counts.begin(), counts.end(), write_offsets.begin());
			const uint32_t total_pairs = (total_top_level_pairs > 0) ? (write_offsets.back() + counts.back()) : 0u;

			pair_voxel_indices->resize(total_pairs);
			pair_prim_indices->resize(total_pairs);
			thrust::for_each(policy,
							 first,
							 first + total_top_level_pairs,
							 Fragments<P, Mode, true>{scene,
													  thrust::raw_pointer_cast(owning_top_cell.data()),
													  top_prim_indices,
													  thrust::raw_pointer_cast(infos.data()),
													  thrust::raw_pointer_cast(write_offsets.data()),
													  thrust::raw_pointer_cast(counts.data()),
													  total_entries,
													  thrust::raw_pointer_cast(pair_voxel_indices->data()),
													  thrust::raw_pointer_cast(pair_prim_indices->data())});
		}

		// 1x1x1-voxel pre-merge cell for finest voxel `v`, in shared-lattice coordinates.
		struct MakeInitialCellFunctor
		{
			const IrregularCellInfo *infos;
			const uint32_t			*owner;
			const uint32_t			*voxel_cell_start;

			HAPPYRAY_HD IrregularCell operator()(uint32_t v) const
			{
				const uint32_t			 c		 = owner[v];
				const IrregularCellInfo &info	 = infos[c];
				const uint32_t			 local_v = v - info.entries_offset;
				const int				 res_xy	 = info.resolution.x * info.resolution.y;
				const int				 cz		 = static_cast<int>(local_v) / res_xy;
				const int				 rem	 = static_cast<int>(local_v) % res_xy;
				const int				 cy		 = rem / info.resolution.x;
				const int				 cx		 = rem % info.resolution.x;
				const Vec3i				 min(info.global_offset.x + (cx << info.shift.x),
											 info.global_offset.y + (cy << info.shift.y),
											 info.global_offset.z + (cz << info.shift.z));
				const Vec3i max(min.x + (1 << info.shift.x), min.y + (1 << info.shift.y), min.z + (1 << info.shift.z));
				const int	min_arr[3] = {min.x, min.y, min.z};
				const int	max_arr[3] = {max.x, max.y, max.z};
				return MakeCell(min_arr, max_arr, voxel_cell_start[v], voxel_cell_start[v + 1]);
			}
		};

		// Phase B: SAH-gated per-axis merge rounds (HAGrid's merge.cu).

		// Shared-lattice position -> index of the cell that claims it. Returns false outside the grid
		// or if `pos` isn't on a voxel boundary of the top-level cell it falls in (possible when it
		// came from a differently-scaled neighbor); callers then decline the merge/expand.
		HAPPYRAY_HD inline bool LookupGlobalCell(const IrregularCellInfo *cell_infos,
												 const uint32_t			 *entries,
												 const int				  pos[3],
												 const Vec3i			 &global_shift,
												 const Vec3i			 &top_resolution,
												 uint32_t				 *out_index)
		{
			const int global_shift_axes[3]	 = {global_shift.x, global_shift.y, global_shift.z};
			const int top_resolution_axes[3] = {top_resolution.x, top_resolution.y, top_resolution.z};
			int		  top_coord[3];
			for (int a = 0; a < 3; ++a)
			{
				if (pos[a] < 0)
				{
					return false;
				}
				top_coord[a] = pos[a] >> global_shift_axes[a];
				if (top_coord[a] >= top_resolution_axes[a])
				{
					return false;
				}
			}
			const uint32_t top_flat =
				FlattenCellIndex(top_coord[0], top_coord[1], top_coord[2], top_resolution.x, top_resolution.y);
			const IrregularCellInfo &info		   = cell_infos[top_flat];
			const int				 shift[3]	   = {info.shift.x, info.shift.y, info.shift.z};
			const int				 offset[3]	   = {info.global_offset.x, info.global_offset.y, info.global_offset.z};
			const int				 resolution[3] = {info.resolution.x, info.resolution.y, info.resolution.z};
			int						 local[3];
			for (int a = 0; a < 3; ++a)
			{
				const int rel = pos[a] - offset[a];
				if (rel & ((1 << shift[a]) - 1))
				{
					return false;
				}
				local[a] = rel >> shift[a];
				if (local[a] < 0 || local[a] >= resolution[a])
				{
					return false;
				}
			}
			const uint32_t voxel = FlattenCellIndex(local[0], local[1], local[2], resolution[0], resolution[1]);
			*out_index			 = entries[info.entries_offset + voxel];
			return true;
		}

		struct MergeCandidate
		{
			int32_t	 next_cell = -1; // >= 0 if this cell wants to (and may) merge with that neighbor.
			uint32_t ref_count = 0;	 // This cell's reference count after merging (or its own, if not).
		};

		struct MergeScratch
		{
			thrust::device_vector<MergeCandidate> candidates;
			thrust::device_vector<int32_t>		  prev;
			thrust::device_vector<uint32_t>		  cell_flags, ref_counts, cell_scan, ref_scan, remap_table;
			thrust::device_vector<IrregularCell>  new_cells, expanded_cells;
			thrust::device_vector<uint32_t>		  new_triangle_indices;
		};

		// Decides whether cell `i` and its +axis neighbor are aligned and merging doesn't raise SAH
		// cost (merge.cu's test, with its early exit before computing the reference union).
		struct MergeCandidateFunctor
		{
			const IrregularCell		*cells;
			const IrregularCellInfo *cell_infos;
			const uint32_t			*entries;
			const uint32_t			*prim_indices;
			int						 axis;
			Vec3f					 global_cell_size;
			Vec3i					 global_shift;
			Vec3i					 top_resolution;

			HAPPYRAY_HD MergeCandidate operator()(uint32_t i) const
			{
				const IrregularCell cell = cells[i];
				int					cell_min[3], cell_max[3];
				CellMinArr(cell, cell_min);
				CellMaxArr(cell, cell_max);
				const uint32_t own_count = cell.references.end - cell.references.start;

				MergeCandidate result;
				result.ref_count = own_count;

				int next_pos[3] = {cell_min[0], cell_min[1], cell_min[2]};
				next_pos[axis]	= cell_max[axis];
				uint32_t next_index;
				if (!LookupGlobalCell(cell_infos, entries, next_pos, global_shift, top_resolution, &next_index))
				{
					return result; // Scene edge, or an incompatibly-scaled neighbor.
				}

				const IrregularCell next_cell = cells[next_index];
				int					next_min[3], next_max[3];
				CellMinArr(next_cell, next_min);
				CellMaxArr(next_cell, next_max);

				// HAGrid's `aligned<axis>`: exact equality on the axes the merge doesn't grow along.
				const int  axis1   = (axis + 1) % 3;
				const int  axis2   = (axis + 2) % 3;
				const bool aligned = next_min[axis] == cell_max[axis] && next_min[axis1] == cell_min[axis1] &&
									 next_min[axis2] == cell_min[axis2] && next_max[axis1] == cell_max[axis1] &&
									 next_max[axis2] == cell_max[axis2];
				if (!aligned)
				{
					return result;
				}

				const uint32_t next_count = next_cell.references.end - next_cell.references.start;
				const Vec3f	   extent1((cell_max[0] - cell_min[0]) * global_cell_size.x,
									   (cell_max[1] - cell_min[1]) * global_cell_size.y,
									   (cell_max[2] - cell_min[2]) * global_cell_size.z);
				const Vec3f	   extent2((next_max[0] - next_min[0]) * global_cell_size.x,
									   (next_max[1] - next_min[1]) * global_cell_size.y,
									   (next_max[2] - next_min[2]) * global_cell_size.z);
				int			   union_min[3], union_max[3];
				for (int a = 0; a < 3; ++a)
				{
					union_min[a] = cell_min[a] < next_min[a] ? cell_min[a] : next_min[a];
					union_max[a] = cell_max[a] > next_max[a] ? cell_max[a] : next_max[a];
				}
				const Vec3f union_extent((union_max[0] - union_min[0]) * global_cell_size.x,
										 (union_max[1] - union_min[1]) * global_cell_size.y,
										 (union_max[2] - union_min[2]) * global_cell_size.z);

				constexpr float kUnitCost  = 1.0f;
				const float		area1	   = HalfSurfaceArea(extent1);
				const float		area2	   = HalfSurfaceArea(extent2);
				const float		area_union = HalfSurfaceArea(union_extent);
				const float		cost1	   = area1 * (static_cast<float>(own_count) + kUnitCost);
				const float		cost2	   = area2 * (static_cast<float>(next_count) + kUnitCost);
				const float		max_count =
					own_count > next_count ? static_cast<float>(own_count) : static_cast<float>(next_count);
				if (area_union * (max_count + kUnitCost) > cost1 + cost2)
				{
					return result; // Early-exit: not even the best case could be worth it.
				}

				const uint32_t union_count = CountSortedUnion(prim_indices + cell.references.start,
															  own_count,
															  prim_indices + next_cell.references.start,
															  next_count);
				const float	   cost_union  = area_union * (static_cast<float>(union_count) + kUnitCost);
				if (cost_union <= cost1 + cost2)
				{
					result.next_cell = static_cast<int32_t>(next_index);
					result.ref_count = union_count;
				}
				return result;
			}
		};

		struct ScatterPrevFunctor
		{
			const MergeCandidate *candidates;
			int32_t				 *prev;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				const int32_t next = candidates[i].next_cell;
				if (next >= 0)
				{
					prev[next] = static_cast<int32_t>(i);
				}
			}
		};

		// Flags even chain positions as kept; odd ones are absorbed into their predecessor. Chains are
		// disjoint, so they parallelize freely.
		struct ChainFlagFunctor
		{
			const MergeCandidate *candidates;
			const int32_t		 *prev;
			uint32_t			 *cell_flags;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				if (prev[i] >= 0)
				{
					return; // Not a chain head.
				}
				int32_t cur	   = static_cast<int32_t>(i);
				int		parity = 0;
				for (;;)
				{
					cell_flags[cur]	   = (parity % 2 == 0) ? 1u : 0u;
					const int32_t next = candidates[cur].next_cell;
					if (next < 0)
					{
						break;
					}
					cur = next;
					++parity;
				}
			}
		};

		struct RefCountForKeptFunctor
		{
			const MergeCandidate *candidates;
			const uint32_t		 *cell_flags;

			HAPPYRAY_HD uint32_t operator()(uint32_t i) const
			{
				return cell_flags[i] ? candidates[i].ref_count : 0u;
			}
		};

		struct WriteMergedFunctor
		{
			const IrregularCell	 *cells;
			const MergeCandidate *candidates;
			const uint32_t		 *cell_flags;
			const uint32_t		 *cell_scan;
			const uint32_t		 *ref_scan;
			const uint32_t		 *prim_indices;
			IrregularCell		 *new_cells;
			uint32_t			 *new_triangle_indices;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				if (!cell_flags[i])
				{
					return;
				}
				const uint32_t		new_index	  = cell_scan[i];
				const uint32_t		new_ref_start = ref_scan[i];
				const IrregularCell cell		  = cells[i];
				const int32_t		next		  = candidates[i].next_cell;

				IrregularCell result;
				if (next >= 0)
				{
					const IrregularCell next_cell = cells[next];
					int					min_arr[3], max_arr[3], next_min[3], next_max[3];
					CellMinArr(cell, min_arr);
					CellMaxArr(cell, max_arr);
					CellMinArr(next_cell, next_min);
					CellMaxArr(next_cell, next_max);
					for (int a = 0; a < 3; ++a)
					{
						min_arr[a] = min_arr[a] < next_min[a] ? min_arr[a] : next_min[a];
						max_arr[a] = max_arr[a] > next_max[a] ? max_arr[a] : next_max[a];
					}
					const uint32_t count = MergeSortedUnique(prim_indices + cell.references.start,
															 cell.references.end - cell.references.start,
															 prim_indices + next_cell.references.start,
															 next_cell.references.end - next_cell.references.start,
															 new_triangle_indices + new_ref_start);
					result				 = MakeCell(min_arr, max_arr, new_ref_start, new_ref_start + count);
				}
				else
				{
					const uint32_t count = cell.references.end - cell.references.start;
					for (uint32_t k = 0; k < count; ++k)
					{
						new_triangle_indices[new_ref_start + k] = prim_indices[cell.references.start + k];
					}
					int min_arr[3], max_arr[3];
					CellMinArr(cell, min_arr);
					CellMaxArr(cell, max_arr);
					result = MakeCell(min_arr, max_arr, new_ref_start, new_ref_start + count);
				}
				new_cells[new_index] = result;
			}
		};

		struct ComputeRemapFunctor
		{
			const uint32_t *cell_flags;
			const uint32_t *cell_scan;
			const int32_t  *prev;

			HAPPYRAY_HD uint32_t operator()(uint32_t i) const
			{
				return cell_flags[i] ? cell_scan[i] : cell_scan[prev[i]];
			}
		};

		struct ApplyRemapFunctor
		{
			const uint32_t *remap_table;

			HAPPYRAY_HD uint32_t operator()(uint32_t old_index) const
			{
				return remap_table[old_index];
			}
		};

		// One merge round along `axis`; updates `cells`/`prim_indices` (swapped) and `entries`.
		template <typename Policy>
		void MergeRound(const Policy								   &policy,
						MergeScratch								   &scratch,
						int												axis,
						const thrust::device_vector<IrregularCellInfo> &cell_infos,
						thrust::device_vector<IrregularCell>		   *cells,
						thrust::device_vector<uint32_t>				   *entries,
						thrust::device_vector<uint32_t>				   *prim_indices,
						const Vec3f									   &global_cell_size,
						const Vec3i									   &global_shift,
						const Vec3i									   &top_resolution)
		{
			const uint32_t n = static_cast<uint32_t>(cells->size());
			if (n == 0)
			{
				return;
			}

			Reserve(&scratch.candidates, n);
			Reserve(&scratch.prev, n);
			Reserve(&scratch.cell_flags, n);
			Reserve(&scratch.ref_counts, n);
			Reserve(&scratch.cell_scan, n);
			Reserve(&scratch.ref_scan, n);
			Reserve(&scratch.remap_table, n);
			const auto first = thrust::counting_iterator<uint32_t>(0);
			thrust::transform(policy,
							  first,
							  first + n,
							  scratch.candidates.begin(),
							  MergeCandidateFunctor{thrust::raw_pointer_cast(cells->data()),
													thrust::raw_pointer_cast(cell_infos.data()),
													thrust::raw_pointer_cast(entries->data()),
													thrust::raw_pointer_cast(prim_indices->data()),
													axis,
													global_cell_size,
													global_shift,
													top_resolution});

			thrust::fill(policy, scratch.prev.begin(), scratch.prev.begin() + n, -1);
			thrust::for_each(policy,
							 first,
							 first + n,
							 ScatterPrevFunctor{thrust::raw_pointer_cast(scratch.candidates.data()),
												thrust::raw_pointer_cast(scratch.prev.data())});
			thrust::for_each(policy,
							 first,
							 first + n,
							 ChainFlagFunctor{thrust::raw_pointer_cast(scratch.candidates.data()),
											  thrust::raw_pointer_cast(scratch.prev.data()),
											  thrust::raw_pointer_cast(scratch.cell_flags.data())});
			thrust::transform(policy,
							  first,
							  first + n,
							  scratch.ref_counts.begin(),
							  RefCountForKeptFunctor{thrust::raw_pointer_cast(scratch.candidates.data()),
													 thrust::raw_pointer_cast(scratch.cell_flags.data())});

			thrust::exclusive_scan(policy,
								   scratch.cell_flags.begin(),
								   scratch.cell_flags.begin() + n,
								   scratch.cell_scan.begin());
			const uint32_t new_num_cells = scratch.cell_scan[n - 1] + scratch.cell_flags[n - 1];
			thrust::exclusive_scan(policy,
								   scratch.ref_counts.begin(),
								   scratch.ref_counts.begin() + n,
								   scratch.ref_scan.begin());
			const uint32_t new_num_refs = scratch.ref_scan[n - 1] + scratch.ref_counts[n - 1];

			Reserve(&scratch.new_cells, new_num_cells);
			scratch.new_cells.resize(new_num_cells);
			Reserve(&scratch.new_triangle_indices, new_num_refs);
			scratch.new_triangle_indices.resize(new_num_refs);
			thrust::for_each(policy,
							 first,
							 first + n,
							 WriteMergedFunctor{thrust::raw_pointer_cast(cells->data()),
												thrust::raw_pointer_cast(scratch.candidates.data()),
												thrust::raw_pointer_cast(scratch.cell_flags.data()),
												thrust::raw_pointer_cast(scratch.cell_scan.data()),
												thrust::raw_pointer_cast(scratch.ref_scan.data()),
												thrust::raw_pointer_cast(prim_indices->data()),
												thrust::raw_pointer_cast(scratch.new_cells.data()),
												thrust::raw_pointer_cast(scratch.new_triangle_indices.data())});

			thrust::transform(policy,
							  first,
							  first + n,
							  scratch.remap_table.begin(),
							  ComputeRemapFunctor{thrust::raw_pointer_cast(scratch.cell_flags.data()),
												  thrust::raw_pointer_cast(scratch.cell_scan.data()),
												  thrust::raw_pointer_cast(scratch.prev.data())});
			thrust::transform(policy,
							  entries->begin(),
							  entries->end(),
							  entries->begin(),
							  ApplyRemapFunctor{thrust::raw_pointer_cast(scratch.remap_table.data())});

			cells->swap(scratch.new_cells);
			prim_indices->swap(scratch.new_triangle_indices);
		}
		// Phase C: subset-safe expand passes (HAGrid's expand.cu). Only bounds change.

		struct ExpandFunctor
		{
			const IrregularCell		*cells;
			const IrregularCellInfo *cell_infos;
			const uint32_t			*entries;
			const uint32_t			*prim_indices;
			int						 axis;
			Vec3i					 global_shift;
			Vec3i					 top_resolution;

			HAPPYRAY_HD IrregularCell ExpandDirection(const IrregularCell &cell, int dir) const
			{
				int cell_min[3], cell_max[3];
				CellMinArr(cell, cell_min);
				CellMaxArr(cell, cell_max);

				int next_pos[3] = {cell_min[0], cell_min[1], cell_min[2]};
				next_pos[axis]	= dir > 0 ? cell_max[axis] : cell_min[axis] - 1;
				uint32_t next_index;
				if (!LookupGlobalCell(cell_infos, entries, next_pos, global_shift, top_resolution, &next_index))
				{
					return cell; // Scene edge, or an incompatibly-scaled neighbor.
				}
				const IrregularCell next_cell = cells[next_index];
				int					next_min[3], next_max[3];
				CellMinArr(next_cell, next_min);
				CellMaxArr(next_cell, next_max);

				const int  axis1   = (axis + 1) % 3;
				const int  axis2   = (axis + 2) % 3;
				const bool aligned = (dir > 0 ? next_min[axis] == cell_max[axis] : next_max[axis] == cell_min[axis]) &&
									 next_min[axis1] == cell_min[axis1] && next_max[axis1] == cell_max[axis1] &&
									 next_min[axis2] == cell_min[axis2] && next_max[axis2] == cell_max[axis2];
				if (!aligned)
				{
					return cell; // Only grow into a single neighbor that spans the whole shared face.
				}

				const uint32_t own_count  = cell.references.end - cell.references.start;
				const uint32_t next_count = next_cell.references.end - next_cell.references.start;
				if (!IsSubsetOf(prim_indices + next_cell.references.start,
								next_count,
								prim_indices + cell.references.start,
								own_count))
				{
					return cell; // Growing could miss a triangle only `next` currently knows about.
				}

				if (dir > 0)
				{
					cell_max[axis] = next_max[axis];
				}
				else
				{
					cell_min[axis] = next_min[axis];
				}
				return MakeCell(cell_min, cell_max, cell.references.start, cell.references.end);
			}

			HAPPYRAY_HD IrregularCell operator()(uint32_t i) const
			{
				const IrregularCell cell	 = cells[i];
				const IrregularCell negative = ExpandDirection(cell, -1);
				const IrregularCell positive = ExpandDirection(cell, 1);
				int					cell_min[3], cell_max[3];
				CellMinArr(negative, cell_min);
				CellMaxArr(positive, cell_max);
				return MakeCell(cell_min, cell_max, cell.references.start, cell.references.end);
			}
		};

		// One SAH term per distinct cell, since merge/expand makes cell sizes differ.
		struct IrregularGridSahFunctor
		{
			IrregularGridView grid;
			SahCostConstants  costs;

			HAPPYRAY_HD float operator()(uint32_t i) const
			{
				const IrregularCell &cell = grid.local_cells[i];
				const Vec3f			 extent((cell.max.x - cell.min.x) * grid.global_cell_size.x,
											(cell.max.y - cell.min.y) * grid.global_cell_size.y,
											(cell.max.z - cell.min.z) * grid.global_cell_size.z);
				const uint32_t		 ref_count = cell.references.end - cell.references.start;
				return HalfSurfaceArea(extent) * (costs.traversal + static_cast<float>(ref_count) * costs.intersection);
			}
		};

	} // namespace

	struct MakeTopCellFunctor
	{
		HAPPYRAY_HD IrregularTopCell operator()(const IrregularCellInfo &info) const
		{
			return MakeIrregularTopCell(info);
		}
	};

	struct IrregularGrid::Impl
	{
		CachingAllocator						 allocator;
		MergeScratch							 merge_scratch;
		UniformGrid								 top_level;
		thrust::device_vector<IrregularCellInfo> cell_infos;
		thrust::device_vector<IrregularTopCell>	 top_cells;
		thrust::device_vector<uint32_t>			 entries;
		thrust::device_vector<IrregularCell>	 cells;
		thrust::device_vector<uint32_t>			 prim_indices;
		Vec3f									 global_cell_size;
		Vec3i									 global_resolution = Vec3i(1, 1, 1);

		Impl(const SceneView &scene, float top_level_density, CellOverlapMode mode)
			: top_level(scene, top_level_density, mode)
		{
		}

		IrregularGridView View() const
		{
			IrregularGridView view;
			view.top_level		   = top_level.View();
			view.cell_infos		   = thrust::raw_pointer_cast(cell_infos.data());
			view.top_cells		   = thrust::raw_pointer_cast(top_cells.data());
			view.local_entries	   = thrust::raw_pointer_cast(entries.data());
			view.local_cells	   = thrust::raw_pointer_cast(cells.data());
			view.prim_indices	   = thrust::raw_pointer_cast(prim_indices.data());
			view.global_cell_size  = global_cell_size;
			view.global_resolution = global_resolution;
			view.global_inv_cell_size =
				Vec3f(1.0f / global_cell_size.x, 1.0f / global_cell_size.y, 1.0f / global_cell_size.z);
			view.global_shift =
				Vec3i(Log2Ceil(global_resolution.x), Log2Ceil(global_resolution.y), Log2Ceil(global_resolution.z));
			return view;
		}
	};

	IrregularGrid::IrregularGrid(const SceneView &scene,
								 float			  top_level_density,
								 float			  second_level_density,
								 uint32_t		  dense_cell_threshold,
								 float			  merge_alpha,
								 int			  expand_iterations,
								 CellOverlapMode  mode)
		: impl_(std::make_unique<Impl>(scene, top_level_density, mode))
	{
		const auto			  policy   = thrust::device(impl_->allocator);
		const UniformGridView top_view = impl_->top_level.View();
		const uint32_t		  num_top_cells =
			static_cast<uint32_t>(top_view.resolution.x) * top_view.resolution.y * top_view.resolution.z;
		const uint32_t total_top_level_pairs = impl_->top_level.PrimIndexCount();

		// Phase A.
		thrust::device_vector<IrregularCellInfo> infos(num_top_cells);
		thrust::transform(policy,
						  thrust::counting_iterator<uint32_t>(0),
						  thrust::counting_iterator<uint32_t>(num_top_cells),
						  infos.begin(),
						  ComputeCellInfoFunctor{top_view.bounds_min,
												 top_view.cell_size,
												 Vec2i(top_view.resolution.x, top_view.resolution.y),
												 top_view.cell_ranges,
												 second_level_density,
												 dense_cell_threshold});

		// All top-level cells are the same size, so the per-axis maximum power-of-2 resolution is
		// divisible by every cell's resolution.
		Vec3i global_resolution(1, 1, 1);
		if (num_top_cells > 0)
		{
			const thrust::host_vector<IrregularCellInfo> host_infos = infos;
			for (const IrregularCellInfo &info : host_infos)
			{
				global_resolution.x = std::max(global_resolution.x, info.resolution.x);
				global_resolution.y = std::max(global_resolution.y, info.resolution.y);
				global_resolution.z = std::max(global_resolution.z, info.resolution.z);
			}
		}
		impl_->global_cell_size	 = Vec3f(top_view.cell_size.x / global_resolution.x,
										 top_view.cell_size.y / global_resolution.y,
										 top_view.cell_size.z / global_resolution.z);
		impl_->global_resolution = global_resolution;
		const Vec3i global_shift(Log2Ceil(global_resolution.x),
								 Log2Ceil(global_resolution.y),
								 Log2Ceil(global_resolution.z));
		if (num_top_cells > 0)
		{
			thrust::transform(
				policy,
				thrust::counting_iterator<uint32_t>(0),
				thrust::counting_iterator<uint32_t>(num_top_cells),
				infos.begin(),
				infos.begin(),
				SetGlobalScaleFunctor{Vec2i(top_view.resolution.x, top_view.resolution.y), global_resolution});
		}

		thrust::device_vector<uint32_t> num_voxels_per_cell(num_top_cells);
		thrust::transform(policy, infos.begin(), infos.end(), num_voxels_per_cell.begin(), ExtractNumVoxelsFunctor());

		thrust::device_vector<uint32_t> entries_offset_bounds(num_top_cells + 1, 0u);
		if (num_top_cells > 0)
		{
			thrust::exclusive_scan(policy,
								   num_voxels_per_cell.begin(),
								   num_voxels_per_cell.end(),
								   entries_offset_bounds.begin());
			entries_offset_bounds[num_top_cells] =
				entries_offset_bounds[num_top_cells - 1] + num_voxels_per_cell[num_top_cells - 1];
		}
		const uint32_t total_entries =
			(num_top_cells > 0) ? static_cast<uint32_t>(entries_offset_bounds[num_top_cells]) : 0u;

		thrust::transform(policy,
						  infos.begin(),
						  infos.end(),
						  entries_offset_bounds.begin(),
						  infos.begin(),
						  SetEntriesOffsetFunctor());

		// Owning top-level cell of each (cell, triangle) pair and of each finest voxel.
		thrust::device_vector<uint32_t> owning_top_cell(total_top_level_pairs);
		thrust::for_each(
			policy,
			thrust::counting_iterator<uint32_t>(0),
			thrust::counting_iterator<uint32_t>(num_top_cells),
			ScatterCellRangeOwnerFunctor{top_view.cell_ranges, thrust::raw_pointer_cast(owning_top_cell.data())});

		thrust::device_vector<uint32_t> voxel_owner(total_entries);
		thrust::for_each(policy,
						 thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(num_top_cells),
						 ScatterOwnerFunctor{thrust::raw_pointer_cast(entries_offset_bounds.data()),
											 thrust::raw_pointer_cast(entries_offset_bounds.data()) + 1,
											 thrust::raw_pointer_cast(voxel_owner.data())});

		thrust::device_vector<uint32_t> pair_voxel_indices;
		thrust::device_vector<uint32_t> pair_prim_indices;
		detail::DispatchBuild(scene.primitive_type,
							  mode,
							  [&](auto prim, auto overlap)
							  {
								  BuildFinestPairs<typename decltype(prim)::type, decltype(overlap)::value>(
									  scene,
									  owning_top_cell,
									  top_view.prim_indices,
									  infos,
									  total_top_level_pairs,
									  total_entries,
									  &pair_voxel_indices,
									  &pair_prim_indices,
									  policy);
							  });
		const uint32_t total_initial_pairs = static_cast<uint32_t>(pair_voxel_indices.size());

		thrust::sort_by_key(policy, pair_voxel_indices.begin(), pair_voxel_indices.end(), pair_prim_indices.begin());

		const uint32_t valid_pairs = static_cast<uint32_t>(
			thrust::lower_bound(policy, pair_voxel_indices.begin(), pair_voxel_indices.end(), total_entries) -
			pair_voxel_indices.begin());
		pair_prim_indices.resize(valid_pairs);
		if (valid_pairs < total_initial_pairs)
			pair_prim_indices.shrink_to_fit();
		thrust::device_vector<uint32_t> voxel_cell_start(total_entries + 1, valid_pairs);
		thrust::for_each(policy,
						 thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(valid_pairs),
						 detail::CellExtractor{thrust::raw_pointer_cast(pair_voxel_indices.data()),
											   thrust::raw_pointer_cast(voxel_cell_start.data())});
		thrust::inclusive_scan(policy,
							   voxel_cell_start.rbegin(),
							   voxel_cell_start.rend(),
							   voxel_cell_start.rbegin(),
							   thrust::minimum<uint32_t>());

		thrust::device_vector<IrregularCell> cells(total_entries);
		thrust::transform(policy,
						  thrust::counting_iterator<uint32_t>(0),
						  thrust::counting_iterator<uint32_t>(total_entries),
						  cells.begin(),
						  MakeInitialCellFunctor{thrust::raw_pointer_cast(infos.data()),
												 thrust::raw_pointer_cast(voxel_owner.data()),
												 thrust::raw_pointer_cast(voxel_cell_start.data())});

		thrust::device_vector<uint32_t> entries(total_entries);
		thrust::sequence(policy, entries.begin(), entries.end());

		thrust::device_vector<uint32_t> prim_indices = std::move(pair_prim_indices);

		// Phase B.
		if (merge_alpha > 0.0f && num_top_cells > 0)
		{
			const int max_global_resolution = std::max({global_resolution.x, global_resolution.y, global_resolution.z});
			const int max_rounds			= Log2Ceil(max_global_resolution) + 1;

			uint32_t prev_count = static_cast<uint32_t>(cells.size());
			for (int round = 0; round < max_rounds; ++round)
			{
				MergeRound(policy,
						   impl_->merge_scratch,
						   0,
						   infos,
						   &cells,
						   &entries,
						   &prim_indices,
						   impl_->global_cell_size,
						   global_shift,
						   top_view.resolution);
				MergeRound(policy,
						   impl_->merge_scratch,
						   1,
						   infos,
						   &cells,
						   &entries,
						   &prim_indices,
						   impl_->global_cell_size,
						   global_shift,
						   top_view.resolution);
				MergeRound(policy,
						   impl_->merge_scratch,
						   2,
						   infos,
						   &cells,
						   &entries,
						   &prim_indices,
						   impl_->global_cell_size,
						   global_shift,
						   top_view.resolution);
				const uint32_t new_count = static_cast<uint32_t>(cells.size());
				if (new_count >= merge_alpha * prev_count)
				{
					break;
				}
				prev_count = new_count;
			}
		}

		// Phase C.
		for (int iter = 0; iter < expand_iterations; ++iter)
		{
			for (int axis = 0; axis < 3; ++axis)
			{
				const uint32_t n = static_cast<uint32_t>(cells.size());
				if (n == 0)
				{
					continue;
				}
				Reserve(&impl_->merge_scratch.expanded_cells, n);
				impl_->merge_scratch.expanded_cells.resize(n);
				thrust::transform(policy,
								  thrust::counting_iterator<uint32_t>(0),
								  thrust::counting_iterator<uint32_t>(n),
								  impl_->merge_scratch.expanded_cells.begin(),
								  ExpandFunctor{thrust::raw_pointer_cast(cells.data()),
												thrust::raw_pointer_cast(infos.data()),
												thrust::raw_pointer_cast(entries.data()),
												thrust::raw_pointer_cast(prim_indices.data()),
												axis,
												global_shift,
												top_view.resolution});
				cells.swap(impl_->merge_scratch.expanded_cells);
			}
		}

		impl_->top_cells.resize(num_top_cells);
		thrust::transform(policy, infos.begin(), infos.end(), impl_->top_cells.begin(), MakeTopCellFunctor());
		impl_->cell_infos	= std::move(infos);
		impl_->entries		= std::move(entries);
		impl_->cells		= std::move(cells);
		impl_->prim_indices = std::move(prim_indices);
	}

	IrregularGrid::~IrregularGrid() = default;

	uint32_t IrregularGrid::TopLevelPrimIndexCount() const
	{
		return impl_->top_level.PrimIndexCount();
	}

	uint32_t IrregularGrid::LocalEntryCount() const
	{
		return static_cast<uint32_t>(impl_->entries.size());
	}

	uint32_t IrregularGrid::LeafCellCount() const
	{
		return static_cast<uint32_t>(impl_->cells.size());
	}

	uint32_t IrregularGrid::PrimIndexCount() const
	{
		return static_cast<uint32_t>(impl_->prim_indices.size());
	}

	float IrregularGrid::ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const
	{
		const IrregularGridView view = impl_->View();
		// Traversal walks the shared lattice directly, so top-level cells cost no steps.
		float		   total		   = 0.0f;
		const uint32_t num_local_cells = LeafCellCount();
		if (num_local_cells > 0)
		{
			total += thrust::transform_reduce(thrust::counting_iterator<uint32_t>(0),
											  thrust::counting_iterator<uint32_t>(num_local_cells),
											  IrregularGridSahFunctor{view, costs},
											  0.0f,
											  thrust::plus<float>());
		}
		return total / HalfSurfaceArea(scene_bounds.Extent());
	}

	IrregularGridView IrregularGrid::View() const
	{
		return impl_->View();
	}

} // namespace happyray
