// Thrust implementation of CollisionDetector (see collision_detector.h), following
// string2shape's CollisionDetector::computeCollisionGraph: trim single-object cells, compact
// the remaining ones, then run primitive bounding-box overlap tests per cell reference. Compiles under
// either the CUDA or the TBB Thrust backend, the same as the grid builders.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "collision_detector.h"

#include <chrono>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <type_traits>
#include <utility>

#include "../core/primitive_traits.h"

namespace happyray
{
	namespace
	{

		HAPPYRAY_HD inline uint32_t LoadCollisionId(const uint32_t *address)
		{
#ifdef __CUDA_ARCH__
			return *static_cast<const volatile uint32_t *>(address);
#else
			return __atomic_load_n(address, __ATOMIC_RELAXED);
#endif
		}

		// Keeps the smallest colliding object id, so the result doesn't depend on
		// which cell/thread records a collision first.
		HAPPYRAY_HD inline void AtomicMinCollisionId(uint32_t *address, uint32_t value)
		{
#ifdef __CUDA_ARCH__
			atomicMin(address, value);
#else
			uint32_t current = __atomic_load_n(address, __ATOMIC_RELAXED);
			while (value < current &&
				   !__atomic_compare_exchange_n(address, &current, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
			{
			}
#endif
		}

		struct NonEmptyRange
		{
			HAPPYRAY_HD bool operator()(const CellRange &cell_range) const
			{
				return cell_range.start < cell_range.end;
			}
		};

		// Replaces cells that cannot contain a collision (fewer than two primitives, or
		// all primitives from the same object) with an empty range.
		struct CellTrimmer
		{
			const uint32_t *object_ids;
			const uint32_t *prim_indices;

			HAPPYRAY_HD CellRange operator()(const CellRange &cell_range) const
			{
				if (cell_range.start + 1 >= cell_range.end)
					return CellRange{}; // Zero or one primitive inside => no collision.

				for (uint32_t ref = cell_range.start; ref + 1 < cell_range.end; ++ref)
				{
					if (object_ids[prim_indices[ref]] != object_ids[prim_indices[ref + 1]])
						return cell_range;
				}
				return CellRange{};
			}

			HAPPYRAY_HD CellRange operator()(const IrregularCell &cell) const
			{
				return (*this)(cell.references);
			}
		};

		struct CellSize
		{
			HAPPYRAY_HD uint32_t operator()(const CellRange &cell_range) const
			{
				return cell_range.end - cell_range.start;
			}
		};

		template <typename P> struct PrimitiveBoundsFunctor
		{
			SceneView		 scene;
			HAPPYRAY_HD BBox operator()(uint32_t prim) const
			{
				return ComputePrimitiveBBox<P>(scene, prim);
			}
		};

		// Tests all primitive pairs from different objects inside the candidate cells
		// for bounding-box overlap. Unlike string2shape (one thread per cell), each work
		// item is one reference of a candidate cell, tested against the cell's later
		// references: the same pairs, but a cell with thousands of primitives (e.g. a
		// uniform grid cell over dense geometry) no longer serializes O(k^2) tests on a
		// single thread.
		struct CollisionOperator
		{
			const uint32_t	*object_ids;
			const BBox		*boxes;
			const uint32_t	*prim_indices;
			uint32_t		*collision_ids;
			const CellRange *cells;
			const uint32_t	*cell_ref_offsets; // Exclusive scan of the cell sizes.
			uint32_t		 cell_count;

			HAPPYRAY_HD void operator()(uint32_t work_item) const
			{
				// Last cell whose first work item is <= work_item.
				uint32_t lo = 0, hi = cell_count;
				while (hi - lo > 1)
				{
					const uint32_t mid = lo + (hi - lo) / 2;
					if (cell_ref_offsets[mid] <= work_item)
						lo = mid;
					else
						hi = mid;
				}
				const CellRange cell_range = cells[lo];
				const uint32_t	ref		   = cell_range.start + (work_item - cell_ref_offsets[lo]);

				const uint32_t my_prim = prim_indices[ref];
				const uint32_t my_obj  = object_ids[my_prim];
				const BBox	   my_box  = boxes[my_prim];

				for (uint32_t other_ref = ref + 1; other_ref < cell_range.end; ++other_ref)
				{
					const uint32_t other_prim = prim_indices[other_ref];
					const uint32_t other_obj  = object_ids[other_prim];
					if (my_obj == other_obj)
						continue;
					// Per-primitive analog of string2shape's lastRecordedPair: skip the
					// test if its outcome could not lower either recorded id.
					if (LoadCollisionId(collision_ids + my_prim) <= other_obj &&
						LoadCollisionId(collision_ids + other_prim) <= my_obj)
						continue;

					if (Overlaps(my_box, boxes[other_prim]))
					{
						AtomicMinCollisionId(collision_ids + my_prim, other_obj);
						AtomicMinCollisionId(collision_ids + other_prim, my_obj);
					}
				}
			}
		};

		struct IsColliding
		{
			HAPPYRAY_HD bool operator()(uint32_t collision_id) const
			{
				return collision_id != kNoCollision;
			}
		};

	} // namespace

	template <typename Grid>
	CollisionStats CollisionDetector::ComputeCollisions(const SceneView &scene,
														uint32_t		 object_count,
														const Grid		&grid,
														uint32_t		*collision_ids) const
	{
		const auto view						  = grid.View();
		const auto [leaf_cells, prim_indices] = [&]()
		{
			if constexpr (std::is_same_v<Grid, UniformGrid>)
				return std::make_pair(view.cell_ranges, view.prim_indices);
			else if constexpr (std::is_same_v<Grid, TwoLevelGrid>)
				return std::make_pair(view.second_level_cell_ranges, view.second_level_prim_indices);
			else
				return std::make_pair(view.local_cells, view.prim_indices);
		}();
		const auto	   cells	  = thrust::device_pointer_cast(leaf_cells);
		const uint32_t cell_count = grid.LeafCellCount();
		const auto	   start_time = std::chrono::steady_clock::now();
		CollisionStats stats;
		stats.object_count = object_count;

		const thrust::device_ptr<uint32_t> ids = thrust::device_pointer_cast(collision_ids);
		thrust::fill(ids, ids + scene.primitive_count, kNoCollision);

		if (object_count >= 2u && scene.object_ids && cell_count > 0)
		{
			// Delete all grid cells that contain primitives from a single object.
			thrust::device_vector<CellRange> trimmed_cells(cell_count);
			thrust::transform(cells,
							  cells + cell_count,
							  trimmed_cells.begin(),
							  CellTrimmer{scene.object_ids, prim_indices});
			thrust::device_vector<CellRange> candidate_cells(cell_count);
			const auto						 candidate_cells_end =
				thrust::copy_if(trimmed_cells.begin(), trimmed_cells.end(), candidate_cells.begin(), NonEmptyRange());
			stats.candidate_cell_count = static_cast<uint32_t>(candidate_cells_end - candidate_cells.begin());

			if (stats.candidate_cell_count > 0)
			{
				thrust::device_vector<uint32_t> cell_ref_offsets(stats.candidate_cell_count);
				thrust::transform_exclusive_scan(candidate_cells.begin(),
												 candidate_cells_end,
												 cell_ref_offsets.begin(),
												 CellSize(),
												 0u,
												 thrust::plus<uint32_t>());
				const CellRange last_cell		= candidate_cells[stats.candidate_cell_count - 1];
				const uint32_t	work_item_count = cell_ref_offsets.back() + (last_cell.end - last_cell.start);

				thrust::device_vector<BBox> boxes(scene.primitive_count);
				DispatchPrimitive(scene.primitive_type,
								  [&](auto tag)
								  {
									  thrust::transform(thrust::counting_iterator<uint32_t>(0),
														thrust::counting_iterator<uint32_t>(scene.primitive_count),
														boxes.begin(),
														PrimitiveBoundsFunctor<typename decltype(tag)::type>{scene});
								  });
				thrust::for_each(thrust::counting_iterator<uint32_t>(0),
								 thrust::counting_iterator<uint32_t>(work_item_count),
								 CollisionOperator{scene.object_ids,
												   thrust::raw_pointer_cast(boxes.data()),
												   prim_indices,
												   collision_ids,
												   thrust::raw_pointer_cast(candidate_cells.data()),
												   thrust::raw_pointer_cast(cell_ref_offsets.data()),
												   stats.candidate_cell_count});
			}

			stats.colliding_primitive_count =
				static_cast<uint32_t>(thrust::count_if(ids, ids + scene.primitive_count, IsColliding()));
		}

		stats.total_ms =
			std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count();
		return stats;
	}

	template CollisionStats CollisionDetector::ComputeCollisions<UniformGrid>(const SceneView &,
																			  uint32_t,
																			  const UniformGrid &,
																			  uint32_t *) const;
	template CollisionStats CollisionDetector::ComputeCollisions<TwoLevelGrid>(const SceneView &,
																			   uint32_t,
																			   const TwoLevelGrid &,
																			   uint32_t *) const;
	template CollisionStats CollisionDetector::ComputeCollisions<IrregularGrid>(const SceneView &,
																				uint32_t,
																				const IrregularGrid &,
																				uint32_t *) const;

} // namespace happyray
