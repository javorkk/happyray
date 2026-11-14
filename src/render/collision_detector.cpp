// CollisionDetector implementation (string2shape's computeCollisionGraph); the grids are not
// modified.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "collision_detector.h"

#include <chrono>
#include <memory>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <type_traits>
#include <utility>

#include "../core/primitive_traits.h"
#include "../core/thrust_scratch.h"

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

		struct PairTestCounts
		{
			uint64_t tests	  = 0;
			uint64_t overlaps = 0;
		};

		struct AddPairTestCounts
		{
			HAPPYRAY_HD PairTestCounts operator()(const PairTestCounts &a, const PairTestCounts &b) const
			{
				return {a.tests + b.tests, a.overlaps + b.overlaps};
			}
		};

		// Work item w is reference w - cell_ref_offsets[cell] of candidate cell `cell`; keys sort
		// each cell's references by object.
		struct GatherCandidateRefs
		{
			const CellRange *cells;
			const uint32_t	*cell_ref_offsets; // Exclusive scan of the candidate cell sizes.
			uint32_t		 cell_count;
			const uint32_t	*prim_indices;
			const uint32_t	*object_ids;
			uint64_t		*ref_keys;
			uint32_t		*ref_prims;

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
				const uint32_t prim = prim_indices[cells[lo].start + (work_item - cell_ref_offsets[lo])];
				ref_keys[work_item]	 = (static_cast<uint64_t>(lo) << 32) | object_ids[prim];
				ref_prims[work_item] = prim;
			}
		};

		// One work item per cell reference, tested against the cell's later references, so huge
		// cells don't serialize O(k^2) tests on one thread. Starting past the reference's own
		// object run skips same-object pairs.
		struct CollisionOperator
		{
			const BBox	   *boxes;
			const uint64_t *ref_keys;		   // Sorted (candidate cell << 32 | object id).
			const uint32_t *ref_prims;		   // Primitive of each sorted reference.
			const uint32_t *object_run_ends;   // First reference past each reference's (cell, object) run.
			const uint32_t *cell_ref_offsets; // First reference of each candidate cell.
			uint32_t		cell_count;
			uint32_t		ref_count;
			uint32_t	   *collision_ids;

			HAPPYRAY_HD PairTestCounts operator()(uint32_t ref) const
			{
				const uint64_t key		= ref_keys[ref];
				const uint32_t cell		= static_cast<uint32_t>(key >> 32);
				const uint32_t cell_end = cell + 1 < cell_count ? cell_ref_offsets[cell + 1] : ref_count;
				const uint32_t my_obj	= static_cast<uint32_t>(key);
				const uint32_t my_prim	= ref_prims[ref];
				const BBox	   my_box	= boxes[my_prim];
				PairTestCounts counts;

				for (uint32_t other_ref = object_run_ends[ref]; other_ref < cell_end; ++other_ref)
				{
					const uint32_t other_obj  = static_cast<uint32_t>(ref_keys[other_ref]);
					const uint32_t other_prim = ref_prims[other_ref];
					// Skip tests that couldn't lower either recorded id (cf. string2shape's
					// lastRecordedPair).
					if (LoadCollisionId(collision_ids + my_prim) <= other_obj &&
						LoadCollisionId(collision_ids + other_prim) <= my_obj)
						continue;

					++counts.tests;
					if (Overlaps(my_box, boxes[other_prim]))
					{
						++counts.overlaps;
						AtomicMinCollisionId(collision_ids + my_prim, other_obj);
						AtomicMinCollisionId(collision_ids + other_prim, my_obj);
					}
				}
				return counts;
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


	struct CollisionDetector::Scratch
	{
		CachingAllocator				 allocator;
		thrust::device_vector<CellRange> trimmed_cells, candidate_cells;
		thrust::device_vector<uint32_t>	 cell_ref_offsets, ref_prims, object_run_ends;
		thrust::device_vector<uint64_t>	 ref_keys;
		thrust::device_vector<BBox>		 boxes;
	};

	CollisionDetector::CollisionDetector() : scratch_(std::make_unique<Scratch>())
	{
	}

	CollisionDetector::~CollisionDetector()									  = default;
	CollisionDetector::CollisionDetector(CollisionDetector &&) noexcept			  = default;
	CollisionDetector &CollisionDetector::operator=(CollisionDetector &&) noexcept = default;

	template <typename Grid>
	CollisionStats CollisionDetector::ComputeCollisions(const SceneView &scene,
														uint32_t		 object_count,
														const Grid		&grid,
														uint32_t		*collision_ids)
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
		if (!scratch_) // Moved from.
			scratch_ = std::make_unique<Scratch>();
		Scratch	  &scratch = *scratch_;
		const auto policy  = thrust::device(scratch.allocator);

		const thrust::device_ptr<uint32_t> ids = thrust::device_pointer_cast(collision_ids);
		thrust::fill(policy, ids, ids + scene.primitive_count, kNoCollision);

		if (object_count >= 2u && scene.object_ids && cell_count > 0)
		{
			const auto trimmed_cells = thrust::device_pointer_cast(Reserve(&scratch.trimmed_cells, cell_count));
			thrust::transform(
				policy, cells, cells + cell_count, trimmed_cells, CellTrimmer{scene.object_ids, prim_indices});
			const auto candidate_cells = thrust::device_pointer_cast(Reserve(&scratch.candidate_cells, cell_count));
			const auto candidate_cells_end =
				thrust::copy_if(policy, trimmed_cells, trimmed_cells + cell_count, candidate_cells, NonEmptyRange());
			stats.candidate_cell_count = static_cast<uint32_t>(candidate_cells_end - candidate_cells);

			if (stats.candidate_cell_count > 0)
			{
				const auto cell_ref_offsets =
					thrust::device_pointer_cast(Reserve(&scratch.cell_ref_offsets, stats.candidate_cell_count));
				thrust::transform_exclusive_scan(policy,
												 candidate_cells,
												 candidate_cells_end,
												 cell_ref_offsets,
												 CellSize(),
												 0u,
												 thrust::plus<uint32_t>());
				const CellRange last_cell = candidate_cells[stats.candidate_cell_count - 1];
				const uint32_t	ref_count =
					cell_ref_offsets[stats.candidate_cell_count - 1] + (last_cell.end - last_cell.start);
				const auto first = thrust::counting_iterator<uint32_t>(0);

				const auto ref_keys	 = thrust::device_pointer_cast(Reserve(&scratch.ref_keys, ref_count));
				const auto ref_prims = thrust::device_pointer_cast(Reserve(&scratch.ref_prims, ref_count));
				thrust::for_each(policy,
								 first,
								 first + ref_count,
								 GatherCandidateRefs{thrust::raw_pointer_cast(candidate_cells),
													 thrust::raw_pointer_cast(cell_ref_offsets),
													 stats.candidate_cell_count,
													 prim_indices,
													 scene.object_ids,
													 thrust::raw_pointer_cast(ref_keys),
													 thrust::raw_pointer_cast(ref_prims)});
				thrust::sort_by_key(policy, ref_keys, ref_keys + ref_count, ref_prims);
				const auto object_run_ends =
					thrust::device_pointer_cast(Reserve(&scratch.object_run_ends, ref_count));
				thrust::upper_bound(
					policy, ref_keys, ref_keys + ref_count, ref_keys, ref_keys + ref_count, object_run_ends);

				const auto boxes = thrust::device_pointer_cast(Reserve(&scratch.boxes, scene.primitive_count));
				DispatchPrimitive(scene.primitive_type,
								  [&](auto tag)
								  {
									  thrust::transform(policy,
														first,
														first + scene.primitive_count,
														boxes,
														PrimitiveBoundsFunctor<typename decltype(tag)::type>{scene});
								  });
				// The reduction evaluates each work item exactly once, so its side effects are safe.
				const PairTestCounts counts =
					thrust::transform_reduce(policy,
											 first,
											 first + ref_count,
											 CollisionOperator{thrust::raw_pointer_cast(boxes),
															   thrust::raw_pointer_cast(ref_keys),
															   thrust::raw_pointer_cast(ref_prims),
															   thrust::raw_pointer_cast(object_run_ends),
															   thrust::raw_pointer_cast(cell_ref_offsets),
															   stats.candidate_cell_count,
															   ref_count,
															   collision_ids},
											 PairTestCounts{},
											 AddPairTestCounts{});
				stats.candidate_pair_count = counts.tests;
				stats.colliding_pair_count = counts.overlaps;
			}

			stats.colliding_primitive_count =
				static_cast<uint32_t>(thrust::count_if(policy, ids, ids + scene.primitive_count, IsColliding()));
		}

		stats.total_ms =
			std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count();
		return stats;
	}

	template CollisionStats CollisionDetector::ComputeCollisions<UniformGrid>(const SceneView &,
																			  uint32_t,
																			  const UniformGrid &,
																			  uint32_t *);
	template CollisionStats CollisionDetector::ComputeCollisions<TwoLevelGrid>(const SceneView &,
																			   uint32_t,
																			   const TwoLevelGrid &,
																			   uint32_t *);
	template CollisionStats CollisionDetector::ComputeCollisions<IrregularGrid>(const SceneView &,
																				uint32_t,
																				const IrregularGrid &,
																				uint32_t *);

} // namespace happyray
