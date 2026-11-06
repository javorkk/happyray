// Thrust implementation of CollisionDetector (see collision_detector.h), following
// string2shape's CollisionDetector::computeCollisionGraph: trim single-object cells, compact
// the remaining ones, then run exact triangle/triangle tests per cell reference. Compiles under either
// the CUDA or the TBB Thrust backend, the same as the grid builders.
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

#include "triangle.h"

namespace happyray {
namespace {

HAPPYRAY_HD inline uint32_t LoadCollisionId(const uint32_t *address) {
#ifdef __CUDA_ARCH__
  return *static_cast<const volatile uint32_t *>(address);
#else
  return __atomic_load_n(address, __ATOMIC_RELAXED);
#endif
}

// Keeps the smallest colliding object id, so the result doesn't depend on
// which cell/thread records a collision first.
HAPPYRAY_HD inline void AtomicMinCollisionId(uint32_t *address, uint32_t value) {
#ifdef __CUDA_ARCH__
  atomicMin(address, value);
#else
  uint32_t current = __atomic_load_n(address, __ATOMIC_RELAXED);
  while (value < current &&
         !__atomic_compare_exchange_n(address, &current, value, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
#endif
}

struct NonEmptyRange {
  HAPPYRAY_HD bool operator()(const CellRange &cell_range) const {
    return cell_range.start < cell_range.end;
  }
};

// Replaces cells that cannot contain a collision (fewer than two triangles, or
// all triangles from the same object) with an empty range.
struct CellTrimmer {
  const uint32_t *object_ids;
  const uint32_t *triangle_indices;

  HAPPYRAY_HD CellRange operator()(const CellRange &cell_range) const {
    if (cell_range.start + 1 >= cell_range.end)
      return CellRange{}; // Zero or one primitive inside => no collision.

    for (uint32_t ref = cell_range.start; ref + 1 < cell_range.end; ++ref) {
      if (object_ids[triangle_indices[ref]] !=
          object_ids[triangle_indices[ref + 1]])
        return cell_range;
    }
    return CellRange{};
  }

  HAPPYRAY_HD CellRange operator()(const IrregularCell &cell) const {
    return (*this)(cell.references);
  }
};

struct CellSize {
  HAPPYRAY_HD uint32_t operator()(const CellRange &cell_range) const {
    return cell_range.end - cell_range.start;
  }
};

// Tests all triangle pairs from different objects inside the candidate cells.
// Unlike string2shape (one thread per cell), each work item is one reference of
// a candidate cell, tested against the cell's later references: the same pairs,
// but a cell with thousands of triangles (e.g. a uniform grid cell over dense
// geometry) no longer serializes O(k^2) tests on a single thread.
struct ExactCollisionOperator {
  SceneView scene;
  const uint32_t *triangle_indices;
  uint32_t *collision_ids;
  const CellRange *cells;
  const uint32_t *cell_ref_offsets; // Exclusive scan of the cell sizes.
  uint32_t cell_count;

  HAPPYRAY_HD void operator()(uint32_t work_item) const {
    // Last cell whose first work item is <= work_item.
    uint32_t lo = 0, hi = cell_count;
    while (hi - lo > 1) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (cell_ref_offsets[mid] <= work_item)
        lo = mid;
      else
        hi = mid;
    }
    const CellRange cell_range = cells[lo];
    const uint32_t ref = cell_range.start + (work_item - cell_ref_offsets[lo]);

    const uint32_t my_prim = triangle_indices[ref];
    const uint32_t my_obj = scene.object_ids[my_prim];
    const Triangle my_tri = FetchTriangle(scene, my_prim);

    for (uint32_t other_ref = ref + 1; other_ref < cell_range.end;
         ++other_ref) {
      const uint32_t other_prim = triangle_indices[other_ref];
      const uint32_t other_obj = scene.object_ids[other_prim];
      if (my_obj == other_obj)
        continue;
      // Per-primitive analog of string2shape's lastRecordedPair: skip the
      // test if its outcome could not lower either recorded id.
      if (LoadCollisionId(collision_ids + my_prim) <= other_obj &&
          LoadCollisionId(collision_ids + other_prim) <= my_obj)
        continue;

      if (IntersectTriangles(my_tri, FetchTriangle(scene, other_prim))) {
        AtomicMinCollisionId(collision_ids + my_prim, other_obj);
        AtomicMinCollisionId(collision_ids + other_prim, my_obj);
      }
    }
  }
};

struct IsColliding {
  HAPPYRAY_HD bool operator()(uint32_t collision_id) const {
    return collision_id != kNoCollision;
  }
};

// `cells` iterates CellRange or IrregularCell leaves; their ranges index
// `triangle_indices`.
template <typename CellIterator>
CollisionStats DetectCollisions(const SceneView &scene, uint32_t object_count,
                                CellIterator cells, uint32_t cell_count,
                                const uint32_t *triangle_indices,
                                uint32_t *collision_ids) {
  const auto start_time = std::chrono::steady_clock::now();
  CollisionStats stats;
  stats.object_count = object_count;

  const thrust::device_ptr<uint32_t> ids =
      thrust::device_pointer_cast(collision_ids);
  thrust::fill(ids, ids + scene.triangle_count, kNoCollision);

  if (object_count >= 2u && scene.object_ids && cell_count > 0) {
    // Delete all grid cells that contain primitives from a single object.
    thrust::device_vector<CellRange> trimmed_cells(cell_count);
    thrust::transform(cells, cells + cell_count, trimmed_cells.begin(),
                      CellTrimmer{scene.object_ids, triangle_indices});
    thrust::device_vector<CellRange> candidate_cells(cell_count);
    const auto candidate_cells_end =
        thrust::copy_if(trimmed_cells.begin(), trimmed_cells.end(),
                        candidate_cells.begin(), NonEmptyRange());
    stats.candidate_cell_count = static_cast<uint32_t>(
        candidate_cells_end - candidate_cells.begin());

    if (stats.candidate_cell_count > 0) {
      thrust::device_vector<uint32_t> cell_ref_offsets(
          stats.candidate_cell_count);
      thrust::transform_exclusive_scan(candidate_cells.begin(),
                                       candidate_cells_end,
                                       cell_ref_offsets.begin(), CellSize(),
                                       0u, thrust::plus<uint32_t>());
      const CellRange last_cell =
          candidate_cells[stats.candidate_cell_count - 1];
      const uint32_t work_item_count =
          cell_ref_offsets.back() + (last_cell.end - last_cell.start);

      thrust::for_each(
          thrust::counting_iterator<uint32_t>(0),
          thrust::counting_iterator<uint32_t>(work_item_count),
          ExactCollisionOperator{
              scene, triangle_indices, collision_ids,
              thrust::raw_pointer_cast(candidate_cells.data()),
              thrust::raw_pointer_cast(cell_ref_offsets.data()),
              stats.candidate_cell_count});
    }

    stats.colliding_triangle_count = static_cast<uint32_t>(
        thrust::count_if(ids, ids + scene.triangle_count, IsColliding()));
  }

  stats.total_ms = std::chrono::duration<float, std::milli>(
                       std::chrono::steady_clock::now() - start_time)
                       .count();
  return stats;
}

} // namespace

CollisionStats CollisionDetector::ComputeCollisions(
    const SceneView &scene, uint32_t object_count, const UniformGrid &grid,
    uint32_t *collision_ids) const {
  const UniformGridView view = grid.View();
  const uint32_t cell_count = static_cast<uint32_t>(view.resolution_x) *
                              view.resolution_y * view.resolution_z;
  return DetectCollisions(scene, object_count,
                          thrust::device_pointer_cast(view.cell_ranges),
                          cell_count, view.triangle_indices, collision_ids);
}

CollisionStats CollisionDetector::ComputeCollisions(
    const SceneView &scene, uint32_t object_count, const TwoLevelGrid &grid,
    uint32_t *collision_ids) const {
  const TwoLevelGridView view = grid.View();
  return DetectCollisions(
      scene, object_count,
      thrust::device_pointer_cast(view.second_level_cell_ranges),
      grid.SecondLevelCellCount(), view.second_level_triangle_indices,
      collision_ids);
}

CollisionStats CollisionDetector::ComputeCollisions(
    const SceneView &scene, uint32_t object_count, const IrregularGrid &grid,
    uint32_t *collision_ids) const {
  const IrregularGridView view = grid.View();
  return DetectCollisions(scene, object_count,
                          thrust::device_pointer_cast(view.local_cells),
                          grid.LocalCellCount(), view.triangle_indices,
                          collision_ids);
}

} // namespace happyray
