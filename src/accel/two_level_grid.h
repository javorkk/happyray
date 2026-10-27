// The two-level grid stores sorted primitive ranges per top-level and nested
// cell. Sparse top-level cells use a 1x1x1 nested grid.
//
// TwoLevelGrid is exposed through a plain-C++ interface so it can be referenced
// from ordinary .cpp translation units without that caller needing to know
// which Thrust backend is active, the same as UniformGrid (uniform_grid.h) and
// Renderer (src/render/renderer.h).
#pragma once

#include <cstdint>
#include <memory>

#include "ray.h"
#include "sah_cost.h"
#include "scene.h"
#include "triangle_intersect.h"
#include "uniform_grid.h"
#include "vec.h"

namespace happyray {

struct GridBoundsView {
  Vec3 bounds_min;
  Vec3 bounds_max;
  Vec3 cell_size;
  Vec3 inv_cell_size;
  int resolution_x = 1;
  int resolution_y = 1;
  int resolution_z = 1;
};

struct GridRangeView : GridBoundsView {
  const CellRange *cell_ranges = nullptr;
  const uint32_t *triangle_indices = nullptr;
};

// Per-top-level-cell second-level grid metadata. `cell_range_offset` locates
// this grid's ranges in the shared second-level arrays.
struct alignas(8) TopLevelCell {
  int resolution = 1;
  uint32_t cell_range_offset = 0;

  HAPPYRAY_HD bool IsEmpty() const { return resolution < 0; }
  HAPPYRAY_HD int Resolution() const {
    return IsEmpty() ? -resolution : resolution;
  }
};
static_assert(sizeof(TopLevelCell) == 2 * sizeof(uint32_t));
static_assert(alignof(TopLevelCell) == 8);

struct TwoLevelGridView {
  GridBoundsView top_level;
  const TopLevelCell *top_level_cells =
      nullptr; // One entry per top-level cell.
  const CellRange *second_level_cell_ranges = nullptr;
  const uint32_t *second_level_triangle_indices =
      nullptr; // Sorted by second-level cell.
};

// Builds the range view for top-level cell `cell_index`'s second-level grid.
HAPPYRAY_HD inline GridRangeView SecondLevelView(const TwoLevelGridView &grid,
                                                 uint32_t cell_index) {
  const GridBoundsView &top = grid.top_level;
  const TopLevelCell &info = grid.top_level_cells[cell_index];
  const int resolution = info.Resolution();
  const int res_xy = top.resolution_x * top.resolution_y;
  const int cz = static_cast<int>(cell_index) / res_xy;
  const int rem = static_cast<int>(cell_index) % res_xy;
  const int cy = rem / top.resolution_x;
  const int cx = rem % top.resolution_x;
  const Vec3 bounds_min(top.bounds_min.x + cx * top.cell_size.x,
                        top.bounds_min.y + cy * top.cell_size.y,
                        top.bounds_min.z + cz * top.cell_size.z);
  const Vec3 cell_size(top.cell_size.x / resolution,
                       top.cell_size.y / resolution,
                       top.cell_size.z / resolution);
  GridRangeView view;
  view.bounds_min = bounds_min;
  view.bounds_max = bounds_min + top.cell_size;
  view.cell_size = cell_size;
  view.inv_cell_size = Vec3(1.0f / cell_size.x, 1.0f / cell_size.y,
                            1.0f / cell_size.z);
  view.resolution_x = resolution;
  view.resolution_y = resolution;
  view.resolution_z = resolution;
  view.cell_ranges = grid.second_level_cell_ranges + info.cell_range_offset;
  view.triangle_indices = grid.second_level_triangle_indices;
  return view;
}

namespace detail {

// Two-level traversal, ported from happyray's TLGridTraverser: the outer DDA
// directly initializes an inner DDA for each non-empty top-level cell, with no
// nested WalkGridCells calls or second-level box test. Top-level cells with no
// triangles are skipped using their resolution sign bit.
// `visit_top(top_index)` is called for every top-level cell visited (a no-op
// for queries; the SAH calibration counts steps with it);
// `visit_leaf(leaf_index, &ray_t_max)` for every second-level cell, where
// `leaf_index` indexes the shared second-level range arrays.
template <typename VisitTopFn, typename VisitLeafFn>
HAPPYRAY_HD inline bool TraverseTwoLevelGrid(const TwoLevelGridView &grid,
                                             const Ray &ray,
                                             VisitTopFn &&visit_top,
                                             VisitLeafFn &&visit_leaf) {
  const GridBoundsView &top = grid.top_level;
  const Vec3 inv_dir = SafeRcp(ray.direction);
  float t_near, t_far;
  if (!IntersectRayBoxRcp(ray.origin, ray.direction, inv_dir, top.bounds_min,
                          top.bounds_max, ray.t_min, ray.t_max, &t_near,
                          &t_far)) {
    return false;
  }
  float ray_t_max = ray.t_max;
  const Vec3 top_entry =
      (ray.origin + ray.direction * t_near - top.bounds_min) * top.inv_cell_size;
  int cx = ClampInt(static_cast<int>(top_entry.x), 0, top.resolution_x - 1);
  int cy = ClampInt(static_cast<int>(top_entry.y), 0, top.resolution_y - 1);
  int cz = ClampInt(static_cast<int>(top_entry.z), 0, top.resolution_z - 1);
  const float step_x = ray.direction.x > 0.0f ? 1.0f : 0.0f;
  const float step_y = ray.direction.y > 0.0f ? 1.0f : 0.0f;
  const float step_z = ray.direction.z > 0.0f ? 1.0f : 0.0f;
  float next_x = ray.direction.x != 0.0f
                     ? ((cx + step_x) * top.cell_size.x + top.bounds_min.x -
                        ray.origin.x) * inv_dir.x
                     : INFINITY;
  float next_y = ray.direction.y != 0.0f
                     ? ((cy + step_y) * top.cell_size.y + top.bounds_min.y -
                        ray.origin.y) * inv_dir.y
                     : INFINITY;
  float next_z = ray.direction.z != 0.0f
                     ? ((cz + step_z) * top.cell_size.z + top.bounds_min.z -
                        ray.origin.z) * inv_dir.z
                     : INFINITY;
  const float delta_x = top.cell_size.x * std::fabs(inv_dir.x);
  const float delta_y = top.cell_size.y * std::fabs(inv_dir.y);
  const float delta_z = top.cell_size.z * std::fabs(inv_dir.z);
  const int inc_x = ray.direction.x > 0.0f ? 1 : -1;
  const int inc_y = ray.direction.y > 0.0f ? 1 : -1;
  const int inc_z = ray.direction.z > 0.0f ? 1 : -1;
  float t_cell = t_near;

  for (;;) {
    const uint32_t top_index =
        static_cast<uint32_t>(cx) +
        static_cast<uint32_t>(top.resolution_x) *
            (static_cast<uint32_t>(cy) +
             static_cast<uint32_t>(top.resolution_y) * static_cast<uint32_t>(cz));
    visit_top(top_index);

    const TopLevelCell &info = grid.top_level_cells[top_index];
    if (!info.IsEmpty()) {
      const int resolution = info.Resolution();
      const int res_x = resolution;
      const int res_y = resolution;
      const int res_z = resolution;
      const Vec3 cell_min(top.bounds_min.x + cx * top.cell_size.x,
                          top.bounds_min.y + cy * top.cell_size.y,
                          top.bounds_min.z + cz * top.cell_size.z);
      const Vec3 sub_inv(top.inv_cell_size.x * res_x,
                         top.inv_cell_size.y * res_y,
                         top.inv_cell_size.z * res_z);
      const Vec3 sub_cell_size(top.cell_size.x / res_x,
                               top.cell_size.y / res_y,
                               top.cell_size.z / res_z);
      const Vec3 leaf_entry =
          (ray.origin + ray.direction * t_cell - cell_min) * sub_inv;
      int leaf_x = ClampInt(static_cast<int>(leaf_entry.x), 0, res_x - 1);
      int leaf_y = ClampInt(static_cast<int>(leaf_entry.y), 0, res_y - 1);
      int leaf_z = ClampInt(static_cast<int>(leaf_entry.z), 0, res_z - 1);
      const float leaf_step_x = ray.direction.x > 0.0f ? 1.0f : 0.0f;
      const float leaf_step_y = ray.direction.y > 0.0f ? 1.0f : 0.0f;
      const float leaf_step_z = ray.direction.z > 0.0f ? 1.0f : 0.0f;
      float leaf_next_x =
          ray.direction.x != 0.0f
              ? ((leaf_x + leaf_step_x) * sub_cell_size.x + cell_min.x -
                 ray.origin.x) * inv_dir.x
              : INFINITY;
      float leaf_next_y =
          ray.direction.y != 0.0f
              ? ((leaf_y + leaf_step_y) * sub_cell_size.y + cell_min.y -
                 ray.origin.y) * inv_dir.y
              : INFINITY;
      float leaf_next_z =
          ray.direction.z != 0.0f
              ? ((leaf_z + leaf_step_z) * sub_cell_size.z + cell_min.z -
                 ray.origin.z) * inv_dir.z
              : INFINITY;
      const float leaf_delta_x = sub_cell_size.x * std::fabs(inv_dir.x);
      const float leaf_delta_y = sub_cell_size.y * std::fabs(inv_dir.y);
      const float leaf_delta_z = sub_cell_size.z * std::fabs(inv_dir.z);
      const int leaf_inc_x = ray.direction.x > 0.0f ? 1 : -1;
      const int leaf_inc_y = ray.direction.y > 0.0f ? 1 : -1;
      const int leaf_inc_z = ray.direction.z > 0.0f ? 1 : -1;

      for (;;) {
        const uint32_t leaf = FlattenCellIndex(leaf_x, leaf_y, leaf_z, res_x, res_y);
        if (visit_leaf(info.cell_range_offset + leaf, &ray_t_max)) {
          return true;
        }

        if (leaf_next_x < leaf_next_y && leaf_next_x < leaf_next_z) {
          if (leaf_next_x > ray_t_max) break;
          leaf_x += leaf_inc_x;
          if (static_cast<unsigned>(leaf_x) >= static_cast<unsigned>(res_x))
            break;
          leaf_next_x += leaf_delta_x;
        } else if (leaf_next_y < leaf_next_z) {
          if (leaf_next_y > ray_t_max) break;
          leaf_y += leaf_inc_y;
          if (static_cast<unsigned>(leaf_y) >= static_cast<unsigned>(res_y))
            break;
          leaf_next_y += leaf_delta_y;
        } else {
          if (leaf_next_z > ray_t_max) break;
          leaf_z += leaf_inc_z;
          if (static_cast<unsigned>(leaf_z) >= static_cast<unsigned>(res_z))
            break;
          leaf_next_z += leaf_delta_z;
        }
      }
    }

    if (next_x < next_y && next_x < next_z) {
      if (next_x > ray_t_max) return false;
      cx += inc_x;
      if (static_cast<unsigned>(cx) >= static_cast<unsigned>(top.resolution_x))
        return false;
      t_cell = next_x;
      next_x += delta_x;
    } else if (next_y < next_z) {
      if (next_y > ray_t_max) return false;
      cy += inc_y;
      if (static_cast<unsigned>(cy) >= static_cast<unsigned>(top.resolution_y))
        return false;
      t_cell = next_y;
      next_y += delta_y;
    } else {
      if (next_z > ray_t_max) return false;
      cz += inc_z;
      if (static_cast<unsigned>(cz) >= static_cast<unsigned>(top.resolution_z))
        return false;
      t_cell = next_z;
      next_z += delta_z;
    }
  }
}

} // namespace detail

// Finds the closest hit by walking the top-level grid and, for every non-empty
// cell the ray passes through, its second-level grid (see
// detail::TraverseTwoLevelGrid above). Returns false if the ray hits nothing.
HAPPYRAY_HD inline bool IntersectSceneTwoLevelGrid(const SceneView &scene,
                                               const TwoLevelGridView &grid,
                                               const Ray &ray,
                                               HitRecord *out_hit) {
  ClosestTriangleHit hit;
  detail::TraverseTwoLevelGrid(
      grid, ray, [](uint32_t) {},
      [&](uint32_t leaf, float *ray_t_max) -> bool {
        IntersectTrianglesClosest(scene, grid.second_level_triangle_indices,
                grid.second_level_cell_ranges[leaf].start,
                grid.second_level_cell_ranges[leaf].end, ray,
                ray_t_max, &hit);
        return false;
      });
  return ResolveClosestHit(scene, ray, hit, out_hit);
}

// Returns true as soon as any triangle in any second-level cell the ray visits
// occludes `ray`.
HAPPYRAY_HD inline bool IsOccludedTwoLevelGrid(const SceneView &scene,
                                           const TwoLevelGridView &grid,
                                           const Ray &ray) {
  return detail::TraverseTwoLevelGrid(
      grid, ray, [](uint32_t) {},
      [&](uint32_t leaf, float *) -> bool {
        return TestTrianglesInRangeOccluded(
            scene, grid.second_level_triangle_indices,
            grid.second_level_cell_ranges[leaf].start,
            grid.second_level_cell_ranges[leaf].end, ray);
      });
}

class TwoLevelGrid {
public:
  static constexpr float kDefaultTopLevelDensity = 0.12f;
  static constexpr float kDefaultSecondLevelDensity = 1.2f;
  // `scene` must point at memory reachable by the selected Thrust backend.
  // Cells at or below `dense_cell_threshold` use a 1x1x1 second-level grid.
  explicit TwoLevelGrid(const SceneView &scene,
                        float top_level_density = kDefaultTopLevelDensity,
                        float second_level_density = kDefaultSecondLevelDensity,
                        uint32_t dense_cell_threshold = 16);
  ~TwoLevelGrid();
  TwoLevelGrid(const TwoLevelGrid &) = delete;
  TwoLevelGrid &operator=(const TwoLevelGrid &) = delete;

  TwoLevelGridView View() const;

  // Per-array element counts of every device buffer this grid actually holds,
  // used by
  // tests/accel_performance_tests.cpp to compute an analytic memory-footprint
  // estimate (count * sizeof(element)) without needing any CUDA-specific
  // allocation introspection. `top_level`'s own cell count is
  // `View().top_level.resolution_x/y/z`'s product, already public; these fill
  // in the remaining counts that aren't otherwise derivable from the view.
  uint32_t SecondLevelCellCount() const;
  uint32_t SecondLevelTriangleIndexCount() const;

  // Structural SAH cost estimate for this grid as currently built (see
  // sah_cost.h), normalized by `scene_bounds`'s own half surface area -- not
  // from casting rays, but the same surface-area-weighted "expected cost"
  // heuristic the grid density search (see src/app/grid_density_search.h) uses
  // to prefer one candidate density over another. Each nested grid's reference
  // count is aggregated while the sorted ranges are built.
  float ComputeSahCost(const BBox &scene_bounds,
                       const SahCostConstants &costs) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace happyray
