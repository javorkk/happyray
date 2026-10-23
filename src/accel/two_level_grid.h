// The two-level grid: a coarse top-level uniform grid where *every* cell
// recurses into a second-level (nested) uniform grid instead of holding
// triangles directly -- for sparse cells, that second-level grid is simply
// forced to 1x1x1 resolution, which is exactly equivalent to a direct triangle
// list (its single cell's CSR range covers all the cell's triangles). This
// unifies "leaf" and "refined" cells into one code path with no branching,
// matching javorkk/happyray's actual
// TwoLevelGrid/TLGridSortBuilder/TLGridTraverser design -- see
// two_level_grid.cpp's header comment for the construction algorithm.
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

// Per-top-level-cell second-level grid metadata. `cell_start_offset` is where
// this instance's CSR entries begin in TwoLevelGridView's shared
// `second_level_cell_start`/ `second_level_triangle_indices` arrays (every
// top-level cell's second-level grid is built together into one combined CSR,
// the same sort-based way as the top level itself). The fields
// detail::TraverseTwoLevelGrid reads (resolution, offset, cell_size) come
// first, so they share one 32-byte memory sector; the rest is build-time data.
struct SecondLevelGridInfo {
  int resolution_x = 1;
  int resolution_y = 1;
  int resolution_z = 1;
  uint32_t cell_start_offset = 0;
  Vec3 cell_size;
  Vec3 inv_cell_size;
  Vec3 bounds_min;
  Vec3 bounds_max;
};

struct TwoLevelGridView {
  // bounds_min/bounds_max/cell_size/inv_cell_size/resolution_* drive the
  // top-level 3D-DDA walk; cell_start is read only to skip empty top-level
  // cells (triangle_indices is build-time-only), since every top-level cell
  // recurses into a second-level grid.
  UniformGridView top_level;
  const SecondLevelGridInfo *second_level_infos =
      nullptr; // One entry per top-level cell.
  const uint32_t *second_level_cell_start = nullptr; // Combined CSR offsets.
  const uint32_t *second_level_triangle_indices =
      nullptr; // Combined sorted triangle refs.
};

// Builds the UniformGridView for top-level cell `cell_index`'s second-level
// grid (pointers already offset into the shared arrays), ready to recurse into
// via IntersectSceneGrid/ IsOccludedGrid (uniform_grid.h).
HAPPYRAY_HD inline UniformGridView SecondLevelView(const TwoLevelGridView &grid,
                                               uint32_t cell_index) {
  const SecondLevelGridInfo &info = grid.second_level_infos[cell_index];
  UniformGridView view;
  view.bounds_min = info.bounds_min;
  view.bounds_max = info.bounds_max;
  view.cell_size = info.cell_size;
  view.inv_cell_size = info.inv_cell_size;
  view.resolution_x = info.resolution_x;
  view.resolution_y = info.resolution_y;
  view.resolution_z = info.resolution_z;
  view.cell_start = grid.second_level_cell_start + info.cell_start_offset;
  view.triangle_indices = grid.second_level_triangle_indices;
  return view;
}

namespace detail {

// Two-level traversal, ported from happyray's TLGridTraverser: the outer 3D-DDA
// walks the top-level grid and hands each visited cell's entry parameter
// `t_cell_entry` straight to the inner 3D-DDA, which starts from that point
// with bounds derived from the top-level cell's own coordinates (bounds_min +
// cell * cell_size, with the cell coordinates taken straight from the outer
// walk) and subCellSizeRCP = cellSizeRCP * res -- so entering a second-level
// grid costs no ray/box test and reads only its resolution, CSR offset and cell
// size. Top-level cells with no triangles are skipped without descending
// (happyray's `cell.notEmpty()`), using the top level's own CSR counts.
// `visit_top(top_index)` is called for every top-level cell visited (a no-op
// for queries; the SAH calibration counts steps with it);
// `visit_leaf(leaf_index, &ray_t_max)` for every second-level cell, where
// `leaf_index` indexes `second_level_cell_start`. Same stopping rules as
// WalkGridCells (uniform_grid.h).
template <typename VisitTopFn, typename VisitLeafFn>
HAPPYRAY_HD inline bool TraverseTwoLevelGrid(const TwoLevelGridView &grid,
                                         const Ray &ray, VisitTopFn &&visit_top,
                                         VisitLeafFn &&visit_leaf) {
  const UniformGridView &top = grid.top_level;
  const Vec3 inv_dir = SafeRcp(ray.direction);
  float t_near, t_far;
  if (!IntersectRayBoxRcp(ray.origin, ray.direction, inv_dir, top.bounds_min,
                          top.bounds_max, ray.t_min, ray.t_max, &t_near,
                          &t_far)) {
    return false;
  }
  float ray_t_max = ray.t_max;
  return WalkGridCells(
      ray.origin, ray.direction, inv_dir, top.bounds_min, top.cell_size,
      top.inv_cell_size, top.resolution_x, top.resolution_y, top.resolution_z,
      t_near, &ray_t_max,
      [&](uint32_t top_index, int cx, int cy, int cz, float t_cell_entry,
          float *t_max) -> bool {
        visit_top(top_index);
        if (top.cell_start[top_index] == top.cell_start[top_index + 1]) {
          return false;
        }
        const SecondLevelGridInfo &info = grid.second_level_infos[top_index];
        const int res_x = info.resolution_x, res_y = info.resolution_y,
                  res_z = info.resolution_z;
        const uint32_t offset = info.cell_start_offset;
        const Vec3 cell_min(top.bounds_min.x + cx * top.cell_size.x,
                            top.bounds_min.y + cy * top.cell_size.y,
                            top.bounds_min.z + cz * top.cell_size.z);
        const Vec3 sub_inv(top.inv_cell_size.x * res_x,
                           top.inv_cell_size.y * res_y,
                           top.inv_cell_size.z * res_z);
        return WalkGridCells(ray.origin, ray.direction, inv_dir, cell_min,
                             info.cell_size, sub_inv, res_x, res_y, res_z,
                             t_cell_entry, t_max,
                             [&](uint32_t leaf, int, int, int, float,
                                 float *leaf_t_max) -> bool {
                               return visit_leaf(offset + leaf, leaf_t_max);
                             });
      });
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
                                  grid.second_level_cell_start[leaf],
                                  grid.second_level_cell_start[leaf + 1], ray,
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
            grid.second_level_cell_start[leaf],
            grid.second_level_cell_start[leaf + 1], ray);
      });
}

class TwoLevelGrid {
public:
  static constexpr float kDefaultTopLevelDensity = 0.12f;
  static constexpr float kDefaultSecondLevelDensity = 2.4f;
  // `scene` must point at memory reachable by whichever Thrust backend this
  // binary was built with (device memory for the CUDA backend, ordinary host
  // memory for the TBB backend). Defaults match happyray's TLGridSortBuilder: a
  // coarse top-level grid (low density, so most triangle counts per cell stay
  // above `dense_cell_threshold`) refined by a much finer second-level grid (~1
  // triangle/cell) wherever a top-level cell is "dense" (more than
  // `dense_cell_threshold` triangles) -- cells at or below the threshold get a
  // degenerate (1x1x1) second-level grid instead, equivalent to a direct
  // triangle list.
  explicit TwoLevelGrid(const SceneView &scene,
                        float top_level_density = kDefaultTopLevelDensity,
                        float second_level_density = kDefaultSecondLevelDensity,
                        uint32_t dense_cell_threshold = 16);
  ~TwoLevelGrid();
  TwoLevelGrid(const TwoLevelGrid &) = delete;
  TwoLevelGrid &operator=(const TwoLevelGrid &) = delete;

  TwoLevelGridView View() const;

  // Per-array element counts of every device buffer this grid actually holds
  // (top_level's own CSR plus the combined second-level CSR), used by
  // tests/accel_performance_tests.cpp to compute an analytic memory-footprint
  // estimate (count * sizeof(element)) without needing any CUDA-specific
  // allocation introspection. `top_level`'s own cell count/second-level info
  // count is `View().top_level.resolution_x/y/z`'s product, already public;
  // these fill in the remaining counts that aren't otherwise derivable from
  // TwoLevelGridView alone.
  uint32_t TopLevelTriangleIndexCount() const;
  uint32_t SecondLevelCellCount() const;
  uint32_t SecondLevelTriangleIndexCount() const;

  // Structural SAH cost estimate for this grid as currently built (see
  // sah_cost.h), normalized by `scene_bounds`'s own half surface area -- not
  // from casting rays, but the same surface-area-weighted "expected cost"
  // heuristic the grid density search (see src/app/grid_density_search.h) uses
  // to prefer one candidate density over another. Every top-level cell's own
  // second-level grid is dense and uniformly sized within that one top-level
  // cell (no merging), so its contribution also collapses to a closed form from
  // just its own resolution and the aggregate size of its CSR reference range.
  float ComputeSahCost(const BBox &scene_bounds,
                       const SahCostConstants &costs) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace happyray
