// The irregular grid: a coarse top-level uniform grid (reusing UniformGrid
// exactly like the two-level grid does) where every cell owns its own flat,
// dense, power-of-2-resolution "finest" local grid, which is then refined
// bottom-up by two passes ported directly from HAGrid (javorkk/happyray's
// companion project, cg-saarland/hagrid):
//
//  - Merge: iteratively pairs up adjacent local cells along each axis and combines them
//    whenever doing so doesn't increase the grid's split-SAH cost (HAGrid's
//    merge.cu), so large empty or uniform regions collapse into a few big cells
//    instead of staying a dense array of tiny ones.
//  - Expand: grows a cell's bounding box into neighboring cells whose triangle references are
//    already a *subset* of its own (HAGrid's expand.cu) -- safe without
//    touching which cell any given finest-voxel coordinate resolves to, since
//    the growing cell's own (unchanged) reference list already covers the
//    absorbed region correctly.
//
// Both passes are allowed to cross top-level cell boundaries, same as HAGrid's own
// (single, flat) grid -- the key piece that makes this tractable without reproducing
// HAGrid's adaptive octree *build* or its `grid_shift`/`empty_mask` bookkeeping is that
// every top-level cell is the exact same world-space size (UniformGrid's own invariant),
// so every top-level cell's independently-chosen, power-of-2 local resolution is
// automatically commensurate with every other one's: picking the coarsest common
// refinement (`IrregularGridView::global_resolution_*`, the smallest power of 2 that is a
// multiple of every top-level cell's own resolution) gives one shared integer lattice the
// *entire* grid's cells can be expressed in, with no adaptive octree descent needed to
// find a shared coordinate for two cells in different top-level cells. `IrregularCell`'s
// bounds are stored in this single shared lattice (directly matching HAGrid's own
// `ivec3 Cell::min/max`), so alignment/merge/expand arithmetic is exact integer
// comparison with no per-cell "whose units is this" bookkeeping at all -- see
// `IrregularCellInfo::shift_x/y/z`/`global_offset_x/y/z` below for how a top-level cell's
// own local voxel index maps into that shared lattice, and LookupGlobalCell
// (irregular_grid.cpp) for how merge/expand resolve a shared-lattice position back to
// whichever top-level cell's local_entries currently claims it (needed since
// `local_entries` itself still stays one independent array per top-level cell, each
// built at that cell's own, density-adapted resolution -- the one piece of HAGrid's
// design this project still doesn't reproduce, in favor of the flat/dense/sort-based
// per-cell construction every other grid in this project already uses). HAGrid's
// adaptive top-down octree build and bit-packed compression are deliberately left out.
//
// IrregularGrid is exposed through a plain-C++ interface so it can be
// referenced from ordinary .cpp translation units without that caller needing
// to know which Thrust backend is active, the same as UniformGrid/TwoLevelGrid
// and Renderer (src/render/renderer.h).
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

// Per-top-level-cell local "finest grid" metadata -- the local analogue of
// TwoLevelGrid's SecondLevelGridInfo. `cell_size`/`resolution_*` describe the
// *finest* voxel granularity (the grid before any merging coarsens it);
// `entries_offset` is where this cell's local voxel-map entries begin in
// IrregularGridView's shared `local_entries` array (every top-level cell's
// local grid is built together, the same combined-array approach as the
// two-level grid). `shift_*`/`global_offset_*` place this cell's own local
// voxel grid within the grid-wide shared integer lattice IrregularCell's own
// bounds are expressed in (see the file header comment above): local voxel
// index `v` on a given axis maps to shared coordinate `global_offset + (v <<
// shift)`, where `shift = log2(IrregularGridView::global_resolution /
// resolution)` (always a whole number of bits, since both are powers of 2 of
// this top-level cell's own, identically-sized, world-space extent -- a
// shift instead of a general multiply/divide keeps the local<->shared
// lattice conversion on detail::LookupIrregularCell's hot path cheap)
// and `global_offset` is this top-level cell's own origin in that same
// shared lattice (`(cx, cy, cz) * global_resolution`, `(cx, cy, cz)` being
// this cell's own position in the top-level grid).
struct IrregularCellInfo {
  Vec3 bounds_min;
  Vec3 cell_size; // Of one *finest* local voxel (pre-merge granularity).
  Vec3 inv_cell_size;
  int resolution_x = 1; // Already rounded up to a power of 2.
  int resolution_y = 1;
  int resolution_z = 1;
  uint32_t entries_offset = 0;
  int shift_x = 0, shift_y = 0, shift_z = 0;
  int global_offset_x = 0, global_offset_y = 0, global_offset_z = 0;
};

// One post-merge/expand irregular cell (NOT one per finest voxel -- that's what
// the local entries array is for; several finest voxels can resolve to the
// same, bigger, IrregularCell). Bounds are stored as integer coordinates in
// the grid-wide shared lattice described in this file's header comment above
// (directly matching HAGrid's own `ivec3 Cell::min/max`), not world space, so
// merge/expand's bbox-growing arithmetic stays exact integer math regardless
// of how many top-level cells a cell has grown across; traversal converts to
// world-space t-values itself using IrregularGridView's shared
// `global_cell_size` (see detail::TraverseIrregularGrid below).
struct alignas(16) IrregularCell {
  int min_x = 0, min_y = 0, min_z = 0;
  int max_x = 0, max_y = 0,
      max_z = 0; // Exclusive, like a UniformGridView cell range.
  uint32_t ref_start = 0;
  uint32_t ref_end = 0;
};

// Traversal-only, 8-byte digest of an IrregularCellInfo (the analogue of HAGrid's 4-byte voxel
// map `Entry`): everything detail::LookupIrregularCell needs to turn a shared-lattice voxel into
// a local_entries slot, packed so each traversal step reads one 8-byte record instead of the
// 76-byte IrregularCellInfo. `packed` holds 5-bit fields: shift_x | shift_y << 5 | shift_z << 10 |
// log2(resolution_x) << 15 | log2(resolution_y) << 20 (resolution_z isn't needed to flatten).
struct alignas(8) IrregularTopCell {
  uint32_t entries_offset = 0;
  uint32_t packed = 0;
};

HAPPYRAY_HD inline int IrregularLog2(int x) {
  int log = 0;
  while ((1 << log) < x) {
    ++log;
  }
  return log;
}

HAPPYRAY_HD inline IrregularTopCell MakeIrregularTopCell(const IrregularCellInfo &info) {
  IrregularTopCell top;
  top.entries_offset = info.entries_offset;
  top.packed = static_cast<uint32_t>(info.shift_x) |
               (static_cast<uint32_t>(info.shift_y) << 5) |
               (static_cast<uint32_t>(info.shift_z) << 10) |
               (static_cast<uint32_t>(IrregularLog2(info.resolution_x)) << 15) |
               (static_cast<uint32_t>(IrregularLog2(info.resolution_y)) << 20);
  return top;
}

struct IrregularGridView {
  // Bounds/cell size/resolution define the top-level grid that the shared lattice subdivides.
  UniformGridView top_level;
  const IrregularCellInfo *cell_infos = nullptr; // One per top-level cell (build-time data).
  const IrregularTopCell *top_cells = nullptr;   // One per top-level cell (traversal lookup).
  // local_entries[info.entries_offset + FlattenCellIndex(x, y, z,
  // info.resolution_x, info.resolution_y)] gives the *absolute* index (into
  // local_cells, no further offset needed) of the IrregularCell that currently
  // owns local finest-voxel (x, y, z) -- before any merging this is the
  // identity permutation; after merging, several finest-voxel entries can
  // resolve to the same (now bigger) cell, possibly owned by a *different*
  // top-level cell than the one whose local_entries slot is being read (see
  // the file header comment above).
  const uint32_t *local_entries = nullptr;
  const IrregularCell *local_cells =
      nullptr; // Combined across every top-level cell.
  const uint32_t *triangle_indices =
      nullptr; // Combined sorted leaf CSR payload.
  // The grid-wide shared lattice every IrregularCell's bounds are expressed
  // in (see the file header comment above): `global_cell_size` is one shared
  // lattice step's world-space size, and `global_resolution_*` is how many
  // lattice steps span one top-level cell (the same for every top-level cell,
  // since they're all the same world-space size) -- i.e. for every top-level
  // cell's own IrregularCellInfo, `global_resolution_<axis> ==
  // resolution_<axis> << shift_<axis>`.
  Vec3 global_cell_size;
  int global_resolution_x = 1, global_resolution_y = 1, global_resolution_z = 1;
  // Traversal helpers derived from the above: 1 / global_cell_size, and log2 of
  // global_resolution_* (always a power of 2).
  Vec3 global_inv_cell_size;
  int global_shift_x = 0, global_shift_y = 0, global_shift_z = 0;
};

namespace detail {

// Resolves shared-lattice voxel (vx, vy, vz) -- which must lie inside the grid -- to the index
// of the IrregularCell that owns it: top-level cell from the high bits, then that cell's own
// local voxel from the low bits (HAGrid's lookup_entry, with the per-top-level-cell local
// resolution standing in for HAGrid's octree levels).
HAPPYRAY_HD inline uint32_t LookupIrregularCell(const IrregularGridView &grid, int vx, int vy,
                                            int vz) {
  const int tx = vx >> grid.global_shift_x;
  const int ty = vy >> grid.global_shift_y;
  const int tz = vz >> grid.global_shift_z;
  const IrregularTopCell top =
      grid.top_cells[FlattenCellIndex(tx, ty, tz, grid.top_level.resolution_x,
                                      grid.top_level.resolution_y)];
  const uint32_t p = top.packed;
  const int lx = (vx & (grid.global_resolution_x - 1)) >> (p & 31u);
  const int ly = (vy & (grid.global_resolution_y - 1)) >> ((p >> 5) & 31u);
  const int lz = (vz & (grid.global_resolution_z - 1)) >> ((p >> 10) & 31u);
  const int log_x = static_cast<int>((p >> 15) & 31u);
  const int log_y = static_cast<int>((p >> 20) & 31u);
  return grid.local_entries[top.entries_offset + static_cast<uint32_t>(lx) +
                            (static_cast<uint32_t>(ly) << log_x) +
                            (static_cast<uint32_t>(lz) << (log_x + log_y))];
}

// Next voxel along one axis after leaving `cell` at `t_exit` (HAGrid's traverse.cu): the axis
// (or axes, on a tie) the ray exits through steps just past the cell's far plane; every other
// axis is re-derived from the exit point, clamped to the cell's own extent on that axis (the
// ray provably hasn't left it there yet) and to never move backwards, so float rounding near a
// boundary can neither skip a neighbor nor stall progress.
// `voxel_offset + voxel_scale * t` is the ray's position at `t` in lattice units on this axis.
HAPPYRAY_HD inline int NextIrregularVoxel(float t_cell, float t_exit, float voxel_offset,
                                      float voxel_scale, bool positive, int voxel,
                                      int cell_min, int cell_max) {
  if (t_cell == t_exit) {
    return positive ? cell_max : cell_min - 1;
  }
  const int exit_voxel = ClampInt(static_cast<int>(voxel_offset + voxel_scale * t_exit),
                                  cell_min, cell_max - 1);
  return positive ? (exit_voxel > voxel ? exit_voxel : voxel)
                  : (exit_voxel < voxel ? exit_voxel : voxel);
}

// Flat irregular-grid traversal, ported from HAGrid's traverse.cu: a single loop over the
// grid-wide shared voxel lattice (irregular_grid.h's file header comment) -- no nested
// top-level 3D-DDA -- that per step looks up the owning cell, intersects the ray with that
// cell's far planes, jumps straight past its exit, and invokes `visit(cell, &ray_t_max)`.
// Stops once `visit` returns true (occlusion), once a hit (or the ray's own end) lies inside
// the current cell (`ray_t_max <= t_exit`), or when the ray leaves the grid. Returns true only
// if `visit` did. Every step strictly advances at least one voxel coordinate monotonically, so
// the walk always terminates.
template <typename VisitFn>
HAPPYRAY_HD inline bool TraverseIrregularGrid(const IrregularGridView &grid, const Ray &ray,
                                          VisitFn &&visit) {
  const UniformGridView &top = grid.top_level;
  const Vec3 &org = ray.origin;
  const Vec3 &dir = ray.direction;
  const Vec3 inv_dir = SafeRcp(dir);
  float t_start, t_end;
  if (!IntersectRayBoxRcp(org, dir, inv_dir, top.bounds_min, top.bounds_max, ray.t_min,
                          ray.t_max, &t_start, &t_end)) {
    return false;
  }
  const int dims_x = top.resolution_x << grid.global_shift_x;
  const int dims_y = top.resolution_y << grid.global_shift_y;
  const int dims_z = top.resolution_z << grid.global_shift_z;
  const Vec3 &gmin = top.bounds_min;
  const Vec3 &gsize = grid.global_cell_size;
  const Vec3 &ginv = grid.global_inv_cell_size;
  const Vec3 start = (org + dir * t_start - gmin) * ginv;
  int vx = ClampInt(static_cast<int>(start.x), 0, dims_x - 1);
  int vy = ClampInt(static_cast<int>(start.y), 0, dims_y - 1);
  int vz = ClampInt(static_cast<int>(start.z), 0, dims_z - 1);

  // Ray-constant per-axis terms: direction signs, and the ray's lattice-space position at `t`
  // as `v_offset + v_scale * t` (used to re-derive non-exit axes after each step).
  const bool pos_x = dir.x >= 0.0f, pos_y = dir.y >= 0.0f, pos_z = dir.z >= 0.0f;
  const float v_scale_x = dir.x * ginv.x, v_offset_x = (org.x - gmin.x) * ginv.x;
  const float v_scale_y = dir.y * ginv.y, v_offset_y = (org.y - gmin.y) * ginv.y;
  const float v_scale_z = dir.z * ginv.z, v_offset_z = (org.z - gmin.z) * ginv.z;

  float ray_t_max = ray.t_max;
  for (;;) {
    const IrregularCell cell = grid.local_cells[LookupIrregularCell(grid, vx, vy, vz)];

    // Intersect the far-most planes of the cell's box (zero direction components never exit).
    const float tcx = dir.x != 0.0f
                          ? ((pos_x ? cell.max_x : cell.min_x) * gsize.x + gmin.x - org.x) *
                                inv_dir.x
                          : INFINITY;
    const float tcy = dir.y != 0.0f
                          ? ((pos_y ? cell.max_y : cell.min_y) * gsize.y + gmin.y - org.y) *
                                inv_dir.y
                          : INFINITY;
    const float tcz = dir.z != 0.0f
                          ? ((pos_z ? cell.max_z : cell.min_z) * gsize.z + gmin.z - org.z) *
                                inv_dir.z
                          : INFINITY;
    const float t_exit = std::fmin(tcx, std::fmin(tcy, tcz));

    if (visit(cell, &ray_t_max)) {
      return true;
    }
    if (ray_t_max <= t_exit) {
      return false;
    }

    vx = NextIrregularVoxel(tcx, t_exit, v_offset_x, v_scale_x, pos_x, vx, cell.min_x,
                            cell.max_x);
    vy = NextIrregularVoxel(tcy, t_exit, v_offset_y, v_scale_y, pos_y, vy, cell.min_y,
                            cell.max_y);
    vz = NextIrregularVoxel(tcz, t_exit, v_offset_z, v_scale_z, pos_z, vz, cell.min_z,
                            cell.max_z);
    if (static_cast<unsigned>(vx) >= static_cast<unsigned>(dims_x) ||
        static_cast<unsigned>(vy) >= static_cast<unsigned>(dims_y) ||
        static_cast<unsigned>(vz) >= static_cast<unsigned>(dims_z)) {
      return false;
    }
  }
}

} // namespace detail

// Finds the closest hit by walking the irregular grid's cells front to back (see
// detail::TraverseIrregularGrid above). Returns false if the ray hits nothing.
HAPPYRAY_HD inline bool IntersectSceneIrregularGrid(const SceneView &scene,
                                                const IrregularGridView &grid,
                                                const Ray &ray,
                                                HitRecord *out_hit) {
  ClosestTriangleHit hit;
  detail::TraverseIrregularGrid(
      grid, ray, [&](const IrregularCell &cell, float *ray_t_max) -> bool {
        IntersectTrianglesClosest(scene, grid.triangle_indices, cell.ref_start, cell.ref_end,
                                  ray, ray_t_max, &hit);
        return false;
      });
  return ResolveClosestHit(scene, ray, hit, out_hit);
}

// Returns true as soon as any triangle in any irregular cell the ray visits occludes `ray`.
HAPPYRAY_HD inline bool IsOccludedIrregularGrid(const SceneView &scene,
                                            const IrregularGridView &grid,
                                            const Ray &ray) {
  return detail::TraverseIrregularGrid(
      grid, ray, [&](const IrregularCell &cell, float *) -> bool {
        return TestTrianglesInRangeOccluded(scene, grid.triangle_indices, cell.ref_start,
                                            cell.ref_end, ray);
      });
}

class IrregularGrid {
public:
  static constexpr float kDefaultTopLevelDensity = 0.12f;
  static constexpr float kDefaultSecondLevelDensity = 2.4f;
  // `scene` must point at memory reachable by whichever Thrust backend this
  // binary was built with (device memory for the CUDA backend, ordinary host
  // memory for the TBB backend).
  // `top_level_density`/`second_level_density`/`dense_cell_threshold` match
  // TwoLevelGrid's defaults and meaning (second_level_density sizes each
  // top-level cell's *finest* local grid before it gets rounded up to a power
  // of 2; dense_cell_threshold forces sparse cells straight to a 1x1x1 local
  // grid, since merging could never shrink them further anyway). `merge_alpha`
  // stops the merge pass once a round shrinks a top-level cell's local cell
  // count by less than this fraction (e.g. 0.7 => stop once a round shrinks it
  // by < 30%). `expand_iterations` is how many outer passes the expand pass
  // runs (each pass can grow a cell by one more layer of neighbors, so more
  // iterations let growth propagate further).
  explicit IrregularGrid(const SceneView &scene,
                         float top_level_density = kDefaultTopLevelDensity,
                         float second_level_density = kDefaultSecondLevelDensity,
                         uint32_t dense_cell_threshold = 16,
                         float merge_alpha = 0.7f, int expand_iterations = 2);
  ~IrregularGrid();
  IrregularGrid(const IrregularGrid &) = delete;
  IrregularGrid &operator=(const IrregularGrid &) = delete;

  IrregularGridView View() const;

  // Total number of distinct (post-merge/expand) IrregularCells, i.e. View().local_cells's own
  // element count -- not derivable from the view alone, since several finest-voxel
  // `local_entries` slots can resolve to the same cell. Used by callers that need to enumerate
  // every distinct cell exactly once (e.g. tests/accel_performance_tests.cpp's SAH cost
  // computation), mirroring UniformGrid::TriangleIndexCount()'s existing precedent.
  uint32_t LocalCellCount() const;

  // Remaining per-array element counts needed (alongside LocalCellCount() above and
  // View().top_level's own resolution_x/y/z) to compute an analytic memory-footprint estimate
  // (count * sizeof(element)) without any CUDA-specific allocation introspection -- see
  // tests/accel_performance_tests.cpp.
  uint32_t TopLevelTriangleIndexCount() const;
  uint32_t LocalEntryCount() const;
  uint32_t TriangleIndexCount() const;

  // Structural SAH cost estimate for this grid as currently built (see sah_cost.h), normalized
  // by `scene_bounds`'s own half surface area -- not from casting rays, but the same
  // surface-area-weighted "expected cost" heuristic the grid density search (see
  // src/app/grid_density_search.h) uses to prefer one candidate density over another. Unlike the
  // two-level grid, local cells here are individually sized (merge/expand grows some, leaves
  // others at the finest voxel size), so this genuinely needs one term per distinct surviving
  // cell -- LocalCellCount() above is exactly the count needed to enumerate `local_cells`
  // without re-visiting any cell once per finest voxel it now spans.
  float ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace happyray
