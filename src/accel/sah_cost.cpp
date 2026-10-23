// Implements MeasureSahCalibration() (see sah_cost.h): the "heuristic test"
// that estimates each acceleration structure's own traversal-step cost and one
// shared ray/triangle intersection-test cost, in nanoseconds, by timing the
// real routines against small synthetic test cases -- mirrors (and is shared
// by) tests/accel_performance_tests.cpp's own calibration. Pure Thrust (no raw
// CUDA runtime API, no CUDA-specific launch syntax), so this file compiles
// either under Thrust's CUDA backend (nvcc, runs on the GPU) or its TBB backend
// (plain g++, runs in parallel on the CPU) -- see CMakeLists.txt's
// HAPPYRAY_THRUST_BACKEND option, and src/render/renderer.cpp's own header comment
// for the same pattern.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "sah_cost.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "bbox.h"
#include "irregular_grid.h"
#include "random.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace happyray {
namespace {

// 4x4x4 top-level cells, each holding a 16x16x16 second-level/local grid (so
// the two-level and irregular mock grids' own top-level 3D-DDA genuinely steps
// across multiple top-level cells, not just a single trivial one) -- their
// product is the uniform grid mock's own flat resolution too, so every mock
// grid spans the same overall bounds at the same overall finest granularity.
constexpr int kMockTopLevelResolution =
    4; // Top-level cells per axis (two-level/irregular).
constexpr int kMockSecondLevelResolution =
    16; // Second-level/local cells per axis, per top cell.
constexpr int kMockResolution =
    kMockTopLevelResolution * kMockSecondLevelResolution;
// The irregular mock grid keeps the same 4^3 x 16^3 layout, but pretends it is
// the result of merging/expanding a 2x denser (per axis) initial grid: each
// leaf cell replaces 2x2x2 = 8 of those initial cells, but a single ray is
// unlikely to cross all 8 of them, so each visited leaf cell is counted as 4
// traversal steps.
constexpr uint32_t kMockIrregularCellsPerLeaf = 4;
// One ray batch, generated once and shared by every traversal and intersection
// measurement below (see GenerateUnitCubeRays) -- not regenerated inside each
// timed functor's operator(), so neither benchmark pays for ray construction,
// and both read the exact same, already-resident rays from device memory the
// way real traversal code reads its own ray batch, hopefully reflecting
// realistic memory access patterns.
constexpr uint32_t kCalibrationRayCount = 1u << 17;
// Triangle count for the intersection benchmark's own mesh (see
// BuildIntersectionMesh below).
constexpr int kIntersectionTriangleCount = 100;

// Generates `count` rays, each spanning the unit box [0,1]^3 from a random
// point on one face straight through to a random point on the *opposite* face
// (a uniformly random choice of which axis is the "entry/exit" one) -- so every
// ray genuinely crosses the whole box on at least one axis (and, depending on
// the other two randomly chosen endpoints, anywhere from a nearly axis-aligned
// sweep to a long corner-to-corner diagonal), rather than only skimming a small
// corner of it. Built once, on the host (not HAPPYRAY_HD/device code -- this runs
// long before any timed region, so there's no need to run it in parallel on the
// GPU), then uploaded by the caller; see kCalibrationRayCount's own comment
// above for why.
std::vector<Ray> GenerateUnitCubeRays(uint64_t seed, uint32_t count) {
  std::vector<Ray> rays(count);
  for (uint32_t i = 0; i < count; ++i) {
    Rng rng(seed, i);
    const int axis = static_cast<int>(rng.NextUint32() % 3u);
    float near_t[3] = {rng.NextFloat(), rng.NextFloat(), rng.NextFloat()};
    float far_t[3] = {rng.NextFloat(), rng.NextFloat(), rng.NextFloat()};
    near_t[axis] = 0.0f;
    far_t[axis] = 1.0f;
    const Vec3 near_point(near_t[0], near_t[1], near_t[2]);
    const Vec3 far_point(far_t[0], far_t[1], far_t[2]);
    Ray ray(near_point, far_point - near_point);
    ray.t_min = -1.0e-3f; // The origin sits exactly on the entry face; a hair
                          // of negative t_min keeps it unambiguously "inside"
                          // despite float rounding.
    ray.t_max =
        1.0f +
        1.0e-3f; // direction already spans exactly the entry->exit segment.
    rays[i] = ray;
  }
  return rays;
}

// Builds a UniformGridView over `bounds` with exactly `resolution` cells per
// axis and zero triangles anywhere -- deliberately *not* built via
// UniformGrid's own density-based constructor, since that always collapses an
// empty scene straight to a trivial 1x1x1 grid.
struct MockUniformGrid {
  thrust::device_vector<uint32_t> cell_start;
  UniformGridView view;

  MockUniformGrid(const BBox &bounds, int resolution) {
    const auto num_cells =
        static_cast<uint32_t>(resolution) * resolution * resolution;
    view.bounds_min = bounds.min_bound;
    view.bounds_max = bounds.max_bound;
    const Vec3 extent = bounds.Extent();
    view.cell_size = Vec3(extent.x / resolution, extent.y / resolution,
                          extent.z / resolution);
    view.inv_cell_size = Vec3(1.0f / view.cell_size.x, 1.0f / view.cell_size.y,
                              1.0f / view.cell_size.z);
    view.resolution_x = view.resolution_y = view.resolution_z = resolution;
    cell_start.assign(num_cells + 1, 0u);
    view.cell_start = thrust::raw_pointer_cast(cell_start.data());
    view.triangle_indices = nullptr;
  }
};

// Builds a SecondLevelGridInfo for top-level cell `t` of a top_resolution^3
// top-level grid, each with its own second_resolution^3 second-level grid --
// the per-top-level-cell bounds/CSR- offset variant of a single whole-bounds
// top-level cell, used by MockTwoLevelGrid below so its top-level 3D-DDA
// genuinely steps across multiple top-level cells.
struct MakeMockSecondLevelInfoFunctor {
  Vec3 top_bounds_min;
  Vec3 top_cell_size;
  int top_resolution_x, top_resolution_y;
  int second_resolution;

  HAPPYRAY_HD SecondLevelGridInfo operator()(uint32_t t) const {
    const int res_xy = top_resolution_x * top_resolution_y;
    const int tz = static_cast<int>(t) / res_xy;
    const int rem = static_cast<int>(t) % res_xy;
    const int ty = rem / top_resolution_x;
    const int tx = rem % top_resolution_x;
    const Vec3 cell_min(top_bounds_min.x + tx * top_cell_size.x,
                        top_bounds_min.y + ty * top_cell_size.y,
                        top_bounds_min.z + tz * top_cell_size.z);
    const Vec3 cell_max = cell_min + top_cell_size;

    SecondLevelGridInfo info;
    info.bounds_min = cell_min;
    info.bounds_max = cell_max;
    info.cell_size = Vec3(top_cell_size.x / second_resolution,
                          top_cell_size.y / second_resolution,
                          top_cell_size.z / second_resolution);
    info.inv_cell_size = Vec3(1.0f / info.cell_size.x, 1.0f / info.cell_size.y,
                              1.0f / info.cell_size.z);
    info.resolution_x = info.resolution_y = info.resolution_z =
        second_resolution;
    const auto num_local = static_cast<uint32_t>(second_resolution) *
                           second_resolution * second_resolution;
    info.cell_start_offset = t * num_local;
    return info;
  }
};

// Same idea as MockUniformGrid, but with a `top_resolution`-cubed top-level
// grid, each cell recursing into its own `second_resolution`-cubed second-level
// grid -- exactly TwoLevelGrid's own shape, just with manually-populated
// SecondLevelGridInfo entries instead of ones chosen by density.
struct MockTwoLevelGrid {
  thrust::device_vector<SecondLevelGridInfo> infos;
  thrust::device_vector<uint32_t> top_cell_start;
  thrust::device_vector<uint32_t> cell_start;
  TwoLevelGridView view;

  MockTwoLevelGrid(const BBox &bounds, int top_resolution,
                   int second_resolution) {
    const auto num_top_cells =
        static_cast<uint32_t>(top_resolution) * top_resolution * top_resolution;
    const auto num_local = static_cast<uint32_t>(second_resolution) *
                           second_resolution * second_resolution;
    const Vec3 extent = bounds.Extent();
    const Vec3 top_cell_size(extent.x / top_resolution,
                             extent.y / top_resolution,
                             extent.z / top_resolution);

    view.top_level.bounds_min = bounds.min_bound;
    view.top_level.bounds_max = bounds.max_bound;
    view.top_level.cell_size = top_cell_size;
    view.top_level.inv_cell_size = Vec3(
        1.0f / top_cell_size.x, 1.0f / top_cell_size.y, 1.0f / top_cell_size.z);
    view.top_level.resolution_x = view.top_level.resolution_y =
        view.top_level.resolution_z = top_resolution;
    // One (never-read) triangle per top-level cell, so traversal descends into
    // every one instead of skipping it as empty -- the second-level CSR below
    // stays empty.
    top_cell_start.resize(num_top_cells + 1);
    thrust::sequence(top_cell_start.begin(), top_cell_start.end());
    view.top_level.cell_start = thrust::raw_pointer_cast(top_cell_start.data());
    view.top_level.triangle_indices = nullptr;

    infos.resize(num_top_cells);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells), infos.begin(),
        MakeMockSecondLevelInfoFunctor{bounds.min_bound, top_cell_size,
                                       top_resolution, top_resolution,
                                       second_resolution});

    cell_start.assign(num_top_cells * num_local + 1, 0u);

    view.second_level_infos = thrust::raw_pointer_cast(infos.data());
    view.second_level_cell_start = thrust::raw_pointer_cast(cell_start.data());
    view.second_level_triangle_indices = nullptr;
  }
};

// Builds the pre-merge (dense, identity-entries, zero-reference) IrregularCell
// for finest voxel `v` of top-level cell `t`'s own local grid -- same shape
// MakeInitialCellFunctor (irregular_grid.cpp) produces, just in this file's own
// mock bounds/lattice. `min_x/y/z` are in the grid-wide shared lattice (see
// irregular_grid.h's file header comment), i.e. offset by `t`'s own position in
// the top-level grid -- not just its local voxel coordinates -- exactly like
// the real per-top-level-cell IrregularCellInfo::global_offset_* this mirrors.
struct MakeMockIrregularCellFunctor {
  int top_resolution_x, top_resolution_y;
  int local_resolution;

  HAPPYRAY_HD IrregularCell operator()(uint32_t v) const {
    const auto num_local = static_cast<uint32_t>(local_resolution) *
                           local_resolution * local_resolution;
    const uint32_t t = v / num_local;
    const uint32_t local_v = v % num_local;

    const int local_res_xy = local_resolution * local_resolution;
    const int lz = static_cast<int>(local_v) / local_res_xy;
    const int lrem = static_cast<int>(local_v) % local_res_xy;
    const int ly = lrem / local_resolution;
    const int lx = lrem % local_resolution;

    const int top_res_xy = top_resolution_x * top_resolution_y;
    const int tz = static_cast<int>(t) / top_res_xy;
    const int trem = static_cast<int>(t) % top_res_xy;
    const int ty = trem / top_resolution_x;
    const int tx = trem % top_resolution_x;

    IrregularCell cell;
    cell.min_x = tx * local_resolution + lx;
    cell.min_y = ty * local_resolution + ly;
    cell.min_z = tz * local_resolution + lz;
    cell.max_x = cell.min_x + 1;
    cell.max_y = cell.min_y + 1;
    cell.max_z = cell.min_z + 1;
    cell.ref_start = 0;
    cell.ref_end = 0;
    return cell;
  }
};

// Builds an IrregularCellInfo for top-level cell `t` of a top_resolution^3
// top-level grid, each with its own local_resolution^3 local grid
// (global_resolution == local_resolution and shift_*==0 for all of them, since
// every top-level cell here has the same local resolution -- no
// coarsest-common-refinement rounding needed, unlike the real density-driven
// construction).
struct MakeMockIrregularCellInfoFunctor {
  Vec3 top_bounds_min;
  Vec3 top_cell_size;
  int top_resolution_x, top_resolution_y;
  int local_resolution;

  HAPPYRAY_HD IrregularCellInfo operator()(uint32_t t) const {
    const int res_xy = top_resolution_x * top_resolution_y;
    const int tz = static_cast<int>(t) / res_xy;
    const int rem = static_cast<int>(t) % res_xy;
    const int ty = rem / top_resolution_x;
    const int tx = rem % top_resolution_x;
    const Vec3 cell_min(top_bounds_min.x + tx * top_cell_size.x,
                        top_bounds_min.y + ty * top_cell_size.y,
                        top_bounds_min.z + tz * top_cell_size.z);

    IrregularCellInfo info;
    info.bounds_min = cell_min;
    info.cell_size = Vec3(top_cell_size.x / local_resolution,
                          top_cell_size.y / local_resolution,
                          top_cell_size.z / local_resolution);
    info.inv_cell_size = Vec3(1.0f / info.cell_size.x, 1.0f / info.cell_size.y,
                              1.0f / info.cell_size.z);
    info.resolution_x = info.resolution_y = info.resolution_z =
        local_resolution;
    const auto num_local = static_cast<uint32_t>(local_resolution) *
                           local_resolution * local_resolution;
    info.entries_offset = t * num_local;
    info.shift_x = info.shift_y = info.shift_z = 0;
    info.global_offset_x = tx * local_resolution;
    info.global_offset_y = ty * local_resolution;
    info.global_offset_z = tz * local_resolution;
    return info;
  }
};

// Same idea again, but shaped like IrregularGrid's own (top_resolution-cubed
// top-level grid, each with its own dense local grid). The mock cells
// themselves are not merged; instead IrregularStepFunctor counts each visited
// cell as kMockIrregularCellsPerLeaf steps, modelling leaf cells that each
// replace 2x2x2 cells of a denser initial grid (of which a ray crosses about
// 4).
struct MakeMockTopCellFunctor {
  HAPPYRAY_HD IrregularTopCell operator()(const IrregularCellInfo &info) const {
    return MakeIrregularTopCell(info);
  }
};

struct MockIrregularGrid {
  thrust::device_vector<IrregularCellInfo> infos;
  thrust::device_vector<IrregularTopCell> top_cells;
  thrust::device_vector<uint32_t> entries;
  thrust::device_vector<IrregularCell> cells;
  IrregularGridView view;

  MockIrregularGrid(const BBox &bounds, int top_resolution,
                    int local_resolution) {
    const auto num_top_cells =
        static_cast<uint32_t>(top_resolution) * top_resolution * top_resolution;
    const auto num_local_per_top = static_cast<uint32_t>(local_resolution) *
                                   local_resolution * local_resolution;
    const uint32_t num_entries = num_top_cells * num_local_per_top;
    const Vec3 extent = bounds.Extent();
    const Vec3 top_cell_size(extent.x / top_resolution,
                             extent.y / top_resolution,
                             extent.z / top_resolution);

    view.top_level.bounds_min = bounds.min_bound;
    view.top_level.bounds_max = bounds.max_bound;
    view.top_level.cell_size = top_cell_size;
    view.top_level.inv_cell_size = Vec3(
        1.0f / top_cell_size.x, 1.0f / top_cell_size.y, 1.0f / top_cell_size.z);
    view.top_level.resolution_x = view.top_level.resolution_y =
        view.top_level.resolution_z = top_resolution;
    view.top_level.cell_start = nullptr;
    view.top_level.triangle_indices = nullptr;

    infos.resize(num_top_cells);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells), infos.begin(),
        MakeMockIrregularCellInfoFunctor{bounds.min_bound, top_cell_size,
                                         top_resolution, top_resolution,
                                         local_resolution});

    top_cells.resize(num_top_cells);
    thrust::transform(infos.begin(), infos.end(), top_cells.begin(),
                      MakeMockTopCellFunctor());

    entries.resize(num_entries);
    thrust::sequence(entries.begin(), entries.end());
    cells.resize(num_entries);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(num_entries),
                      cells.begin(),
                      MakeMockIrregularCellFunctor{
                          top_resolution, top_resolution, local_resolution});

    view.cell_infos = thrust::raw_pointer_cast(infos.data());
    view.top_cells = thrust::raw_pointer_cast(top_cells.data());
    view.local_entries = thrust::raw_pointer_cast(entries.data());
    view.local_cells = thrust::raw_pointer_cast(cells.data());
    view.triangle_indices = nullptr;
    view.global_cell_size = Vec3(top_cell_size.x / local_resolution,
                                 top_cell_size.y / local_resolution,
                                 top_cell_size.z / local_resolution);
    view.global_resolution_x = view.global_resolution_y =
        view.global_resolution_z = local_resolution;
    view.global_inv_cell_size =
        Vec3(1.0f / view.global_cell_size.x, 1.0f / view.global_cell_size.y,
             1.0f / view.global_cell_size.z);
    view.global_shift_x = view.global_shift_y = view.global_shift_z =
        IrregularLog2(local_resolution);
  }
};

// Counts traversal steps only (every visited cell, top- and/or second-level,
// increments the count -- by kMockIrregularCellsPerLeaf for the irregular grid
// -- and the walk never stops early) -- one functor per structure, each calling
// that structure's own real traversal code exactly as
// IntersectSceneGrid/IntersectSceneTwoLevelGrid/ IntersectSceneIrregularGrid
// do, just swapping the triangle-testing callback for a counter. `rays` is the
// one pregenerated, shared ray batch (see GenerateUnitCubeRays/
// kCalibrationRayCount above) -- read by index, not recomputed here.
struct UniformStepFunctor {
  UniformGridView grid;
  const Ray *rays;

  HAPPYRAY_HD float operator()(uint32_t i) const {
    uint32_t steps = 0;
    detail::TraverseGrid(grid, rays[i], [&](uint32_t, float *) -> bool {
      ++steps;
      return false;
    });
    return static_cast<float>(steps);
  }
};

struct TwoLevelStepFunctor {
  TwoLevelGridView grid;
  const Ray *rays;

  HAPPYRAY_HD float operator()(uint32_t i) const {
    uint32_t steps = 0;
    detail::TraverseTwoLevelGrid(
        grid, rays[i], [&](uint32_t) { ++steps; },
        [&](uint32_t, float *) -> bool {
          ++steps;
          return false;
        });
    return static_cast<float>(steps);
  }
};

struct IrregularStepFunctor {
  IrregularGridView grid;
  const Ray *rays;

  HAPPYRAY_HD float operator()(uint32_t i) const {
    uint32_t steps = 0;
    detail::TraverseIrregularGrid(grid, rays[i],
                                  [&](const IrregularCell &, float *) -> bool {
                                    steps += kMockIrregularCellsPerLeaf;
                                    return false;
                                  });
    return static_cast<float>(steps);
  }
};

// Runs `functor` over `count` indices (one per calibration ray) via a sum
// reduction -- so the whole measurement needs only O(1) device memory
// regardless of `count` -- timed with std::chrono wall-clock timing.
// thrust::transform_reduce's own return value (the reduction's sum) must be
// copied back to host before it can return at all, so -- unlike
// thrust::transform/thrust::for_each, which can launch and return immediately
// on the CUDA backend -- it's already an implicit synchronization point, with
// no separate sync call needed. By default, divides by the *sum* the reduction
// returns (total steps or total tests actually performed across every ray), not
// by `count` itself, since individual rays cross a different number of cells
// depending on their random direction -- what's wanted is the average
// per-step/per-test cost, not the average per-ray one. `known_unit_count`, if
// positive, overrides that divisor with an explicit, already-known unit count
// instead (used by MeasureIntersectionCostNs below, whose functor returns each
// test's own computed `t` -- precisely so the real IntersectRayTriangle result
// can't be optimized away as dead code -- rather than a per-test constant
// suitable for summing into a unit count itself); the reduction still runs and
// is still checked for validity either way, so the real computation it forces
// is never skipped. Runs a throwaway warm-up trial first (the very first launch
// of any given kernel can be dominated by one-time JIT/module-load cost) plus a
// few more timed trials, keeping the minimum -- standard microbenchmark
// practice, since scheduling noise can only inflate a trial's time, never
// deflate it below the routine's true cost. Returns nanoseconds per unit.
template <typename Functor>
float TimePerUnitNs(Functor functor, uint32_t count,
                    float known_unit_count = 0.0f) {
  constexpr int kWarmupTrials = 1;
  constexpr int kTimedTrials = 2;
  float best_ms = 1e30f;
  float total_units = 0.0f;
  for (int trial = 0; trial < kWarmupTrials + kTimedTrials; ++trial) {
    const auto begin = std::chrono::steady_clock::now();
    const float units =
        thrust::transform_reduce(thrust::counting_iterator<uint32_t>(0),
                                 thrust::counting_iterator<uint32_t>(count),
                                 functor, 0.0f, thrust::plus<float>());
    const auto finish = std::chrono::steady_clock::now();
    const float elapsed_ms =
        std::chrono::duration<float, std::milli>(finish - begin).count();
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0 ||
        !std::isfinite(units) || units <= 0)
      throw std::runtime_error("Invalid measured calibration duration");
    total_units = known_unit_count > 0 ? known_unit_count : units;
    if (trial >= kWarmupTrials)
      best_ms = std::min(best_ms, elapsed_ms);
  }
  return best_ms * 1.0e6f / total_units;
}

// One triangle of the diagonal-blocking mesh (see BuildIntersectionMesh below),
// used only while building it on the host.
struct HostTriangle {
  Vec3 a, b, c;
};

// Appends the 6 "diagonal" quads connecting opposite edges of `bounds`, each
// split into 2 triangles (12 total) -- one quad per pair of opposite parallel
// edges, in each of the 3 axis directions, both diagonal orientations --
// forming a symmetric cage of cutting planes through the box's interior (planes
// x=y, x+y=1, y=z, y+z=1, x=z, x+z=1 for a unit box). A single such plane only
// blocks rays that happen to cross *that* one; six of them, in every
// orientation, means a large fraction of rays genuinely crossing the box -- not
// just ones aimed at one specific plane -- hit at least one, without needing to
// bias ray generation toward them at all.
void AppendDiagonalQuads(const BBox &bounds, std::vector<HostTriangle> *out) {
  const Vec3 &lo = bounds.min_bound;
  const Vec3 &hi = bounds.max_bound;
  auto corner = [&](int bx, int by, int bz) {
    return Vec3(bx ? hi.x : lo.x, by ? hi.y : lo.y, bz ? hi.z : lo.z);
  };
  auto add_quad = [&](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
    out->push_back({p0, p1, p2});
    out->push_back({p0, p2, p3});
  };
  // z-parallel edges: main diagonal (plane x=y), anti-diagonal (plane x+y=1).
  add_quad(corner(0, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(0, 0, 1));
  add_quad(corner(1, 0, 0), corner(0, 1, 0), corner(0, 1, 1), corner(1, 0, 1));
  // x-parallel edges: main diagonal (plane y=z), anti-diagonal (plane y+z=1).
  add_quad(corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 1), corner(0, 1, 1));
  add_quad(corner(0, 1, 0), corner(1, 1, 0), corner(1, 0, 1), corner(0, 0, 1));
  // y-parallel edges: main diagonal (plane x=z), anti-diagonal (plane x+z=1).
  add_quad(corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 1), corner(1, 0, 1));
  add_quad(corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 1), corner(0, 0, 1));
}

// Recursively subdivides (longest-edge bisection of the largest-area triangle,
// repeated) until `triangles` holds exactly `target_count` entries -- a simple
// way to grow a coarse mesh to an arbitrary finer triangle count while keeping
// it reasonably uniform.
void SubdivideTriangles(std::vector<HostTriangle> *triangles,
                        int target_count) {
  while (static_cast<int>(triangles->size()) < target_count) {
    size_t largest = 0;
    float largest_area_squared = -1.0f;
    for (size_t i = 0; i < triangles->size(); ++i) {
      const HostTriangle &t = (*triangles)[i];
      const float area_squared = LengthSquared(Cross(t.b - t.a, t.c - t.a));
      if (area_squared > largest_area_squared) {
        largest_area_squared = area_squared;
        largest = i;
      }
    }
    const HostTriangle t = (*triangles)[largest];
    const float ab = LengthSquared(t.a - t.b);
    const float bc = LengthSquared(t.b - t.c);
    const float ca = LengthSquared(t.c - t.a);
    HostTriangle t1, t2;
    if (ab >= bc && ab >= ca) {
      const Vec3 m = (t.a + t.b) * 0.5f;
      t1 = {t.a, m, t.c};
      t2 = {m, t.b, t.c};
    } else if (bc >= ca) {
      const Vec3 m = (t.b + t.c) * 0.5f;
      t1 = {t.a, t.b, m};
      t2 = {t.a, m, t.c};
    } else {
      const Vec3 m = (t.c + t.a) * 0.5f;
      t1 = {t.a, t.b, m};
      t2 = {m, t.b, t.c};
    }
    (*triangles)[largest] = t1;
    triangles->push_back(t2);
  }
}

// Flat positions/indices arrays (3 unique vertices per triangle, no sharing
// needed for a benchmark mesh this small), matching the same layout real scene
// triangle data uses (SceneView::positions/indices).
struct HostMesh {
  std::vector<Vec3> positions;
  std::vector<uint32_t> indices;
};

// Builds a kIntersectionTriangleCount-triangle mesh blocking `bounds`'s own
// diagonal cross-sections (see AppendDiagonalQuads above), refined to that
// exact count via SubdivideTriangles.
HostMesh BuildIntersectionMesh(const BBox &bounds) {
  std::vector<HostTriangle> triangles;
  AppendDiagonalQuads(bounds, &triangles);
  SubdivideTriangles(&triangles, kIntersectionTriangleCount);

  HostMesh mesh;
  mesh.positions.reserve(triangles.size() * 3);
  mesh.indices.reserve(triangles.size() * 3);
  for (const HostTriangle &t : triangles) {
    const auto base = static_cast<uint32_t>(mesh.positions.size());
    mesh.positions.push_back(t.a);
    mesh.positions.push_back(t.b);
    mesh.positions.push_back(t.c);
    mesh.indices.push_back(base + 0);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);
  }
  return mesh;
}

// Performs kIntersectionTriangleCount real IntersectRayTriangle calls per ray
// -- one per triangle in the diagonal-blocking mesh (see BuildIntersectionMesh
// above) -- the exact same routine every structure's own
// IntersectTrianglesClosest ends up calling per reference, so (unlike
// traversal-step cost, which does genuinely differ per structure) this one
// calibration applies equally to all three. Every triangle is tested against
// the ray's own original [t_min, t_max] range (not narrowed as closer hits are
// found, unlike IntersectTrianglesClosest's own progressive-narrowing
// optimization), so every one of the kIntersectionTriangleCount tests pays the
// same, representative cost regardless of where in the loop a hit happens to be
// found. `rays`/`positions`/`indices` are the shared, pregenerated ray batch
// and mesh (see GenerateUnitCubeRays/BuildIntersectionMesh above), read by
// index, not recomputed here. Returns the closest hit's `t` (or the ray's own
// `t_max` if none hit) rather than a constant, so the real IntersectRayTriangle
// results can't be proven unused and optimized away; TimePerUnitNs is still
// told the true, already-known unit count explicitly (ray count times triangle
// count) rather than normalizing by summing these `t` values, which aren't a
// meaningful substitute for a test count.
struct IntersectionTestFunctor {
  const Ray *rays;
  const Vec3 *positions;
  const uint32_t *indices;

  HAPPYRAY_HD float operator()(uint32_t i) const {
    const Ray &ray = rays[i];
    float closest_t = ray.t_max;
    for (int tri = 0; tri < kIntersectionTriangleCount; ++tri) {
      const Vec3 &v0 = positions[indices[tri * 3 + 0]];
      const Vec3 &v1 = positions[indices[tri * 3 + 1]];
      const Vec3 &v2 = positions[indices[tri * 3 + 2]];
      float t, u, v;
      if (IntersectRayTriangle(ray, v0, v1, v2, &t, &u, &v) && t < closest_t)
        closest_t = t;
    }
    return closest_t;
  }
};

// Measures the one, shared, structure-independent intersection-test cost
// directly (see IntersectionTestFunctor above), rather than inferring it as the
// marginal difference between two separately-measured, similarly-scaled grid
// traversals -- the latter approach turned out to be noise-prone whenever a
// structure's true marginal intersection cost was itself small relative to its
// own (possibly much larger) traversal cost.
float MeasureIntersectionCostNs(const thrust::device_vector<Ray> &rays) {
  const BBox bounds(Vec3(0, 0, 0), Vec3(1, 1, 1));
  const HostMesh mesh = BuildIntersectionMesh(bounds);
  const thrust::device_vector<Vec3> positions(mesh.positions);
  const thrust::device_vector<uint32_t> indices(mesh.indices);
  const auto ray_count = static_cast<uint32_t>(rays.size());
  const float intersection_ns = TimePerUnitNs(
      IntersectionTestFunctor{thrust::raw_pointer_cast(rays.data()),
                              thrust::raw_pointer_cast(positions.data()),
                              thrust::raw_pointer_cast(indices.data())},
      ray_count, static_cast<float>(ray_count) * kIntersectionTriangleCount);
  if (!std::isfinite(intersection_ns) || intersection_ns <= 0)
    throw std::runtime_error("Invalid measured intersection cost");
  return intersection_ns;
}

// Measures each structure's own traversal-step cost against a mock,
// empty-space, fixed-resolution grid (see MockUniformGrid et al. above) over
// the same unit box the shared ray batch spans -- deliberately *not* shared
// across the three functions below, so a structure's own extra indirection (or
// lack of it) shows up as a genuinely different cost rather than being averaged
// away.
float MeasureUniformTraversalNs(const thrust::device_vector<Ray> &rays) {
  const BBox bounds(Vec3(0, 0, 0), Vec3(1, 1, 1));
  const MockUniformGrid grid(bounds, kMockResolution);
  return TimePerUnitNs(
      UniformStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
      static_cast<uint32_t>(rays.size()));
}

float MeasureTwoLevelTraversalNs(const thrust::device_vector<Ray> &rays) {
  const BBox bounds(Vec3(0, 0, 0), Vec3(1, 1, 1));
  const MockTwoLevelGrid grid(bounds, kMockTopLevelResolution,
                              kMockSecondLevelResolution);
  return TimePerUnitNs(
      TwoLevelStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
      static_cast<uint32_t>(rays.size()));
}

float MeasureIrregularTraversalNs(const thrust::device_vector<Ray> &rays) {
  const BBox bounds(Vec3(0, 0, 0), Vec3(1, 1, 1));
  const MockIrregularGrid grid(bounds, kMockTopLevelResolution,
                               kMockSecondLevelResolution);
  return TimePerUnitNs(
      IrregularStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
      static_cast<uint32_t>(rays.size()));
}

} // namespace

SahCalibration MeasureSahCalibration() {
  const thrust::device_vector<Ray> rays(
      GenerateUnitCubeRays(0x1234ULL, kCalibrationRayCount));
  SahCalibration calibration;
  const float intersection_ns = MeasureIntersectionCostNs(rays);
  calibration.uniform = {MeasureUniformTraversalNs(rays), intersection_ns};
  calibration.two_level = {MeasureTwoLevelTraversalNs(rays), intersection_ns};
  calibration.irregular = {MeasureIrregularTraversalNs(rays), intersection_ns};
  return calibration;
}

void PrintSahCalibration(const SahCalibration &calibration) {
  auto print = [](const char *name, const SahCostConstants &costs) {
    std::printf("  %-12s traversal %8.3f ns/step, intersection %8.3f ns/test "
                "(ratio %.3f)\n",
                name, costs.traversal, costs.intersection,
                costs.intersection / costs.traversal);
  };
  std::printf(
      "Measured SAH cost constants (happyray::MeasureSahCalibration(), see"
      " src/accel/sah_cost.h): traversal ns/step, intersection ns/test:\n");
  print("uniform", calibration.uniform);
  print("two-level", calibration.two_level);
  print("irregular", calibration.irregular);
}

} // namespace happyray
