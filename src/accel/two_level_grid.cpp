// Thrust implementation of TwoLevelGrid (see two_level_grid.h). Grounded in javorkk/happyray's
// actual TLGridSortBuilder/TLGridBuildKernels: build the top level exactly like a UniformGrid,
// then -- for every top-level cell, not just "dense" ones -- compute a second-level resolution
// from that cell's own triangle count (forced to 1x1x1 when the count is at or below
// `dense_cell_threshold`, degenerating to a direct triangle list), and build all those
// second-level grids together as one combined sort-based CSR (count -> scan -> write -> sort ->
// adjacent-key boundary extraction -> CSR gap filling), the same technique as the top level,
// expressed purely as Thrust functors/algorithm calls -- so, like uniform_grid.cpp, this file
// compiles under either
// Thrust's CUDA backend (nvcc) or its TBB backend (plain g++); see CMakeLists.txt's
// HAPPYRAY_THRUST_BACKEND option.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "two_level_grid.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace happyray {
namespace {

// Computes top-level cell `c`'s second-level grid metadata (world bounds decoded from its
// (cx,cy,cz) coordinates, resolution sized from its own triangle count) -- everything except
// `cell_start_offset`, which is filled in afterwards once every cell's cell count is known (see
// SetCellStartOffsetFunctor below).
struct ComputeSecondLevelInfoFunctor {
    Vec3 top_bounds_min;
    Vec3 top_cell_size;
    int top_resolution_x;
    int top_resolution_y;
    const uint32_t* top_cell_start;
    float second_level_density;
    uint32_t dense_cell_threshold;

    HAPPYRAY_HD SecondLevelGridInfo operator()(uint32_t c) const {
        const int cz = static_cast<int>(c) / (top_resolution_x * top_resolution_y);
        const int rem = static_cast<int>(c) % (top_resolution_x * top_resolution_y);
        const int cy = rem / top_resolution_x;
        const int cx = rem % top_resolution_x;
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
        const Vec3 cell_max = cell_min + top_cell_size;
        const uint32_t triangle_count = top_cell_start[c + 1] - top_cell_start[c];

        const GridResolution res =
            (triangle_count <= dense_cell_threshold)
                ? GridResolution{1, 1, 1}
                : ComputeGridResolution(BBox(cell_min, cell_max), triangle_count,
                                       second_level_density);

        SecondLevelGridInfo info;
        info.resolution = res.x > res.y ? res.x : res.y;
        if (res.z > info.resolution) info.resolution = res.z;
        info.cell_start_offset = 0;
        return info;
    }
};

struct ExtractNumCellsFunctor {
    HAPPYRAY_HD uint32_t operator()(const SecondLevelGridInfo& info) const {
        return static_cast<uint32_t>(info.resolution) * info.resolution * info.resolution;
    }
};

struct SetCellStartOffsetFunctor {
    HAPPYRAY_HD SecondLevelGridInfo operator()(SecondLevelGridInfo info, uint32_t offset) const {
        info.cell_start_offset = offset;
        return info;
    }
};

// Writes top-level cell id `c` into every position of `owning_cell` that `c`'s own triangle
// range covers, so a later per-position pass can look up which cell (and therefore which
// second-level grid) each top-level (cell, triangle) pair belongs to.
struct ScatterOwningCellFunctor {
    const uint32_t* top_cell_start;
    uint32_t* owning_cell;

    HAPPYRAY_HD void operator()(uint32_t c) const {
        const uint32_t begin = top_cell_start[c];
        const uint32_t end = top_cell_start[c + 1];
        for (uint32_t i = begin; i < end; ++i) {
            owning_cell[i] = c;
        }
    }
};

// Counts how many of its owning top-level cell's second-level sub-cells a top-level
// (cell, triangle) pair's triangle *actually* intersects (full SAT test via
// ForEachIntersectedCell in uniform_grid.h, not just its AABB-derived cell range -- see that
// helper's comment for why the distinction matters for hair-strand-like geometry).
struct FragmentCounter {
    SceneView scene;
    UniformGridView top_level;
    const uint32_t* owning_cell;
    const uint32_t* top_triangle_indices;
    const SecondLevelGridInfo* infos;

    HAPPYRAY_HD uint32_t operator()(uint32_t i) const {
        const uint32_t owner = owning_cell[i];
        const SecondLevelGridInfo& info = infos[owner];
        const int cx = static_cast<int>(owner % top_level.resolution_x);
        const int cy = static_cast<int>((owner / top_level.resolution_x) %
                                        top_level.resolution_y);
        const int cz = static_cast<int>(owner / (top_level.resolution_x *
                                                 top_level.resolution_y));
        const Vec3 cell_min(top_level.bounds_min.x + cx * top_level.cell_size.x,
                            top_level.bounds_min.y + cy * top_level.cell_size.y,
                            top_level.bounds_min.z + cz * top_level.cell_size.z);
        const Vec3 cell_size(top_level.cell_size.x / info.resolution,
                     top_level.cell_size.y / info.resolution,
                     top_level.cell_size.z / info.resolution);
        const Vec3 inv_cell_size(top_level.inv_cell_size.x * info.resolution,
                     top_level.inv_cell_size.y * info.resolution,
                     top_level.inv_cell_size.z * info.resolution);
        const uint32_t tri = top_triangle_indices[i];
        const BBox box = ComputeTriangleBBox(scene, tri);
        uint32_t count = 0;
        detail::ForEachIntersectedCell(scene, tri, box, cell_min, inv_cell_size,
                                       cell_size, info.resolution, info.resolution,
                                       info.resolution,
                                       [&](int, int, int) { ++count; });
        return count;
    }
};

// Writes filtered fragments in each owning cell's second-level grid -- exactly
// `FragmentCounter`'s count for this pair, packed contiguously from `write_offsets[i]` (no
// invalid/sentinel entries to filter out afterwards).
struct FragmentWriter {
    SceneView scene;
    UniformGridView top_level;
    const uint32_t* owning_cell;
    const uint32_t* top_triangle_indices;
    const SecondLevelGridInfo* infos;
    const uint32_t* write_offsets;
    uint32_t* out_cell_indices;
    uint32_t* out_triangle_indices;

    HAPPYRAY_HD void operator()(uint32_t i) const {
        const uint32_t owner = owning_cell[i];
        const SecondLevelGridInfo& info = infos[owner];
        const int cx = static_cast<int>(owner % top_level.resolution_x);
        const int cy = static_cast<int>((owner / top_level.resolution_x) %
                                        top_level.resolution_y);
        const int cz = static_cast<int>(owner / (top_level.resolution_x *
                                                 top_level.resolution_y));
        const Vec3 cell_min(top_level.bounds_min.x + cx * top_level.cell_size.x,
                            top_level.bounds_min.y + cy * top_level.cell_size.y,
                            top_level.bounds_min.z + cz * top_level.cell_size.z);
        const Vec3 cell_size(top_level.cell_size.x / info.resolution,
                     top_level.cell_size.y / info.resolution,
                     top_level.cell_size.z / info.resolution);
        const Vec3 inv_cell_size(top_level.inv_cell_size.x * info.resolution,
                     top_level.inv_cell_size.y * info.resolution,
                     top_level.inv_cell_size.z * info.resolution);
        const uint32_t tri = top_triangle_indices[i];
        const BBox box = ComputeTriangleBBox(scene, tri);
        uint32_t offset = write_offsets[i];
        detail::ForEachIntersectedCell(
            scene, tri, box, cell_min, inv_cell_size, cell_size,
            info.resolution, info.resolution, info.resolution,
            [&](int cx, int cy, int cz) {
                out_cell_indices[offset] =
                    info.cell_start_offset +
                    FlattenCellIndex(cx, cy, cz, info.resolution, info.resolution);
                out_triangle_indices[offset] = tri;
                ++offset;
            });
    }
};

// Every top-level cell's own second-level grid is dense and uniformly sized within that one
// top-level cell (no merging), so its contribution to TwoLevelGrid::ComputeSahCost also
// collapses to a closed form from just its own resolution and the aggregate size of its CSR
// reference range -- one term per top-level cell, reduced via thrust::transform_reduce.
struct TwoLevelGridSahFunctor {
    TwoLevelGridView grid;
    SahCostConstants costs;

    HAPPYRAY_HD float operator()(uint32_t t) const {
        const SecondLevelGridInfo& info = grid.second_level_infos[t];
        const auto num_local = static_cast<uint32_t>(info.resolution) * info.resolution *
                               info.resolution;
        const uint32_t ref_count =
            grid.second_level_cell_start[info.cell_start_offset + num_local] -
            grid.second_level_cell_start[info.cell_start_offset];
        const Vec3 cell_size(grid.top_level.cell_size.x / info.resolution,
                 grid.top_level.cell_size.y / info.resolution,
                 grid.top_level.cell_size.z / info.resolution);
        return HalfSurfaceArea(cell_size) *
              (static_cast<float>(num_local) * costs.traversal +
               static_cast<float>(ref_count) * costs.intersection);
    }
};

}  // namespace

struct TwoLevelGrid::Impl {
    UniformGrid top_level;
    thrust::device_vector<SecondLevelGridInfo> second_level_infos;
    thrust::device_vector<uint32_t> second_level_cell_start;
    thrust::device_vector<uint32_t> second_level_triangle_indices;

    Impl(const SceneView& scene, float top_level_density) : top_level(scene, top_level_density) {}

    TwoLevelGridView View() const {
        TwoLevelGridView view;
        view.top_level = top_level.View();
        view.second_level_infos = thrust::raw_pointer_cast(second_level_infos.data());
        view.second_level_cell_start = thrust::raw_pointer_cast(second_level_cell_start.data());
        view.second_level_triangle_indices =
            thrust::raw_pointer_cast(second_level_triangle_indices.data());
        return view;
    }
};

TwoLevelGrid::TwoLevelGrid(const SceneView& scene, float top_level_density,
                          float second_level_density, uint32_t dense_cell_threshold)
    : impl_(std::make_unique<Impl>(scene, top_level_density)) {
    const UniformGridView top_view = impl_->top_level.View();
    const uint32_t num_top_cells =
        static_cast<uint32_t>(top_view.resolution_x) * top_view.resolution_y * top_view.resolution_z;
    const uint32_t total_top_level_pairs = impl_->top_level.TriangleIndexCount();

    // 1. Per-top-level-cell second-level grid metadata (resolution sized from that cell's own
    //    triangle count; forced to 1x1x1 -- a direct triangle list -- at or below the density
    //    threshold).
    thrust::device_vector<SecondLevelGridInfo> infos(num_top_cells);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(num_top_cells), infos.begin(),
                      ComputeSecondLevelInfoFunctor{top_view.bounds_min, top_view.cell_size,
                                                    top_view.resolution_x, top_view.resolution_y,
                                                    top_view.cell_start, second_level_density,
                                                    dense_cell_threshold});

    // 2. Exclusive-scan each cell's own sub-cell count into a global CSR offset, then fold that
    //    offset back into each cell's info (so SecondLevelView, in two_level_grid.h, can slice
    //    the shared arrays directly).
    thrust::device_vector<uint32_t> num_cells_per_top_cell(num_top_cells);
    thrust::transform(infos.begin(), infos.end(), num_cells_per_top_cell.begin(),
                      ExtractNumCellsFunctor());

    thrust::device_vector<uint32_t> second_cell_offset(num_top_cells);
    thrust::exclusive_scan(num_cells_per_top_cell.begin(), num_cells_per_top_cell.end(),
                          second_cell_offset.begin());
    const uint32_t total_second_level_cells =
        (num_top_cells > 0)
            ? (second_cell_offset.back() + num_cells_per_top_cell.back())
            : 0u;

    thrust::transform(infos.begin(), infos.end(), second_cell_offset.begin(), infos.begin(),
                      SetCellStartOffsetFunctor());

    // 3. For every top-level (cell, triangle) pair, find which cell it belongs to (so later
    //    steps can look up that cell's own second-level grid).
    thrust::device_vector<uint32_t> owning_cell(total_top_level_pairs);
    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(num_top_cells),
                     ScatterOwningCellFunctor{top_view.cell_start,
                                              thrust::raw_pointer_cast(owning_cell.data())});

    // 4. Per-pair overlapped-second-level-sub-cell counts, exclusive-scanned into write offsets
    //    -- same sort-based pattern as the top level, just over all cells' triangles at once.
    thrust::device_vector<uint32_t> counts(total_top_level_pairs);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(total_top_level_pairs), counts.begin(),
                      FragmentCounter{scene, top_view,
                                      thrust::raw_pointer_cast(owning_cell.data()),
                                      top_view.triangle_indices,
                                      thrust::raw_pointer_cast(infos.data())});

    thrust::device_vector<uint32_t> write_offsets(total_top_level_pairs);
    thrust::exclusive_scan(counts.begin(), counts.end(), write_offsets.begin());
    const uint32_t total_second_level_pairs =
        (total_top_level_pairs > 0) ? (write_offsets.back() + counts.back()) : 0u;

    // 5. Write fragments (exactly `counts[i]` of them per pair, no sentinel filtering needed).
    thrust::device_vector<uint32_t> pair_cell_indices(total_second_level_pairs);
    thrust::device_vector<uint32_t> pair_triangle_indices(total_second_level_pairs);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_level_pairs),
        FragmentWriter{scene, top_view, thrust::raw_pointer_cast(owning_cell.data()),
                       top_view.triangle_indices, thrust::raw_pointer_cast(infos.data()),
                       thrust::raw_pointer_cast(write_offsets.data()),
                       thrust::raw_pointer_cast(pair_cell_indices.data()),
                       thrust::raw_pointer_cast(pair_triangle_indices.data())});

    // 6. Sort by (global) cell index -- the defining "sort-based" step, covering every
    //    second-level grid at once.
    thrust::sort_by_key(pair_cell_indices.begin(), pair_cell_indices.end(),
                       pair_triangle_indices.begin());

    // 7. Extract occupied-cell starts from adjacent sorted keys, then fill empty-cell offsets.
    const uint32_t valid_pairs = static_cast<uint32_t>(
        thrust::lower_bound(pair_cell_indices.begin(), pair_cell_indices.end(),
                             total_second_level_cells) - pair_cell_indices.begin());
    pair_triangle_indices.resize(valid_pairs);
    if (valid_pairs < total_second_level_pairs) pair_triangle_indices.shrink_to_fit();
    impl_->second_level_cell_start.assign(total_second_level_cells + 1, valid_pairs);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(valid_pairs),
        detail::CellExtractor{thrust::raw_pointer_cast(pair_cell_indices.data()),
                              thrust::raw_pointer_cast(impl_->second_level_cell_start.data())});
    thrust::inclusive_scan(impl_->second_level_cell_start.rbegin(),
                           impl_->second_level_cell_start.rend(),
                           impl_->second_level_cell_start.rbegin(), thrust::minimum<uint32_t>());

    impl_->second_level_infos = std::move(infos);
    impl_->second_level_triangle_indices = std::move(pair_triangle_indices);
}

TwoLevelGrid::~TwoLevelGrid() = default;

uint32_t TwoLevelGrid::TopLevelTriangleIndexCount() const {
    return impl_->top_level.TriangleIndexCount();
}

uint32_t TwoLevelGrid::SecondLevelCellCount() const {
    return static_cast<uint32_t>(impl_->second_level_cell_start.size()) - 1;
}

uint32_t TwoLevelGrid::SecondLevelTriangleIndexCount() const {
    return static_cast<uint32_t>(impl_->second_level_triangle_indices.size());
}

float TwoLevelGrid::ComputeSahCost(const BBox& scene_bounds, const SahCostConstants& costs) const {
    const TwoLevelGridView view = impl_->View();
    const auto num_top_cells = static_cast<uint32_t>(view.top_level.resolution_x) *
                               view.top_level.resolution_y * view.top_level.resolution_z;
    float total = HalfSurfaceArea(view.top_level.cell_size) *
                  static_cast<float>(num_top_cells) * costs.traversal;
    if (num_top_cells > 0) {
        total += thrust::transform_reduce(
            thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(num_top_cells),
            TwoLevelGridSahFunctor{view, costs}, 0.0f, thrust::plus<float>());
    }
    return total / HalfSurfaceArea(scene_bounds.Extent());
}

TwoLevelGridView TwoLevelGrid::View() const { return impl_->View(); }

}  // namespace happyray
