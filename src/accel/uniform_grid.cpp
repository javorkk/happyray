// Thrust implementation of UniformGrid (see uniform_grid.h): count exact SAT-overlapped cells,
// write (cell, triangle) pairs, sort by cell, and extract each cell's explicit range from
// adjacent keys. The builder is expressed purely as Thrust
// functors/algorithm calls, with no CUDA-specific launch syntax. That is what lets this exact
// file compile either under Thrust's CUDA backend (nvcc, builds the grid on the GPU) or its TBB
// backend (plain g++, builds it in parallel on the CPU); see CMakeLists.txt's
// HAPPYRAY_THRUST_BACKEND option.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "uniform_grid.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace happyray {
namespace {

struct ComputeTriangleBBoxFunctor {
    SceneView scene;
    HAPPYRAY_HD BBox operator()(uint32_t tri) const { return ComputeTriangleBBox(scene, tri); }
};

struct BBoxUnionFunctor {
    HAPPYRAY_HD BBox operator()(const BBox& a, const BBox& b) const {
        BBox result = a;
        result.Grow(b);
        return result;
    }
};

// Counts how many cells a triangle *actually* intersects (full SAT test via
// ForEachIntersectedCell in uniform_grid.h, not just its AABB-derived cell range -- see that
// helper's comment for why the distinction matters for hair-strand-like geometry).
struct FragmentCounter {
    SceneView scene;
    const BBox* triangle_bboxes;
    Vec3 bounds_min;
    Vec3 inv_cell_size;
    Vec3 cell_size;
    int resolution_x, resolution_y, resolution_z;

    HAPPYRAY_HD uint32_t operator()(uint32_t tri) const {
        uint32_t count = 0;
        detail::ForEachIntersectedCell(scene, tri, triangle_bboxes[tri], bounds_min, inv_cell_size,
                                       cell_size, resolution_x, resolution_y, resolution_z,
                                       [&](int, int, int) { ++count; });
        return count;
    }
};

// Writes fragments -- exactly `FragmentCounter`'s count for this triangle, packed contiguously
// from `write_offsets[tri]` (no invalid/sentinel entries to filter out afterwards).
struct FragmentWriter {
    SceneView scene;
    const BBox* triangle_bboxes;
    const uint32_t* write_offsets;
    Vec3 bounds_min;
    Vec3 inv_cell_size;
    Vec3 cell_size;
    int resolution_x, resolution_y, resolution_z;
    uint32_t* out_cell_indices;
    uint32_t* out_triangle_indices;

    HAPPYRAY_HD void operator()(uint32_t tri) const {
        uint32_t offset = write_offsets[tri];
        detail::ForEachIntersectedCell(
            scene, tri, triangle_bboxes[tri], bounds_min, inv_cell_size, cell_size, resolution_x,
            resolution_y, resolution_z, [&](int cx, int cy, int cz) {
                out_cell_indices[offset] = FlattenCellIndex(cx, cy, cz, resolution_x, resolution_y);
                out_triangle_indices[offset] = tri;
                ++offset;
            });
    }
};

}  // namespace

struct UniformGrid::Impl {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int resolution_x = 1;
    int resolution_y = 1;
    int resolution_z = 1;
    thrust::device_vector<uint32_t> cell_range_start;
    thrust::device_vector<uint32_t> cell_range_end;
    thrust::device_vector<uint32_t> triangle_indices;

    UniformGridView View() const {
        UniformGridView view;
        view.bounds_min = bounds_min;
        view.bounds_max = bounds_max;
        view.cell_size = Vec3((bounds_max.x - bounds_min.x) / resolution_x,
                              (bounds_max.y - bounds_min.y) / resolution_y,
                              (bounds_max.z - bounds_min.z) / resolution_z);
        view.inv_cell_size =
            Vec3(1.0f / view.cell_size.x, 1.0f / view.cell_size.y, 1.0f / view.cell_size.z);
        view.resolution_x = resolution_x;
        view.resolution_y = resolution_y;
        view.resolution_z = resolution_z;
        view.cell_range_start = thrust::raw_pointer_cast(cell_range_start.data());
        view.cell_range_end = thrust::raw_pointer_cast(cell_range_end.data());
        view.triangle_indices = thrust::raw_pointer_cast(triangle_indices.data());
        return view;
    }
};

UniformGrid::UniformGrid(const SceneView& scene, float target_triangles_per_cell)
    : impl_(std::make_unique<Impl>()) {
    const uint32_t triangle_count = scene.triangle_count;

    // 1. Per-triangle BBoxes, reduced to the overall scene bounds (slightly expanded so
    //    boundary-coincident geometry reliably falls inside a valid cell despite
    //    floating-point rounding).
    thrust::device_vector<BBox> triangle_bboxes(triangle_count);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(triangle_count), triangle_bboxes.begin(),
                      ComputeTriangleBBoxFunctor{scene});

    BBox bounds = (triangle_count > 0)
                      ? thrust::reduce(triangle_bboxes.begin(), triangle_bboxes.end(), BBox(),
                                       BBoxUnionFunctor())
                      : BBox();
    const float diagonal = Length(bounds.Extent());
    const float eps = diagonal > 0.0f ? diagonal * 1e-4f : 1e-4f;
    bounds.min_bound -= Vec3(eps, eps, eps);
    bounds.max_bound += Vec3(eps, eps, eps);

    const GridResolution resolution =
        ComputeGridResolution(bounds, triangle_count, target_triangles_per_cell);
    impl_->bounds_min = bounds.min_bound;
    impl_->bounds_max = bounds.max_bound;
    impl_->resolution_x = resolution.x;
    impl_->resolution_y = resolution.y;
    impl_->resolution_z = resolution.z;

    const Vec3 cell_size((bounds.max_bound.x - bounds.min_bound.x) / resolution.x,
                        (bounds.max_bound.y - bounds.min_bound.y) / resolution.y,
                        (bounds.max_bound.z - bounds.min_bound.z) / resolution.z);
    const Vec3 inv_cell_size(1.0f / cell_size.x, 1.0f / cell_size.y, 1.0f / cell_size.z);

    // 2. Per-triangle overlapped-cell counts, exclusive-scanned into write offsets.
    thrust::device_vector<uint32_t> counts(triangle_count);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(triangle_count), counts.begin(),
                      FragmentCounter{scene, thrust::raw_pointer_cast(triangle_bboxes.data()),
                                      bounds.min_bound, inv_cell_size, cell_size, resolution.x,
                                      resolution.y, resolution.z});

    thrust::device_vector<uint32_t> write_offsets(triangle_count);
    thrust::exclusive_scan(counts.begin(), counts.end(), write_offsets.begin());
    const uint32_t total_pairs =
        (triangle_count > 0) ? (write_offsets.back() + counts.back()) : 0u;

    // 3. Write fragments (exactly `counts[tri]` of them per triangle, no sentinel filtering
    //    needed).
    thrust::device_vector<uint32_t> pair_cell_indices(total_pairs);
    thrust::device_vector<uint32_t> pair_triangle_indices(total_pairs);
    FragmentWriter write_fragments{scene,
                                  thrust::raw_pointer_cast(triangle_bboxes.data()),
                                  thrust::raw_pointer_cast(write_offsets.data()),
                                  bounds.min_bound,
                                  inv_cell_size,
                                  cell_size,
                                  resolution.x,
                                  resolution.y,
                                  resolution.z,
                                  thrust::raw_pointer_cast(pair_cell_indices.data()),
                                  thrust::raw_pointer_cast(pair_triangle_indices.data())};
    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(triangle_count), write_fragments);

    // 4. Sort by cell index -- the defining "sort-based" step.
    thrust::sort_by_key(pair_cell_indices.begin(), pair_cell_indices.end(),
                       pair_triangle_indices.begin());

    // 5. Extract start/end ranges directly from adjacent sorted cell keys.
    const uint32_t num_cells =
        static_cast<uint32_t>(resolution.x) * resolution.y * resolution.z;
    impl_->cell_range_start.assign(num_cells, 0u);
    impl_->cell_range_end.assign(num_cells, 0u);
    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(total_pairs),
                     detail::CellRangeExtractor{
                         thrust::raw_pointer_cast(pair_cell_indices.data()),
                         thrust::raw_pointer_cast(impl_->cell_range_start.data()),
                         thrust::raw_pointer_cast(impl_->cell_range_end.data()),
                         total_pairs});

    impl_->triangle_indices = std::move(pair_triangle_indices);
}

UniformGrid::~UniformGrid() = default;

uint32_t UniformGrid::TriangleIndexCount() const {
    return static_cast<uint32_t>(impl_->triangle_indices.size());
}

float UniformGrid::ComputeSahCost(const BBox& scene_bounds, const SahCostConstants& costs) const {
    const UniformGridView view = View();
    const auto num_cells =
        static_cast<uint32_t>(view.resolution_x) * view.resolution_y * view.resolution_z;
    const float cell_area = HalfSurfaceArea(view.cell_size);
    const float total =
        cell_area * (static_cast<float>(num_cells) * costs.traversal +
                    static_cast<float>(TriangleIndexCount()) * costs.intersection);
    return total / HalfSurfaceArea(scene_bounds.Extent());
}

UniformGridView UniformGrid::View() const { return impl_->View(); }

}  // namespace happyray
