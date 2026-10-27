// Thrust implementation of TwoLevelGrid (see two_level_grid.h), following the legacy
// TLGridSortBuilder: build the top level directly, then build each nested grid from its owning
// cell's sorted primitive range. Both levels store explicit begin/end ranges rather than CSR
// offsets.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "two_level_grid.h"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

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

struct TopPairCounter {
    SceneView scene;
    Vec3 bounds_min, inv_cell_size, cell_size;
    int res_x, res_y, res_z;
    HAPPYRAY_HD uint32_t operator()(uint32_t tri) const {
        const Triangle triangle = FetchTriangle(scene, tri);
        const BBox box = BBoxExtractor<Triangle>::Get(triangle);
        return detail::CountTriangleCellIntersections(
            triangle, box, bounds_min, inv_cell_size, cell_size, res_x, res_y,
            res_z);
    }
};

struct TopPairWriter {
    SceneView scene;
    const BBox* triangle_bboxes;
    const uint32_t* inclusive_offsets;
    Vec3 bounds_min, inv_cell_size, cell_size;
    int res_x, res_y, res_z;
    uint32_t invalid_cell_index;
    uint32_t* cell_keys;
    uint32_t* triangle_indices;

    HAPPYRAY_HD void operator()(uint32_t tri) const {
        uint32_t offset = tri == 0 ? 0u : inclusive_offsets[tri - 1];
        const Triangle triangle = FetchTriangle(scene, tri);
        detail::ForEachIntersectedCell(
            triangle, triangle_bboxes[tri], bounds_min, inv_cell_size,
            cell_size, res_x, res_y, res_z, [&](int x, int y, int z) {
                cell_keys[offset] = FlattenCellIndex(x, y, z, res_x, res_y);
                triangle_indices[offset] = tri;
                ++offset;
            });
        const uint32_t write_end = inclusive_offsets[tri];
        while (offset < write_end) {
            cell_keys[offset] = invalid_cell_index;
            triangle_indices[offset] = tri;
            ++offset;
        }
    }
};

struct CellRangeExtractor {
    const uint32_t* keys;
    CellRange* ranges;
    uint32_t pair_count;
    uint32_t cell_count;

    HAPPYRAY_HD void operator()(uint32_t i) const {
        const uint32_t cell = keys[i];
        if (cell >= cell_count) return;
        if (i == 0 || keys[i - 1] != cell) ranges[cell].start = i;
        if (i + 1 == pair_count || keys[i + 1] != cell) ranges[cell].end = i + 1;
    }
};

// Computes each top-level cell's cubic second-level resolution.
struct ComputeTopLevelCellFunctor {
    Vec3 top_bounds_min;
    Vec3 top_cell_size;
    int top_resolution_x;
    int top_resolution_y;
    const CellRange* top_ranges;
    float second_level_density;
    uint32_t dense_cell_threshold;

    HAPPYRAY_HD TopLevelCell operator()(uint32_t c) const {
        const int cz = static_cast<int>(c) / (top_resolution_x * top_resolution_y);
        const int rem = static_cast<int>(c) % (top_resolution_x * top_resolution_y);
        const int cy = rem / top_resolution_x;
        const int cx = rem % top_resolution_x;
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
        const Vec3 cell_max = cell_min + top_cell_size;
        const uint32_t triangle_count = top_ranges[c].end - top_ranges[c].start;

        const GridResolution computed_res =
            (triangle_count <= dense_cell_threshold)
                ? GridResolution{1, 1, 1}
                : ComputeGridResolution(BBox(cell_min, cell_max), triangle_count,
                                        second_level_density);

        TopLevelCell info;
        const int resolution = computed_res.x > computed_res.y
                                   ? (computed_res.x > computed_res.z ? computed_res.x
                                                                      : computed_res.z)
                                   : (computed_res.y > computed_res.z ? computed_res.y
                                                                      : computed_res.z);
        info.resolution = resolution;
        return info;
    }
};

struct ExtractNumCellsFunctor {
    HAPPYRAY_HD uint32_t operator()(const TopLevelCell& info) const {
        return static_cast<uint32_t>(info.resolution) * info.resolution *
               info.resolution;
    }
};

struct SetCellRangeOffsetFunctor {
    HAPPYRAY_HD TopLevelCell operator()(TopLevelCell info,
                                        uint32_t offset) const {
        info.cell_range_offset = offset;
        return info;
    }
};

struct FragmentCounter {
    SceneView scene;
    const uint32_t* top_cell_keys;
    const uint32_t* top_triangle_indices;
    const TopLevelCell* top_level_cells;
    Vec3 top_bounds_min, top_cell_size, top_inv_cell_size;
    int top_resolution_x, top_resolution_y;
    uint32_t top_cell_count;

    HAPPYRAY_HD uint32_t operator()(uint32_t i) const {
        const uint32_t owner = top_cell_keys[i];
        if (owner >= top_cell_count) return 0;
        const TopLevelCell& info = top_level_cells[owner];
        const int cx = static_cast<int>(owner % top_resolution_x);
        const int cy = static_cast<int>((owner / top_resolution_x) % top_resolution_y);
        const int cz = static_cast<int>(owner / (top_resolution_x * top_resolution_y));
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
          const Vec3 sub_inv(top_inv_cell_size.x * info.resolution,
                       top_inv_cell_size.y * info.resolution,
                       top_inv_cell_size.z * info.resolution);
        const Vec3 sub_cell_size(top_cell_size.x / info.resolution,
                           top_cell_size.y / info.resolution,
                           top_cell_size.z / info.resolution);
        const uint32_t tri = top_triangle_indices[i];
        const Triangle triangle = FetchTriangle(scene, tri);
        const BBox box = BBoxExtractor<Triangle>::Get(triangle);
        return detail::CountTriangleCellIntersections(
            triangle, box, cell_min, sub_inv, sub_cell_size, info.resolution,
            info.resolution, info.resolution);
    }
};

struct FragmentWriter {
    SceneView scene;
    const BBox* triangle_bboxes;
    const uint32_t* top_cell_keys;
    const uint32_t* top_triangle_indices;
    const TopLevelCell* top_level_cells;
    const uint32_t* inclusive_offsets;
    uint32_t* exact_counts;
    uint32_t invalid_cell_index;
    uint32_t* out_cell_indices;
    uint32_t* out_triangle_indices;
    Vec3 top_bounds_min, top_cell_size, top_inv_cell_size;
    int top_resolution_x, top_resolution_y;
    uint32_t top_cell_count;

    HAPPYRAY_HD void operator()(uint32_t i) const {
        const uint32_t owner = top_cell_keys[i];
        if (owner >= top_cell_count) {
            exact_counts[i] = 0;
            return;
        }
        const TopLevelCell& info = top_level_cells[owner];
        const int cx = static_cast<int>(owner % top_resolution_x);
        const int cy = static_cast<int>((owner / top_resolution_x) % top_resolution_y);
        const int cz = static_cast<int>(owner / (top_resolution_x * top_resolution_y));
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
        const Vec3 sub_inv(top_inv_cell_size.x * info.resolution,
                   top_inv_cell_size.y * info.resolution,
                   top_inv_cell_size.z * info.resolution);
        const uint32_t tri = top_triangle_indices[i];
        const uint32_t write_begin = i == 0 ? 0u : inclusive_offsets[i - 1];
        const uint32_t write_end = inclusive_offsets[i];
        const Vec3 sub_cell_size(top_cell_size.x / info.resolution,
                                 top_cell_size.y / info.resolution,
                                 top_cell_size.z / info.resolution);
        uint32_t offset = write_begin;
        const Triangle triangle = FetchTriangle(scene, tri);
        detail::ForEachIntersectedCell(
            triangle, triangle_bboxes[tri], cell_min, sub_inv, sub_cell_size,
            info.resolution, info.resolution, info.resolution,
            [&](int x, int y, int z) {
                out_cell_indices[offset] =
                    info.cell_range_offset +
                    FlattenCellIndex(x, y, z, info.resolution, info.resolution);
                out_triangle_indices[offset] = tri;
                ++offset;
            });
        exact_counts[i] = offset - write_begin;
        while (offset < write_end) {
            out_cell_indices[offset] = invalid_cell_index;
            out_triangle_indices[offset] = tri;
            ++offset;
        }
    }
};

struct ComputeTopLevelCellRefCountFunctor {
    const CellRange* top_ranges;
    const uint32_t* pair_counts;

    HAPPYRAY_HD uint32_t operator()(uint32_t cell) const {
        uint32_t count = 0;
        for (uint32_t i = top_ranges[cell].start; i < top_ranges[cell].end; ++i)
            count += pair_counts[i];
        return count;
    }
};

struct TwoLevelGridSahFunctor {
    const TopLevelCell* top_level_cells;
    const uint32_t* top_level_triangle_ref_counts;
    Vec3 top_cell_size;
    SahCostConstants costs;

    HAPPYRAY_HD float operator()(uint32_t t) const {
        const TopLevelCell& cell = top_level_cells[t];
        const auto num_local = static_cast<uint32_t>(cell.resolution) *
                               cell.resolution * cell.resolution;
        const Vec3 cell_size(top_cell_size.x / cell.resolution,
                             top_cell_size.y / cell.resolution,
                             top_cell_size.z / cell.resolution);
        return HalfSurfaceArea(cell_size) *
               (static_cast<float>(num_local) * costs.traversal +
                static_cast<float>(top_level_triangle_ref_counts[t]) *
                    costs.intersection);
    }
};

}  // namespace

struct TwoLevelGrid::Impl {
    GridRangeView top_level;
    thrust::device_vector<CellRange> top_cell_ranges;
    thrust::device_vector<uint32_t> top_triangle_indices;
    thrust::device_vector<TopLevelCell> top_level_cells;
    thrust::device_vector<uint32_t> top_level_triangle_ref_counts;
    thrust::device_vector<CellRange> second_level_cell_ranges;
    thrust::device_vector<uint32_t> second_level_triangle_indices;

    TwoLevelGridView View() const {
        TwoLevelGridView view;
        view.top_level = top_level;
        view.top_level.cell_ranges = thrust::raw_pointer_cast(top_cell_ranges.data());
        view.top_level.triangle_indices = thrust::raw_pointer_cast(top_triangle_indices.data());
        view.top_level_cells = thrust::raw_pointer_cast(top_level_cells.data());
        view.second_level_cell_ranges =
            thrust::raw_pointer_cast(second_level_cell_ranges.data());
        view.second_level_triangle_indices =
            thrust::raw_pointer_cast(second_level_triangle_indices.data());
        return view;
    }
};

TwoLevelGrid::TwoLevelGrid(const SceneView& scene, float top_level_density,
                          float second_level_density, uint32_t dense_cell_threshold)
    : impl_(std::make_unique<Impl>()) {
    const uint32_t triangle_count = scene.triangle_count;
    thrust::device_vector<BBox> triangle_bboxes(triangle_count);
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(triangle_count),
                      triangle_bboxes.begin(), ComputeTriangleBBoxFunctor{scene});

    BBox bounds = triangle_count > 0
                      ? thrust::reduce(triangle_bboxes.begin(), triangle_bboxes.end(),
                                       BBox(), BBoxUnionFunctor())
                      : BBox();
    const float diagonal = Length(bounds.Extent());
    const float epsilon = diagonal > 0.0f ? diagonal * 1e-4f : 1e-4f;
    bounds.min_bound -= Vec3(epsilon, epsilon, epsilon);
    bounds.max_bound += Vec3(epsilon, epsilon, epsilon);

    const GridResolution top_resolution =
        ComputeGridResolution(bounds, triangle_count, top_level_density);
    impl_->top_level.bounds_min = bounds.min_bound;
    impl_->top_level.bounds_max = bounds.max_bound;
    impl_->top_level.resolution_x = top_resolution.x;
    impl_->top_level.resolution_y = top_resolution.y;
    impl_->top_level.resolution_z = top_resolution.z;
    impl_->top_level.cell_size = Vec3(bounds.Extent().x / top_resolution.x,
                                      bounds.Extent().y / top_resolution.y,
                                      bounds.Extent().z / top_resolution.z);
    impl_->top_level.inv_cell_size =
        Vec3(1.0f / impl_->top_level.cell_size.x,
             1.0f / impl_->top_level.cell_size.y,
             1.0f / impl_->top_level.cell_size.z);

    const uint32_t num_top_cells = static_cast<uint32_t>(top_resolution.x) *
                                   top_resolution.y * top_resolution.z;
    thrust::device_vector<uint32_t> top_pair_counts(triangle_count);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(triangle_count), top_pair_counts.begin(),
        TopPairCounter{scene,
                   bounds.min_bound, impl_->top_level.inv_cell_size,
                       impl_->top_level.cell_size,
                       top_resolution.x, top_resolution.y, top_resolution.z});
    thrust::inclusive_scan(top_pair_counts.begin(), top_pair_counts.end(),
                           top_pair_counts.begin());
    const uint32_t total_top_pairs =
        triangle_count > 0 ? top_pair_counts.back() : 0u;
    thrust::device_vector<uint32_t> top_cell_keys(total_top_pairs);
    impl_->top_triangle_indices.resize(total_top_pairs);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(triangle_count),
        TopPairWriter{scene, thrust::raw_pointer_cast(triangle_bboxes.data()),
                      thrust::raw_pointer_cast(top_pair_counts.data()),
                      bounds.min_bound, impl_->top_level.inv_cell_size,
                  impl_->top_level.cell_size,
                      top_resolution.x, top_resolution.y, top_resolution.z,
                      num_top_cells,
                      thrust::raw_pointer_cast(top_cell_keys.data()),
                      thrust::raw_pointer_cast(impl_->top_triangle_indices.data())});
    thrust::sort_by_key(top_cell_keys.begin(), top_cell_keys.end(),
                        impl_->top_triangle_indices.begin());

    impl_->top_cell_ranges.assign(num_top_cells, CellRange{0u, 0u});
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_pairs),
        CellRangeExtractor{thrust::raw_pointer_cast(top_cell_keys.data()),
                           thrust::raw_pointer_cast(impl_->top_cell_ranges.data()),
                           total_top_pairs,
                           num_top_cells});

    const GridRangeView top_view = impl_->View().top_level;
    thrust::device_vector<TopLevelCell> top_level_cells(num_top_cells);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells), top_level_cells.begin(),
        ComputeTopLevelCellFunctor{
            top_view.bounds_min, top_view.cell_size, top_view.resolution_x,
            top_view.resolution_y,
            thrust::raw_pointer_cast(impl_->top_cell_ranges.data()),
            second_level_density, dense_cell_threshold});

    thrust::device_vector<uint32_t> num_second_cells(num_top_cells);
    thrust::transform(top_level_cells.begin(), top_level_cells.end(),
                      num_second_cells.begin(), ExtractNumCellsFunctor());
    thrust::device_vector<uint32_t> second_cell_offsets(num_top_cells);
    thrust::exclusive_scan(num_second_cells.begin(), num_second_cells.end(),
                           second_cell_offsets.begin());
    const uint32_t total_second_cells =
        second_cell_offsets.back() + num_second_cells.back();
    thrust::transform(top_level_cells.begin(), top_level_cells.end(),
                      second_cell_offsets.begin(), top_level_cells.begin(),
                      SetCellRangeOffsetFunctor());

    thrust::device_vector<uint32_t> second_pair_counts(total_top_pairs);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_pairs), second_pair_counts.begin(),
        FragmentCounter{scene,
                        thrust::raw_pointer_cast(top_cell_keys.data()),
                        thrust::raw_pointer_cast(impl_->top_triangle_indices.data()),
                        thrust::raw_pointer_cast(top_level_cells.data()), top_view.bounds_min,
                        top_view.cell_size, top_view.inv_cell_size,
                        top_view.resolution_x, top_view.resolution_y,
                        num_top_cells});
    thrust::inclusive_scan(second_pair_counts.begin(), second_pair_counts.end(),
                           second_pair_counts.begin());
    const uint32_t total_second_pairs =
        total_top_pairs > 0 ? second_pair_counts.back() : 0u;
    thrust::device_vector<uint32_t> second_cell_keys(total_second_pairs);
    thrust::device_vector<uint32_t> second_triangle_indices(total_second_pairs);
    thrust::device_vector<uint32_t> exact_second_pair_counts(total_top_pairs, 0u);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_pairs),
        FragmentWriter{scene, thrust::raw_pointer_cast(triangle_bboxes.data()),
                       thrust::raw_pointer_cast(top_cell_keys.data()),
                       thrust::raw_pointer_cast(impl_->top_triangle_indices.data()),
                       thrust::raw_pointer_cast(top_level_cells.data()),
                       thrust::raw_pointer_cast(second_pair_counts.data()),
                       thrust::raw_pointer_cast(exact_second_pair_counts.data()),
                       total_second_cells,
                       thrust::raw_pointer_cast(second_cell_keys.data()),
                       thrust::raw_pointer_cast(second_triangle_indices.data()),
                       top_view.bounds_min, top_view.cell_size,
                       top_view.inv_cell_size, top_view.resolution_x,
                       top_view.resolution_y, num_top_cells});
    thrust::device_vector<uint32_t> top_level_triangle_ref_counts(num_top_cells);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells),
        top_level_triangle_ref_counts.begin(),
        ComputeTopLevelCellRefCountFunctor{
            thrust::raw_pointer_cast(impl_->top_cell_ranges.data()),
            thrust::raw_pointer_cast(exact_second_pair_counts.data())});
    thrust::sort_by_key(second_cell_keys.begin(), second_cell_keys.end(),
                        second_triangle_indices.begin());
    impl_->second_level_cell_ranges.assign(total_second_cells, CellRange{0u, 0u});
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_second_pairs),
        CellRangeExtractor{
            thrust::raw_pointer_cast(second_cell_keys.data()),
            thrust::raw_pointer_cast(impl_->second_level_cell_ranges.data()),
            total_second_pairs,
            total_second_cells});

    impl_->top_level_cells = std::move(top_level_cells);
    impl_->top_level_triangle_ref_counts =
        std::move(top_level_triangle_ref_counts);
    impl_->second_level_triangle_indices = std::move(second_triangle_indices);
}

TwoLevelGrid::~TwoLevelGrid() = default;

uint32_t TwoLevelGrid::TopLevelTriangleIndexCount() const {
    return static_cast<uint32_t>(impl_->top_triangle_indices.size());
}

uint32_t TwoLevelGrid::SecondLevelCellCount() const {
    return static_cast<uint32_t>(impl_->second_level_cell_ranges.size());
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
            thrust::counting_iterator<uint32_t>(0),
            thrust::counting_iterator<uint32_t>(num_top_cells),
            TwoLevelGridSahFunctor{
                thrust::raw_pointer_cast(impl_->top_level_cells.data()),
                thrust::raw_pointer_cast(
                    impl_->top_level_triangle_ref_counts.data()),
                view.top_level.cell_size, costs},
            0.0f, thrust::plus<float>());
    }
    return total / HalfSurfaceArea(scene_bounds.Extent());
}

TwoLevelGridView TwoLevelGrid::View() const { return impl_->View(); }

}  // namespace happyray
