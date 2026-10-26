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
    uint32_t* starts;
    uint32_t* ends;
    uint32_t pair_count;
    uint32_t cell_count;

    HAPPYRAY_HD void operator()(uint32_t i) const {
        const uint32_t cell = keys[i];
        if (cell >= cell_count) return;
        if (i == 0 || keys[i - 1] != cell) starts[cell] = i;
        if (i + 1 == pair_count || keys[i + 1] != cell) ends[cell] = i + 1;
    }
};

// Computes each top-level cell's cubic second-level resolution.
struct ComputeSecondLevelInfoFunctor {
    Vec3 top_bounds_min;
    Vec3 top_cell_size;
    int top_resolution_x;
    int top_resolution_y;
    const uint32_t* top_range_start;
    const uint32_t* top_range_end;
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
        const uint32_t triangle_count = top_range_end[c] - top_range_start[c];

        const GridResolution computed_res =
            (triangle_count <= dense_cell_threshold)
                ? GridResolution{1, 1, 1}
                : ComputeGridResolution(BBox(cell_min, cell_max), triangle_count,
                                       second_level_density);

        SecondLevelGridInfo info;
        const int resolution = computed_res.x > computed_res.y
                                   ? (computed_res.x > computed_res.z ? computed_res.x
                                                                      : computed_res.z)
                                   : (computed_res.y > computed_res.z ? computed_res.y
                                                                      : computed_res.z);
        info.resolution = {resolution, resolution, resolution};
        return info;
    }
};

struct ExtractNumCellsFunctor {
    HAPPYRAY_HD uint32_t operator()(const SecondLevelGridInfo& info) const {
        return static_cast<uint32_t>(info.resolution.x) * info.resolution.y *
               info.resolution.z;
    }
};

struct SetCellRangeOffsetFunctor {
    HAPPYRAY_HD SecondLevelGridInfo operator()(SecondLevelGridInfo info,
                                               uint32_t offset) const {
        info.cell_range_offset = offset;
        return info;
    }
};

struct FragmentCounter {
    SceneView scene;
    const uint32_t* top_cell_keys;
    const uint32_t* top_triangle_indices;
    const SecondLevelGridInfo* infos;
    Vec3 top_bounds_min, top_cell_size, top_inv_cell_size;
    int top_resolution_x, top_resolution_y;
    uint32_t top_cell_count;

    HAPPYRAY_HD uint32_t operator()(uint32_t i) const {
        const uint32_t owner = top_cell_keys[i];
        if (owner >= top_cell_count) return 0;
        const SecondLevelGridInfo& info = infos[owner];
        const int cx = static_cast<int>(owner % top_resolution_x);
        const int cy = static_cast<int>((owner / top_resolution_x) % top_resolution_y);
        const int cz = static_cast<int>(owner / (top_resolution_x * top_resolution_y));
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
        const Vec3 sub_inv(top_inv_cell_size.x * info.resolution.x,
                           top_inv_cell_size.y * info.resolution.y,
                           top_inv_cell_size.z * info.resolution.z);
        const Vec3 sub_cell_size(top_cell_size.x / info.resolution.x,
                                 top_cell_size.y / info.resolution.y,
                                 top_cell_size.z / info.resolution.z);
        const uint32_t tri = top_triangle_indices[i];
        const Triangle triangle = FetchTriangle(scene, tri);
        const BBox box = BBoxExtractor<Triangle>::Get(triangle);
        return detail::CountTriangleCellIntersections(
            triangle, box, cell_min, sub_inv, sub_cell_size, info.resolution.x,
            info.resolution.y, info.resolution.z);
    }
};

struct FragmentWriter {
    SceneView scene;
    const BBox* triangle_bboxes;
    const uint32_t* top_cell_keys;
    const uint32_t* top_triangle_indices;
    const SecondLevelGridInfo* infos;
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
        const SecondLevelGridInfo& info = infos[owner];
        const int cx = static_cast<int>(owner % top_resolution_x);
        const int cy = static_cast<int>((owner / top_resolution_x) % top_resolution_y);
        const int cz = static_cast<int>(owner / (top_resolution_x * top_resolution_y));
        const Vec3 cell_min(top_bounds_min.x + cx * top_cell_size.x,
                            top_bounds_min.y + cy * top_cell_size.y,
                            top_bounds_min.z + cz * top_cell_size.z);
        const Vec3 sub_inv(top_inv_cell_size.x * info.resolution.x,
                           top_inv_cell_size.y * info.resolution.y,
                           top_inv_cell_size.z * info.resolution.z);
        const uint32_t tri = top_triangle_indices[i];
        const uint32_t write_begin = i == 0 ? 0u : inclusive_offsets[i - 1];
        const uint32_t write_end = inclusive_offsets[i];
        const Vec3 sub_cell_size(top_cell_size.x / info.resolution.x,
                                 top_cell_size.y / info.resolution.y,
                                 top_cell_size.z / info.resolution.z);
        uint32_t offset = write_begin;
        const Triangle triangle = FetchTriangle(scene, tri);
        detail::ForEachIntersectedCell(
            triangle, triangle_bboxes[tri], cell_min, sub_inv, sub_cell_size,
            info.resolution.x, info.resolution.y, info.resolution.z,
            [&](int x, int y, int z) {
                out_cell_indices[offset] = info.cell_range_offset +
                    FlattenCellIndex(x, y, z, info.resolution.x,
                                     info.resolution.y);
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

struct SetTriangleRefCountFunctor {
    const uint32_t* top_range_start;
    const uint32_t* top_range_end;
    const uint32_t* pair_counts;

    HAPPYRAY_HD SecondLevelGridInfo operator()(uint32_t cell,
                                               SecondLevelGridInfo info) const {
        uint32_t count = 0;
        for (uint32_t i = top_range_start[cell]; i < top_range_end[cell]; ++i)
            count += pair_counts[i];
        info.triangle_ref_count = count;
        return info;
    }
};

struct TwoLevelGridSahFunctor {
    TwoLevelGridView grid;
    SahCostConstants costs;

    HAPPYRAY_HD float operator()(uint32_t t) const {
        const SecondLevelGridInfo& info = grid.second_level_infos[t];
        const auto num_local = static_cast<uint32_t>(info.resolution.x) *
                               info.resolution.y * info.resolution.z;
        const Vec3 cell_size(grid.top_level.cell_size.x / info.resolution.x,
                             grid.top_level.cell_size.y / info.resolution.y,
                             grid.top_level.cell_size.z / info.resolution.z);
        return HalfSurfaceArea(cell_size) *
              (static_cast<float>(num_local) * costs.traversal +
               static_cast<float>(info.triangle_ref_count) * costs.intersection);
    }
};

}  // namespace

struct TwoLevelGrid::Impl {
    GridRangeView top_level;
    thrust::device_vector<uint32_t> top_cell_range_start;
    thrust::device_vector<uint32_t> top_cell_range_end;
    thrust::device_vector<uint32_t> top_triangle_indices;
    thrust::device_vector<SecondLevelGridInfo> second_level_infos;
    thrust::device_vector<uint32_t> second_level_cell_range_start;
    thrust::device_vector<uint32_t> second_level_cell_range_end;
    thrust::device_vector<uint32_t> second_level_triangle_indices;

    TwoLevelGridView View() const {
        TwoLevelGridView view;
        view.top_level = top_level;
        view.top_level.cell_range_start = thrust::raw_pointer_cast(top_cell_range_start.data());
        view.top_level.cell_range_end = thrust::raw_pointer_cast(top_cell_range_end.data());
        view.top_level.triangle_indices = thrust::raw_pointer_cast(top_triangle_indices.data());
        view.second_level_infos = thrust::raw_pointer_cast(second_level_infos.data());
        view.second_level_cell_range_start =
            thrust::raw_pointer_cast(second_level_cell_range_start.data());
        view.second_level_cell_range_end =
            thrust::raw_pointer_cast(second_level_cell_range_end.data());
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

    impl_->top_cell_range_start.assign(num_top_cells, 0u);
    impl_->top_cell_range_end.assign(num_top_cells, 0u);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_pairs),
        CellRangeExtractor{thrust::raw_pointer_cast(top_cell_keys.data()),
                           thrust::raw_pointer_cast(impl_->top_cell_range_start.data()),
                           thrust::raw_pointer_cast(impl_->top_cell_range_end.data()),
                           total_top_pairs,
                           num_top_cells});

    const GridRangeView top_view = impl_->View().top_level;
    thrust::device_vector<SecondLevelGridInfo> infos(num_top_cells);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells), infos.begin(),
        ComputeSecondLevelInfoFunctor{
            top_view.bounds_min, top_view.cell_size, top_view.resolution_x,
            top_view.resolution_y,
            thrust::raw_pointer_cast(impl_->top_cell_range_start.data()),
            thrust::raw_pointer_cast(impl_->top_cell_range_end.data()),
            second_level_density, dense_cell_threshold});

    thrust::device_vector<uint32_t> num_second_cells(num_top_cells);
    thrust::transform(infos.begin(), infos.end(), num_second_cells.begin(),
                      ExtractNumCellsFunctor());
    thrust::device_vector<uint32_t> second_cell_offsets(num_top_cells);
    thrust::exclusive_scan(num_second_cells.begin(), num_second_cells.end(),
                           second_cell_offsets.begin());
    const uint32_t total_second_cells =
        second_cell_offsets.back() + num_second_cells.back();
    thrust::transform(infos.begin(), infos.end(), second_cell_offsets.begin(),
                      infos.begin(), SetCellRangeOffsetFunctor());

    thrust::device_vector<uint32_t> second_pair_counts(total_top_pairs);
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_top_pairs), second_pair_counts.begin(),
        FragmentCounter{scene,
                        thrust::raw_pointer_cast(top_cell_keys.data()),
                        thrust::raw_pointer_cast(impl_->top_triangle_indices.data()),
                        thrust::raw_pointer_cast(infos.data()), top_view.bounds_min,
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
                       thrust::raw_pointer_cast(infos.data()),
                       thrust::raw_pointer_cast(second_pair_counts.data()),
                       thrust::raw_pointer_cast(exact_second_pair_counts.data()),
                       total_second_cells,
                       thrust::raw_pointer_cast(second_cell_keys.data()),
                       thrust::raw_pointer_cast(second_triangle_indices.data()),
                       top_view.bounds_min, top_view.cell_size,
                       top_view.inv_cell_size, top_view.resolution_x,
                       top_view.resolution_y, num_top_cells});
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_top_cells), infos.begin(),
        infos.begin(),
        SetTriangleRefCountFunctor{
            thrust::raw_pointer_cast(impl_->top_cell_range_start.data()),
            thrust::raw_pointer_cast(impl_->top_cell_range_end.data()),
            thrust::raw_pointer_cast(exact_second_pair_counts.data())});
    thrust::sort_by_key(second_cell_keys.begin(), second_cell_keys.end(),
                        second_triangle_indices.begin());
    impl_->second_level_cell_range_start.assign(total_second_cells, 0u);
    impl_->second_level_cell_range_end.assign(total_second_cells, 0u);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(total_second_pairs),
        CellRangeExtractor{
            thrust::raw_pointer_cast(second_cell_keys.data()),
            thrust::raw_pointer_cast(impl_->second_level_cell_range_start.data()),
            thrust::raw_pointer_cast(impl_->second_level_cell_range_end.data()),
            total_second_pairs,
            total_second_cells});

    impl_->second_level_infos = std::move(infos);
    impl_->second_level_triangle_indices = std::move(second_triangle_indices);
}

TwoLevelGrid::~TwoLevelGrid() = default;

uint32_t TwoLevelGrid::TopLevelTriangleIndexCount() const {
    return static_cast<uint32_t>(impl_->top_triangle_indices.size());
}

uint32_t TwoLevelGrid::SecondLevelCellCount() const {
    return static_cast<uint32_t>(impl_->second_level_cell_range_start.size());
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
            TwoLevelGridSahFunctor{view, costs}, 0.0f, thrust::plus<float>());
    }
    return total / HalfSurfaceArea(scene_bounds.Extent());
}

TwoLevelGridView TwoLevelGrid::View() const { return impl_->View(); }

}  // namespace happyray
