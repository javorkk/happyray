#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++g_failures;
    }
}

struct OverlapCase {
    happyray::Vec3 a, b, c;
    bool expected;
};

struct CheckOverlapFunctor {
    happyray::SceneView scene;
    uint32_t* results;

    HAPPYRAY_HD void operator()(uint32_t i) const {
        results[i] = happyray::detail::TriangleCellIntersection(scene, i)(
            happyray::Vec3(0, 0, 0), happyray::Vec3(2, 2, 2));
    }
};

void CheckTriangleCellIntersection() {
    using happyray::Vec3;
    const std::vector<OverlapCase> cases = {
        {Vec3(0, 0, 0), Vec3(0.5f, 0, 0), Vec3(0, 0.5f, 0), true},
        {Vec3(2, 0, 0), Vec3(2, 1, 0), Vec3(2, 0, 1), false},  // Box axis.
        {Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4), false},  // Plane.
        {Vec3(2.5f, 0, 0), Vec3(0, 2.5f, 0), Vec3(2.5f, 0, 1), false},  // Edge axis.
        {Vec3(-4, 0, 0), Vec3(4, 0, 0), Vec3(0, 4, 0), true},  // All vertices outside.
        {Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(1, 0, 1), true},  // Face contact.
        {Vec3(1, 1, 1), Vec3(2, 1, 1), Vec3(1, 2, 1), true},  // Corner contact.
        {Vec3(1.005f, 0, 0), Vec3(1.005f, 1, 0), Vec3(1.005f, 0, 1), true},
        {Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), true},  // Point.
        {Vec3(2, 2, 2), Vec3(2, 2, 2), Vec3(2, 2, 2), false},
        {Vec3(-2, 0, 0), Vec3(0, 0, 0), Vec3(2, 0, 0), true},  // Line.
        {Vec3(2, 0, 0), Vec3(3, 0, 0), Vec3(4, 0, 0), false},
    };
    std::vector<Vec3> positions;
    std::vector<uint32_t> indices;
    for (const auto& test : cases) {
        for (const Vec3& vertex : {test.a, test.b, test.c}) {
            indices.push_back(static_cast<uint32_t>(positions.size()));
            positions.push_back(vertex);
        }
    }
    happyray::SceneView host_scene;
    host_scene.positions = positions.data();
    host_scene.indices = indices.data();
    host_scene.triangle_count = static_cast<uint32_t>(cases.size());
    const thrust::device_vector<Vec3> device_positions(positions.begin(), positions.end());
    const thrust::device_vector<uint32_t> device_indices(indices.begin(), indices.end());
    happyray::SceneView scene = host_scene;
    scene.positions = thrust::raw_pointer_cast(device_positions.data());
    scene.indices = thrust::raw_pointer_cast(device_indices.data());
    thrust::device_vector<uint32_t> results(cases.size());
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(scene.triangle_count),
        CheckOverlapFunctor{scene, thrust::raw_pointer_cast(results.data())});
    const thrust::host_vector<uint32_t> actual = results;
    for (uint32_t i = 0; i < cases.size(); ++i) {
        Check(happyray::detail::TriangleCellIntersection(host_scene, i)(Vec3(0, 0, 0), Vec3(2, 2, 2)) ==
                  cases[i].expected, "host triangle/cell overlap");
        Check(actual[i] == cases[i].expected, "backend triangle/cell overlap");
    }
}

template <typename T>
thrust::host_vector<T> Download(const T* data, uint32_t count) {
    const auto begin = thrust::device_pointer_cast(data);
    return thrust::host_vector<T>(begin, begin + count);
}

void CheckFilteredFragments(const happyray::SceneView& scene, const happyray::UniformGridView& grid) {
    const uint32_t num_cells = grid.resolution_x * grid.resolution_y * grid.resolution_z;
    const auto ranges = Download(grid.cell_ranges, num_cells);
    uint32_t reference_count = 0;
    for (const auto& range : ranges) {
        reference_count = std::max(reference_count, range.end);
    }
    const auto refs = Download(grid.triangle_indices, reference_count);
    const happyray::BBox box = happyray::ComputeTriangleBBox(scene, 0);
    int x0, x1, y0, y1, z0, z1;
    happyray::ComputeCellRange(box, grid.bounds_min, grid.inv_cell_size, grid.resolution_x,
                          grid.resolution_y, grid.resolution_z, &x0, &x1, &y0, &y1, &z0, &z1);
    const happyray::detail::TriangleCellIntersection intersects(scene, 0);
    uint32_t expected_refs = 0;
    for (int z = 0; z < grid.resolution_z; ++z) {
        for (int y = 0; y < grid.resolution_y; ++y) {
            for (int x = 0; x < grid.resolution_x; ++x) {
                const uint32_t cell = happyray::FlattenCellIndex(x, y, z, grid.resolution_x,
                                                            grid.resolution_y);
                const bool expected = x >= x0 && x <= x1 && y >= y0 && y <= y1 &&
                    z >= z0 && z <= z1 &&
                    intersects(grid.bounds_min + happyray::Vec3(x + 0.5f, y + 0.5f, z + 0.5f) *
                                   grid.cell_size, grid.cell_size);
                Check(ranges[cell].end - ranges[cell].start ==
                          static_cast<uint32_t>(expected),
                      "writer/extractor must retain exactly the intersecting candidates");
                expected_refs += expected;
            }
        }
    }
    Check(reference_count == expected_refs, "uniform ranges cover SAT references");
    Check(expected_refs > 0 && expected_refs < num_cells, "BBox false positives removed");
    for (uint32_t tri : refs) Check(tri == 0, "filtered triangle ID");
}

void CheckSatCellRanges(const happyray::SceneView& scene,
                        const happyray::GridRangeView& grid,
                        uint32_t reference_count) {
    const uint32_t num_cells = grid.resolution_x * grid.resolution_y * grid.resolution_z;
    const auto ranges = Download(grid.cell_ranges, num_cells);
    const auto refs = Download(grid.triangle_indices, reference_count);
    const happyray::BBox box = happyray::ComputeTriangleBBox(scene, 0);
    int x0, x1, y0, y1, z0, z1;
    happyray::ComputeCellRange(box, grid.bounds_min, grid.inv_cell_size,
                               grid.resolution_x, grid.resolution_y,
                               grid.resolution_z, &x0, &x1, &y0, &y1, &z0, &z1);
    const happyray::detail::TriangleCellIntersection intersects(scene, 0);
    uint32_t expected_refs = 0;
    uint32_t aabb_candidates = 0;
    for (int z = 0; z < grid.resolution_z; ++z) {
        for (int y = 0; y < grid.resolution_y; ++y) {
            for (int x = 0; x < grid.resolution_x; ++x) {
                const uint32_t cell = happyray::FlattenCellIndex(
                    x, y, z, grid.resolution_x, grid.resolution_y);
                const bool in_aabb_range = x >= x0 && x <= x1 && y >= y0 && y <= y1 &&
                                           z >= z0 && z <= z1;
                aabb_candidates += in_aabb_range;
                const bool expected =
                    in_aabb_range && intersects(
                        grid.bounds_min +
                            happyray::Vec3(x + 0.5f, y + 0.5f, z + 0.5f) *
                                grid.cell_size,
                        grid.cell_size);
                Check(ranges[cell].end - ranges[cell].start ==
                          static_cast<uint32_t>(expected),
                      "two-level ranges must retain exactly SAT-intersecting cells");
                expected_refs += expected;
            }
        }
    }
    Check(expected_refs == reference_count, "two-level range endpoints cover SAT references");
    Check(expected_refs < aabb_candidates, "SAT rejects AABB-only cell candidates");
    for (uint32_t tri : refs) Check(tri == 0, "two-level range triangle ID");
}

void CheckFragmentWriters() {
    using happyray::Vec3;
    const std::vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 1)};
    const std::vector<uint32_t> indices = {0, 1, 2};
    happyray::SceneView host_scene;
    host_scene.positions = positions.data();
    host_scene.indices = indices.data();
    host_scene.triangle_count = 1;
    const thrust::device_vector<Vec3> device_positions(positions.begin(), positions.end());
    const thrust::device_vector<uint32_t> device_indices(indices.begin(), indices.end());
    happyray::SceneView scene = host_scene;
    scene.positions = thrust::raw_pointer_cast(device_positions.data());
    scene.indices = thrust::raw_pointer_cast(device_indices.data());

    const happyray::UniformGrid uniform(scene, 128);
    CheckFilteredFragments(host_scene, uniform.View());
    const auto uniform_ranges = Download(
        uniform.View().cell_ranges,
        uniform.View().resolution_x * uniform.View().resolution_y *
            uniform.View().resolution_z);
    uint32_t uniform_reference_count = 0;
    for (const auto& range : uniform_ranges) {
        uniform_reference_count = std::max(uniform_reference_count, range.end);
    }
    Check(uniform.TriangleIndexCount() >= uniform_reference_count,
          "uniform ranges fit in pair storage including padding");

    const happyray::TwoLevelGrid two_level(scene, 128, 128, 0);
    auto two_view = two_level.View();
    const uint32_t num_top_cells = two_view.top_level.resolution_x *
                                   two_view.top_level.resolution_y *
                                   two_view.top_level.resolution_z;
    const auto top_ranges = Download(two_view.top_level.cell_ranges, num_top_cells);
    uint32_t top_reference_count = 0;
    for (const auto& range : top_ranges) {
        top_reference_count = std::max(top_reference_count, range.end);
    }
    CheckSatCellRanges(host_scene, two_view.top_level, top_reference_count);
    const auto top_level_cells = Download(two_view.top_level_cells, num_top_cells);
    two_view.top_level_cells = top_level_cells.data();
    const auto second_level_view = happyray::SecondLevelView(two_view, 0);
    Check(second_level_view.resolution_x == second_level_view.resolution_y &&
              second_level_view.resolution_y == second_level_view.resolution_z,
          "two-level second grid has equal axis resolutions");
    const uint32_t second_level_cell_count =
        second_level_view.resolution_x * second_level_view.resolution_y *
        second_level_view.resolution_z;
    const auto second_level_ranges =
        Download(second_level_view.cell_ranges, second_level_cell_count);
    uint32_t second_level_ref_end = 0;
    for (const auto& range : second_level_ranges) {
        second_level_ref_end = std::max(second_level_ref_end, range.end);
    }
    CheckSatCellRanges(host_scene, second_level_view, second_level_ref_end);

    const happyray::IrregularGrid irregular(scene, 0.12f, 128, 0, 0, 0);
    const auto irregular_view = irregular.View();
    const auto info = Download(irregular_view.cell_infos, 1).front();
    const uint32_t num_cells = info.resolution_x * info.resolution_y * info.resolution_z;
    const auto cells = Download(irregular_view.local_cells, num_cells);
    std::vector<happyray::CellRange> ranges;
    for (const auto& cell : cells) {
        ranges.push_back(cell.references);
    }
    const thrust::device_vector<happyray::CellRange> device_ranges(ranges.begin(),
                                                                   ranges.end());
    happyray::UniformGridView finest;
    finest.bounds_min = info.bounds_min;
    finest.cell_size = info.cell_size;
    finest.inv_cell_size = info.inv_cell_size;
    finest.resolution_x = info.resolution_x;
    finest.resolution_y = info.resolution_y;
    finest.resolution_z = info.resolution_z;
    finest.cell_ranges = thrust::raw_pointer_cast(device_ranges.data());
    finest.triangle_indices = irregular_view.triangle_indices;
    CheckFilteredFragments(host_scene, finest);

    const happyray::UniformGrid single_cell(scene, 1);
    Check(single_cell.TriangleIndexCount() == 1, "no rejected sentinel required for final cell");
        const auto single_ranges = Download(single_cell.View().cell_ranges, 1);
        Check(single_ranges[0].start == 0 && single_ranges[0].end == 1,
                    "single occupied-cell range endpoints");
}

void CheckCellStarts(const std::vector<uint32_t>& keys, uint32_t num_cells) {
    const thrust::device_vector<uint32_t> device_keys(keys.begin(), keys.end());
    thrust::device_vector<uint32_t> starts(num_cells + 1, static_cast<uint32_t>(keys.size()));
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(keys.size())),
        happyray::detail::CellExtractor{thrust::raw_pointer_cast(device_keys.data()),
                                    thrust::raw_pointer_cast(starts.data())});
    thrust::inclusive_scan(starts.rbegin(), starts.rend(), starts.rbegin(),
                           thrust::minimum<uint32_t>());
    const thrust::host_vector<uint32_t> actual = starts;
    if (actual.size() != num_cells + 1) {
        std::fprintf(stderr, "FAILED: incorrect CSR size\n");
        ++g_failures;
        return;
    }
    for (uint32_t cell = 0; cell <= num_cells; ++cell) {
        const uint32_t expected = static_cast<uint32_t>(
            std::lower_bound(keys.begin(), keys.end(), cell) - keys.begin());
        if (actual[cell] != expected) {
            std::fprintf(stderr,
                         "FAILED: cell %u of %u (%zu pairs): offset %u, expected %u\n",
                         cell, num_cells, keys.size(), actual[cell], expected);
            ++g_failures;
        }
    }
}

}  // namespace

int main() {
    CheckTriangleCellIntersection();
    CheckFragmentWriters();
    CheckCellStarts({}, 0);
    CheckCellStarts({}, 8);
    CheckCellStarts({0}, 1);
    CheckCellStarts({0, 0, 0}, 1);
    CheckCellStarts({0, 1, 2, 3}, 4);
    CheckCellStarts({3, 3, 6, 6, 6}, 10);  // Leading, interior, and trailing empty cells.
    CheckCellStarts({7}, 8);
    CheckCellStarts({0}, 8);
    CheckCellStarts({0, 8191}, 8192);
    CheckCellStarts(std::vector<uint32_t>(4097, 3), 8);

    // Exhaust every combination of empty, singleton, and repeated runs in a small grid.
    for (uint32_t pattern = 0; pattern < 243; ++pattern) {
        std::vector<uint32_t> keys;
        uint32_t counts = pattern;
        for (uint32_t cell = 0; cell < 5; ++cell) {
            keys.insert(keys.end(), counts % 3, cell);
            counts /= 3;
        }
        CheckCellStarts(keys, 5);
    }

    // Multiple runs crossing typical parallel execution block boundaries.
    std::vector<uint32_t> keys;
    for (uint32_t cell = 2; cell < 1024; cell += 7) {
        keys.insert(keys.end(), (cell * 17) % 513 + 1, cell);
    }
    CheckCellStarts(keys, 1024);

    std::printf("Grid construction tests: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
