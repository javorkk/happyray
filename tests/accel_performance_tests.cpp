// One warmup construction, one timed construction, and one random-ray batch per grid. Ordering is diagnostic;
// only correctness and allocation-tracking errors fail the test.
// Pure Thrust (no raw CUDA runtime API, no CUDA-specific launch syntax), so this file compiles
// either under Thrust's CUDA backend (nvcc, runs on the GPU) or its TBB backend (plain g++, runs
// in parallel on the CPU) -- see CMakeLists.txt's HAPPYRAY_THRUST_BACKEND option, and
// src/render/renderer.cpp's own header comment for the same pattern.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "sah_cost.h"
#include "scene.h"
#include "obj_loader.h"
#include "random.h"
#include "scene_intersect.h"

namespace {

using namespace happyray;

constexpr uint32_t kRayCount = 1024 * 1024;

template <typename T>
T* Raw(thrust::device_vector<T>& values) {
    return thrust::raw_pointer_cast(values.data());
}

struct DeviceScene {
    thrust::device_vector<Vec3> positions, normals, colors;
    thrust::device_vector<uint32_t> indices, materials;

    explicit DeviceScene(const FlatScene& scene)
        : positions(scene.positions), normals(scene.normals), colors(scene.material_diffuse_colors),
          indices(scene.indices), materials(scene.material_ids) {}

    SceneView View() {
        return {Raw(positions), Raw(normals), Raw(indices), Raw(materials), Raw(colors),
                static_cast<uint32_t>(materials.size())};
    }
};

HAPPYRAY_HD bool Intersect(const SceneView& scene, const UniformGridView& grid,
                        const Ray& ray, HitRecord* hit) {
    return IntersectSceneGrid(scene, grid, ray, hit);
}

HAPPYRAY_HD bool Intersect(const SceneView& scene, const TwoLevelGridView& grid,
                        const Ray& ray, HitRecord* hit) {
    return IntersectSceneTwoLevelGrid(scene, grid, ray, hit);
}

HAPPYRAY_HD bool Intersect(const SceneView& scene, const IrregularGridView& grid,
                        const Ray& ray, HitRecord* hit) {
    return IntersectSceneIrregularGrid(scene, grid, ray, hit);
}

template <typename View>
struct Trace {
    SceneView scene;
    View grid;
    const Ray* rays;

    HAPPYRAY_HD float operator()(uint32_t i) const {
        HitRecord hit;
        return Intersect(scene, grid, rays[i], &hit) ? hit.t : -1.0f;
    }
};

// Forces whichever backend's pending device work (if any -- none at all on the TBB backend,
// where device_vector already lives in ordinary host memory) to finish before timing continues:
// dereferencing a thrust::device_ptr requires the CUDA backend to synchronize and copy that one
// element back to host first, the same effect cudaDeviceSynchronize() would have, without any
// raw CUDA API call. Used (instead of CUDA events) to bound build/trace timing below.
template <typename T>
void Synchronize(const T* device_data) {
    static_cast<void>(static_cast<T>(*thrust::device_pointer_cast(device_data)));
}

void SyncGrid(const UniformGridView& view) { Synchronize(view.cell_range_start); }
void SyncGrid(const TwoLevelGridView& view) { Synchronize(view.top_level.cell_range_start); }
void SyncGrid(const IrregularGridView& view) {
    Synchronize(view.top_level.cell_range_start);
}

// Analytic per-grid-type memory-footprint estimate: count * sizeof(element), summed over every
// device buffer that grid type actually holds, using only its own plain public element-count
// accessors (TriangleIndexCount(), LocalCellCount(), etc. -- see uniform_grid.h/two_level_grid.h/
// irregular_grid.h) and View()'s own resolution fields -- no CUDA-specific allocation
// introspection (this file used to track real cudaMalloc/cudaFree calls via CUPTI) needed, at
// the cost of only reporting each structure's final, persistent footprint, not the transient
// peak during construction (its own internal scratch buffers aren't visible from outside the
// class).
size_t GridMemoryBytes(const UniformGrid& grid) {
    const UniformGridView view = grid.View();
    const auto num_cells =
        static_cast<size_t>(view.resolution_x) * view.resolution_y * view.resolution_z;
    return 2 * num_cells * sizeof(uint32_t) +
          static_cast<size_t>(grid.TriangleIndexCount()) * sizeof(uint32_t);
}

size_t GridMemoryBytes(const TwoLevelGrid& grid) {
    const TwoLevelGridView view = grid.View();
    const auto num_top_cells = static_cast<size_t>(view.top_level.resolution_x) *
                               view.top_level.resolution_y * view.top_level.resolution_z;
    size_t bytes = 2 * num_top_cells * sizeof(uint32_t);
    bytes += static_cast<size_t>(grid.TopLevelTriangleIndexCount()) * sizeof(uint32_t);
    bytes += num_top_cells * sizeof(SecondLevelGridInfo);
    bytes += 2 * static_cast<size_t>(grid.SecondLevelCellCount()) * sizeof(uint32_t);
    bytes += static_cast<size_t>(grid.SecondLevelTriangleIndexCount()) * sizeof(uint32_t);
    return bytes;
}

size_t GridMemoryBytes(const IrregularGrid& grid) {
    const IrregularGridView view = grid.View();
    const auto num_top_cells = static_cast<size_t>(view.top_level.resolution_x) *
                               view.top_level.resolution_y * view.top_level.resolution_z;
    size_t bytes = 2 * num_top_cells * sizeof(uint32_t);
    bytes += static_cast<size_t>(grid.TopLevelTriangleIndexCount()) * sizeof(uint32_t);
    bytes += num_top_cells * sizeof(IrregularCellInfo);
    bytes += num_top_cells * sizeof(IrregularTopCell);
    bytes += static_cast<size_t>(grid.LocalEntryCount()) * sizeof(uint32_t);
    bytes += static_cast<size_t>(grid.LocalCellCount()) * sizeof(IrregularCell);
    bytes += static_cast<size_t>(grid.TriangleIndexCount()) * sizeof(uint32_t);
    return bytes;
}

struct Measurement {
    std::string name;
    double build_ms;
    float trace_ms;
    size_t memory_bytes;
    float sah_cost;
    std::vector<float> hits;

    double Mrays() const { return kRayCount / (1000.0 * trace_ms); }
};

template <typename MakeGrid>
Measurement Measure(const std::string& name, MakeGrid make_grid, const SceneView& scene,
                     const BBox& scene_bounds, const SahCostConstants& costs,
                     thrust::device_vector<Ray>& rays) {
    {
        auto warmup_grid = make_grid();
        SyncGrid(warmup_grid->View());
    }

    const auto start = std::chrono::steady_clock::now();
    auto grid = make_grid();
    SyncGrid(grid->View());
    const auto end = std::chrono::steady_clock::now();
    const size_t memory_bytes = GridMemoryBytes(*grid);
    const double build_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const float sah_cost = grid->ComputeSahCost(scene_bounds, costs);

    thrust::device_vector<float> hits(kRayCount);
    const auto trace_begin = std::chrono::steady_clock::now();
    thrust::transform(thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(kRayCount), hits.begin(),
                      Trace<decltype(grid->View())>{scene, grid->View(), Raw(rays)});
    Synchronize(Raw(hits));
    const auto trace_end = std::chrono::steady_clock::now();
    const float trace_ms =
        std::chrono::duration<float, std::milli>(trace_end - trace_begin).count();
    if (!std::isfinite(build_ms) || build_ms <= 0 || !std::isfinite(trace_ms) || trace_ms <= 0)
        throw std::runtime_error("Invalid measured duration");
    if (!std::isfinite(sah_cost) || sah_cost <= 0)
        throw std::runtime_error("Invalid SAH cost estimate");
    if (memory_bytes == 0) throw std::runtime_error("Invalid memory accounting");
    const thrust::host_vector<float> host = hits;
    return {name, build_ms, trace_ms, memory_bytes, sah_cost, {host.begin(), host.end()}};
}

bool SameHit(float a, float b) {
    return (a == -1 && b == -1) ||
           (a >= 0 && b >= 0 && std::isfinite(a) && std::isfinite(b) &&
            std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(b)));
}

void Run(const char* path) {
    Scene scene;
    if (!LoadObj(path, &scene)) throw std::runtime_error("Cannot load scene");
    const BBox bounds = scene.WorldBounds();
    const double limit_squared =
        0.01 * SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
    double max_edge_squared = 0;
    for (const Mesh& mesh : scene.meshes) {
        for (size_t tri = 0; tri < mesh.indices.size(); tri += 3) {
            for (size_t edge = 0; edge < 3; ++edge) {
                const double length = SceneEdgeLengthSquared(
                    mesh.positions[mesh.indices[tri + edge]],
                    mesh.positions[mesh.indices[tri + (edge + 1) % 3]]);
                if (!std::isfinite(length) || length > limit_squared)
                    throw std::runtime_error("Scene subdivision edge limit violated");
                max_edge_squared = std::max(max_edge_squared, length);
            }
        }
    }
    const FlatScene flat = FlattenScene(scene);
    if (flat.material_ids.empty()) throw std::runtime_error("Scene contains no triangles");
    DeviceScene device(flat);
    const SceneView view = device.View();
    std::vector<Ray> rays;
    rays.reserve(kRayCount);
    Rng rng(20261021, 17);
    for (uint32_t i = 0; i < kRayCount; ++i) {
        const float x = rng.NextFloat(), y = rng.NextFloat(), z = rng.NextFloat();
        const Vec3 origin = bounds.min_bound + bounds.Extent() * Vec3(x, y, z);
        const float dz = 2 * rng.NextFloat() - 1;
        const float phi = 6.283185307179586f * rng.NextFloat();
        const float r = std::sqrt(std::max(0.0f, 1 - dz * dz));
        rays.emplace_back(origin, Vec3(r * std::cos(phi), r * std::sin(phi), dz));
    }
    thrust::device_vector<Ray> device_rays(rays);
    const SahCalibration calibration = MeasureSahCalibration();
    const SahCostConstants& uniform_costs = calibration.uniform;
    const SahCostConstants& two_level_costs = calibration.two_level;
    const SahCostConstants& irregular_costs = calibration.irregular;
    std::vector<Measurement> results;
    auto label = [](const char* name, float top, float leaf) {
        char text[96];
        if (top > 0)
            std::snprintf(text, sizeof(text), "%s (top=%g, leaf=%g)", name, top, leaf);
        else
            std::snprintf(text, sizeof(text), "%s (density=%g)", name, leaf);
        return std::string(text);
    };
    results.push_back(Measure(label("uniform", 0, UniformGrid::kDefaultDensity), [&] {
        return std::make_unique<UniformGrid>(view);
    }, view, bounds, uniform_costs, device_rays));
    results.push_back(Measure(label("two-level", TwoLevelGrid::kDefaultTopLevelDensity,
                                   TwoLevelGrid::kDefaultSecondLevelDensity), [&] {
        return std::make_unique<TwoLevelGrid>(view);
    }, view, bounds, two_level_costs, device_rays));
    results.push_back(Measure(label("irregular", IrregularGrid::kDefaultTopLevelDensity,
                                   IrregularGrid::kDefaultSecondLevelDensity), [&] {
        return std::make_unique<IrregularGrid>(view);
    }, view, bounds, irregular_costs, device_rays));

    std::printf("\nScene: %s (%u triangles), %u random closest-hit rays\n", path,
                view.triangle_count, kRayCount);
    std::printf("Subdivision: edge limit %.12g, maximum edge %.12g (0.1 x scene diagonal)\n",
                std::sqrt(limit_squared), std::sqrt(max_edge_squared));
    PrintSahCalibration(calibration);
    std::printf("%-38s %12s %12s %12s %14s %12s\n", "Grid (default densities)", "Build ms",
                "Random ms", "Mrays/s", "Memory MiB", "SAH cost");
    for (const auto& result : results)
        std::printf("%-38s %12.3f %12.3f %12.2f %14.3f %12.3f\n", result.name.c_str(),
                    result.build_ms, result.trace_ms, result.Mrays(),
                    result.memory_bytes / 1048576.0, result.sah_cost);
    const auto& u = results[0];
    const auto& t = results[1];
    const auto& g = results[2];
    auto order = [](const char* metric, bool matches) {
        if (!matches)
            std::printf("Ordering uniform < two-level < irregular (%s): NOT OBSERVED\n", metric);
    };
    order("build time", u.build_ms < t.build_ms && t.build_ms < g.build_ms);
    order("random throughput", u.Mrays() < t.Mrays() && t.Mrays() < g.Mrays());
    order("memory", u.memory_bytes < t.memory_bytes && t.memory_bytes < g.memory_bytes);
    order("SAH cost", u.sah_cost > t.sah_cost && t.sah_cost > g.sah_cost);

    size_t mismatches = 0;
    for (const auto& result : results) {
        for (uint32_t i = 0; i < kRayCount; ++i)
            if (!SameHit(result.hits[i], u.hits[i])) ++mismatches;
        for (uint32_t sample = 0; sample < 256; ++sample) {
            const uint32_t i = sample * (kRayCount / 256);
            HitRecord hit;
            const float expected = IntersectSceneBruteForce(flat.View(), rays[i], &hit) ? hit.t : -1;
            if (!SameHit(result.hits[i], expected)) ++mismatches;
        }
    }
    if (mismatches) {
        std::printf("Closest-hit mismatches: %zu (full-batch comparison + 256 oracle rays/grid)\n",
                    mismatches);
        throw std::runtime_error("Closest-hit correctness check failed");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: happyray_accel_performance_tests scene.obj");
        Run(argv[1]);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Acceleration performance test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
