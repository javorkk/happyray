// Headless benchmark harness: loads one or more OBJ scenes and times Renderer::RenderFrame (see
// src/render/renderer.h) across every acceleration structure it supports (brute force, uniform
// grid, two-level grid, irregular grid), reporting mean/min/max frame time and an estimated
// AO-ray throughput.
// Whichever Thrust backend this binary was built with (CUDA or TBB -- see CMakeLists.txt's
// HAPPYRAY_THRUST_BACKEND) is whatever gets measured; run the CUDA- and TBB-backend binaries
// separately and compare their printed tables to compare the two.
//
// Usage: happyray_benchmark [scene.obj ...] [--width W] [--height H] [--frames N] [--warmup N]
//                       [--samples N]
// With no scene arguments, benchmarks a small built-in set of bundled assets (found relative to
// --assets-dir, default "assets").
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "camera.h"
#include "scene.h"
#include "obj_loader.h"
#include "renderer.h"

namespace {

struct BenchmarkOptions {
    std::vector<std::string> scene_paths;
    std::string assets_dir = "assets";
    int width = 640;
    int height = 360;
    int warmup_frames = 4;
    int measured_frames = 32;
    int samples_per_pixel_per_frame = 4;
};

struct FrameStats {
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mrays_per_sec = 0.0;
};

struct AccelCase {
    happyray::AccelType type;
    const char* label;
};

bool ParseOptions(int argc, char** argv, BenchmarkOptions* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_int = [&](int* out) -> bool {
            if (i + 1 >= argc) return false;
            *out = std::atoi(argv[++i]);
            return true;
        };
        if (arg == "--width") {
            if (!next_int(&options->width)) return false;
        } else if (arg == "--height") {
            if (!next_int(&options->height)) return false;
        } else if (arg == "--frames") {
            if (!next_int(&options->measured_frames)) return false;
        } else if (arg == "--warmup") {
            if (!next_int(&options->warmup_frames)) return false;
        } else if (arg == "--samples") {
            if (!next_int(&options->samples_per_pixel_per_frame)) return false;
        } else if (arg == "--assets-dir") {
            if (i + 1 >= argc) return false;
            options->assets_dir = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "Unknown option '%s'\n", arg.c_str());
            return false;
        } else {
            options->scene_paths.push_back(arg);
        }
    }
    return true;
}

// Measures `options.measured_frames` calls to RenderFrame (after `options.warmup_frames`
// untimed ones, so the first call's one-off costs -- e.g. the CUDA driver's lazy context/kernel
// setup -- don't skew the reported numbers), assuming each covers the whole framebuffer at
// `options.samples_per_pixel_per_frame` AO samples per pixel.
FrameStats MeasureRenderFrame(happyray::Renderer* renderer, const happyray::Camera& camera,
                              const happyray::AOIntegrator& integrator, const BenchmarkOptions& options) {
    for (int i = 0; i < options.warmup_frames; ++i) {
        renderer->RenderFrame(camera, integrator);
    }

    std::vector<double> frame_ms(options.measured_frames);
    for (int i = 0; i < options.measured_frames; ++i) {
        const auto start = std::chrono::steady_clock::now();
        renderer->RenderFrame(camera, integrator);
        const auto end = std::chrono::steady_clock::now();
        frame_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    FrameStats stats;
    stats.min_ms = *std::min_element(frame_ms.begin(), frame_ms.end());
    stats.max_ms = *std::max_element(frame_ms.begin(), frame_ms.end());
    double total_ms = 0.0;
    for (double ms : frame_ms) total_ms += ms;
    stats.mean_ms = total_ms / options.measured_frames;

    const double rays_per_frame = static_cast<double>(options.width) * options.height *
                                  options.samples_per_pixel_per_frame;
    stats.mrays_per_sec = stats.mean_ms > 0.0 ? (rays_per_frame / 1000.0) / stats.mean_ms : 0.0;
    return stats;
}

void PrintRow(const std::string& scene_name, const char* accel_label, uint32_t triangle_count,
             const FrameStats& stats) {
    std::printf("%-24s %-14s %10u %12.3f %12.3f %12.3f %12.2f\n", scene_name.c_str(), accel_label,
               triangle_count, stats.mean_ms, stats.min_ms, stats.max_ms, stats.mrays_per_sec);
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkOptions options;
    if (!ParseOptions(argc, argv, &options)) {
        std::fprintf(stderr,
                     "Usage: %s [scene.obj ...] [--width W] [--height H] [--frames N] "
                     "[--warmup N] [--samples N] [--assets-dir DIR]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    if (options.scene_paths.empty()) {
        // A small, deliberately varied default set: a sparse scene with a floor (exercises the
        // uniform/two-level grids' degenerate, few-triangles-per-cell case) and the densely
        // tessellated cube added in Phase 5 (exercises real second-level refinement).
        for (const char* name : {"box_on_floor.obj", "dense_cube.obj"}) {
            options.scene_paths.push_back(options.assets_dir + "/" + name);
        }
    }

    std::printf("HappyRay benchmark -- backend: %s, resolution: %dx%d, samples/pixel/frame: %d, "
               "warmup/measured frames: %d/%d\n",
               happyray::RendererBackendName(), options.width, options.height,
               options.samples_per_pixel_per_frame, options.warmup_frames,
               options.measured_frames);
    std::printf("%-24s %-14s %10s %12s %12s %12s %12s\n", "Scene", "Accel", "Triangles",
               "Mean ms", "Min ms", "Max ms", "Mrays/s");

    bool any_failure = false;
    for (const std::string& scene_path : options.scene_paths) {
        happyray::Scene scene;
        if (!happyray::LoadObj(scene_path, &scene)) {
            std::fprintf(stderr, "Failed to load '%s', skipping.\n", scene_path.c_str());
            any_failure = true;
            continue;
        }
        const happyray::FlatScene flat_scene = happyray::FlattenScene(scene);

        happyray::Camera camera = happyray::MakeDefaultCamera(scene);
        happyray::LoadCameraConfig(happyray::CameraConfigPathFor(scene_path), &camera);
        happyray::AOIntegrator integrator;
        integrator.samples_per_pixel_per_frame = options.samples_per_pixel_per_frame;
        integrator.max_distance = happyray::DefaultMaxDistance(scene);

        happyray::Renderer renderer(flat_scene);
        renderer.ResizeFramebuffer(options.width, options.height);
        renderer.BuildUniformGrid();
        renderer.BuildTwoLevelGrid();
        renderer.BuildIrregularGrid();

        const size_t last_slash = scene_path.find_last_of('/');
        const std::string scene_name =
            last_slash == std::string::npos ? scene_path : scene_path.substr(last_slash + 1);

        static constexpr AccelCase kAccelCases[] = {
            {happyray::AccelType::kBruteForce, "brute force"},
            {happyray::AccelType::kUniformGrid, "uniform grid"},
            {happyray::AccelType::kTwoLevelGrid, "two-level grid"},
            {happyray::AccelType::kIrregularGrid, "irregular grid"},
        };
        for (const AccelCase& accel_case : kAccelCases) {
            renderer.SetAccelType(accel_case.type);
            renderer.ResetAccumulation();
            const FrameStats stats = MeasureRenderFrame(&renderer, camera, integrator, options);
            PrintRow(scene_name, accel_case.label, scene.TriangleCount(), stats);
        }
    }

    return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
