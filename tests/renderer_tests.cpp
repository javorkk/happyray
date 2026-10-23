// Automated regression test for Renderer (src/render/renderer.h): validates whichever Thrust
// backend this binary was built with (CUDA or TBB -- see CMakeLists.txt's HAPPYRAY_THRUST_BACKEND)
// against a dead-simple, single-threaded, Thrust-free host reference loop that calls
// AOIntegrator::Integrate (src/render/ao_integrator.h) directly. Also validates the uniform grid
// (src/accel/uniform_grid.h), the two-level grid (src/accel/two_level_grid.h), and the
// irregular grid (src/accel/irregular_grid.h) by checking Renderer-with-the-grid-enabled against
// the same brute-force reference: since both trace identical deterministic Halton samples, any
// discrepancy points at the grid's construction/traversal rather than at noise. Each grid is
// checked against two scenes: box_on_floor.obj (forced sparse threshold so
// each grid's degenerate path is exercised) and dense_cube.obj (dense enough that some
// top-level cells get real, non-degenerate refinement -- for the irregular grid, this also
// exercises real merge and expand, not just the flat initial build).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "camera.h"
#include "scene.h"
#include "obj_loader.h"
#include "random.h"
#include "scene_intersect.h"
#include "renderer.h"
#include "grid_density_search.h"

namespace {

int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                             \
    do {                                                                             \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                           \
        }                                                                           \
    } while (0)

// Reproduces Renderer::ResizeFramebuffer's per-pixel jitter table exactly (same Rng/HashSeed
// calls), so this reference loop draws the identical Halton samples Renderer does.
std::vector<happyray::Vec2> MakePixelJitter(int width, int height) {
    std::vector<happyray::Vec2> jitter(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            happyray::Rng rng(happyray::HashSeed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0u),
                          static_cast<uint64_t>(y) * width + x);
            jitter[static_cast<size_t>(y) * width + x] =
                happyray::Vec2(rng.NextFloat(), rng.NextFloat());
        }
    }
    return jitter;
}

// Renders `num_frames` accumulated AO samples single-threaded on the host, using brute force
// only -- this is the project's simplest possible ground truth, with no Thrust/parallelism
// involved at all.
template <typename Integrator>
std::vector<happyray::Vec3> RenderBruteForceReference(const happyray::SceneView& scene,
                                                  const happyray::Camera& camera,
                                                  const Integrator& integrator, int width,
                                                  int height,
                                                  const std::vector<happyray::Vec2>& pixel_jitter,
                                                  int num_frames, int samples_per_frame = 1) {
    const happyray::AccelStructures accel;  // Default: brute force.
    std::vector<happyray::Vec3> accumulated(static_cast<size_t>(width) * height,
                                        happyray::Vec3(0.0f, 0.0f, 0.0f));
    for (int frame = 0; frame < num_frames; ++frame) {
        const uint32_t sample_offset =
            static_cast<uint32_t>(frame) * samples_per_frame;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t idx = static_cast<size_t>(y) * width + x;
                const happyray::Ray ray = camera.GenerateRay(static_cast<float>(x),
                                                         static_cast<float>(y), width, height);
                accumulated[idx] += integrator.Integrate(scene, accel, ray, sample_offset,
                                                         pixel_jitter[idx]);
            }
        }
    }
    std::vector<happyray::Vec3> resolved(accumulated.size());
    for (size_t i = 0; i < accumulated.size(); ++i) {
        resolved[i] = accumulated[i] / static_cast<float>(num_frames);
    }
    return resolved;
}

// Renders `num_frames` accumulated AO samples via `renderer` and checks the result agrees closely
// with `reference` (the brute-force host loop above).
template <typename Integrator>
void CompareRendererToReference(const char* label, happyray::Renderer* renderer,
                             const happyray::Camera& camera, const Integrator& integrator,
                             int num_frames, const std::vector<happyray::Vec3>& reference) {
    renderer->ResetAccumulation();
    for (int frame = 0; frame < num_frames; ++frame) {
        renderer->RenderFrame(camera, integrator);
    }
    const std::vector<happyray::Vec3>& resolved = renderer->ResolvedPixels();

    HAPPYRAY_CHECK(resolved.size() == reference.size());

    float max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    const size_t pixel_count = reference.size();
    for (size_t i = 0; i < pixel_count; ++i) {
        const happyray::Vec3 diff = reference[i] - resolved[i];
        const float max_channel = std::fmax(std::fabs(diff.x),
                                            std::fmax(std::fabs(diff.y), std::fabs(diff.z)));
        max_abs_diff = std::fmax(max_channel, max_abs_diff);
        sum_abs_diff += (std::fabs(diff.x) + std::fabs(diff.y) + std::fabs(diff.z)) / 3;
    }
    const float mean_abs_diff = static_cast<float>(sum_abs_diff / pixel_count);

    std::printf("[%s] Renderer (%s) vs host reference: max abs diff = %.4f, mean abs diff = %.5f (%zu pixels)\n",
                label, happyray::RendererBackendName(), max_abs_diff, mean_abs_diff, pixel_count);

    // Both traces draw identical Halton samples against identical geometry, so they should agree
    // almost exactly; allow a small tolerance for host/device transcendental function rounding
    // differences (CUDA backend) or summation-order differences (TBB backend).
    HAPPYRAY_CHECK(max_abs_diff < 0.05f);
    HAPPYRAY_CHECK(mean_abs_diff < 0.01f);
}

void TestRendererLifecycle() {
    happyray::Renderer renderer(happyray::FlatScene{});
    const happyray::Camera camera;
    happyray::AOIntegrator integrator;
    integrator.background_color = happyray::Vec3(0.2f, 0.4f, 0.6f);
    integrator.samples_per_pixel_per_frame = 4;
    HAPPYRAY_CHECK(renderer.SampleCount() == 0);
    HAPPYRAY_CHECK(!renderer.HasUniformGrid());
    HAPPYRAY_CHECK(!renderer.HasTwoLevelGrid());
    HAPPYRAY_CHECK(!renderer.HasIrregularGrid());
    renderer.ResizeFramebuffer(3, 2);
    for (int frame = 0; frame < 3; ++frame) renderer.RenderFrame(camera, integrator);
    HAPPYRAY_CHECK(renderer.SampleCount() == 3);
    HAPPYRAY_CHECK(renderer.ResolvedPixels().size() == 6);
    for (const happyray::Vec3& pixel : renderer.ResolvedPixels()) {
        HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
    }
    renderer.ResetAccumulation();
    HAPPYRAY_CHECK(renderer.SampleCount() == 0);
    integrator.background_color = happyray::Vec3(0.8f, 0.6f, 0.4f);
    renderer.RenderFrame(camera, integrator);
    HAPPYRAY_CHECK(renderer.SampleCount() == 1);
    for (const happyray::Vec3& pixel : renderer.ResolvedPixels()) {
        HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
    }
    renderer.ResizeFramebuffer(2, 1);
    HAPPYRAY_CHECK(renderer.SampleCount() == 0);
    renderer.RenderFrame(camera, integrator);
    HAPPYRAY_CHECK(renderer.ResolvedPixels().size() == 2);
    for (const happyray::Vec3& pixel : renderer.ResolvedPixels()) {
        HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
    }
}

void TestProgressiveBlend(const happyray::FlatScene& scene, happyray::Camera camera) {
    happyray::Renderer progressive(scene), ao_renderer(scene), simple_renderer(scene);
    happyray::AOIntegrator ao;
    ao.samples_per_pixel_per_frame = 3;
    ao.max_distance = 1.5f;
    ao.background_color = happyray::Vec3(0.1f, 0.3f, 0.5f);
    happyray::SimpleIntegrator simple;
    simple.background_color = happyray::Vec3(0.6f, 0.4f, 0.2f);
    simple.light_direction = camera.Forward() * -1.0f;
    progressive.RenderProgressiveFrame(camera, ao, simple);
    HAPPYRAY_CHECK(progressive.SampleCount() == 0);

    auto check_pixels = [&](const std::vector<happyray::Vec3>& expected) {
        HAPPYRAY_CHECK(progressive.ResolvedPixels().size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            HAPPYRAY_CHECK(happyray::Length(progressive.ResolvedPixels()[i] - expected[i]) < 1e-5f);
    };
    auto check_sequence = [&](int width, int height) {
        progressive.ResizeFramebuffer(width, height);
        ao_renderer.ResizeFramebuffer(width, height);
        simple_renderer.ResizeFramebuffer(width, height);
        simple_renderer.RenderFrame(camera, simple);
        progressive.RenderProgressiveFrame(camera, ao, simple);
        HAPPYRAY_CHECK(progressive.SampleCount() == 1);
        check_pixels(simple_renderer.ResolvedPixels());
        for (uint32_t n = 1; n <= 64; ++n) {
            ao_renderer.RenderFrame(camera, ao);
            progressive.RenderProgressiveFrame(camera, ao, simple);
            HAPPYRAY_CHECK(progressive.SampleCount() == n + 1);
            const float weight = static_cast<float>(n) / (n + 32.0f);
            std::vector<happyray::Vec3> expected(static_cast<size_t>(width) * height);
            for (size_t i = 0; i < expected.size(); ++i)
                expected[i] = happyray::Lerp(simple_renderer.ResolvedPixels()[i],
                                         ao_renderer.ResolvedPixels()[i], weight);
            check_pixels(expected);
        }
    };
    check_sequence(12, 8);
    // Repeated motion resets must never retain AO or a preview from the previous view.
    for (int move = 0; move < 3; ++move) {
        camera.MoveRight(0.2f);
        simple.light_direction = camera.Forward() * -1.0f;
        progressive.ResetAccumulation();
        simple_renderer.ResetAccumulation();
        simple_renderer.RenderFrame(camera, simple);
        progressive.RenderProgressiveFrame(camera, ao, simple);
        HAPPYRAY_CHECK(progressive.SampleCount() == 1);
        check_pixels(simple_renderer.ResolvedPixels());
    }
    check_sequence(9, 5);
    progressive.BuildUniformGrid();
    progressive.SetAccelType(happyray::AccelType::kUniformGrid);
    progressive.ResetAccumulation();
    progressive.RenderProgressiveFrame(camera, ao, simple);
    check_pixels(simple_renderer.ResolvedPixels());
}

void TestDensitySearchScheduling() {
    using namespace happyray;
    GridDensitySearch search;
    const AccelType types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid,
                              AccelType::kIrregularGrid};
    for (AccelType type : types) {
        search.Reset(type, 100.0f);
        HAPPYRAY_CHECK(search.CanRebuild(32));   // No waiting for a prior measurement anymore --
                                             // a SAH cost is available the instant a candidate
                                             // finishes building.
        HAPPYRAY_CHECK(!search.CanRebuild(31));  // Still needs the view to have settled.
        const GridDensities minimum = search.Candidate();
        float cost = 100.0f;
        for (int i = 0; i < 100; ++i) {
            HAPPYRAY_CHECK(search.CanRebuild(32));
            const GridDensities candidate = search.Candidate();
            HAPPYRAY_CHECK(candidate.top_level >= minimum.top_level);
            HAPPYRAY_CHECK(candidate.top_level <= minimum.top_level * 16);
            HAPPYRAY_CHECK(candidate.second_level >= minimum.second_level);
            HAPPYRAY_CHECK(candidate.second_level <= minimum.second_level * 16);
            cost *= 0.9f;  // Strictly decreasing (but always positive) -- every candidate
                          // improves on the last.
            HAPPYRAY_CHECK(search.Evaluate(cost));
            HAPPYRAY_CHECK(search.CanRebuild(32));  // Next candidate needs no waiting frames.
        }
    }

    search.Reset(AccelType::kUniformGrid, 10.0f);
    for (int i = 0; i < 20; ++i) {
        HAPPYRAY_CHECK(search.CanRebuild(32));
        const bool improve = (i == 9);  // One improvement resets nine misses.
        HAPPYRAY_CHECK(search.Evaluate(improve ? 5.0f : 20.0f) == improve);
        if (i == 9) HAPPYRAY_CHECK(search.Misses() == 0);
    }
    HAPPYRAY_CHECK(search.Attempts() == 20);
    HAPPYRAY_CHECK(search.Stopped());
    HAPPYRAY_CHECK(!search.CanRebuild(1000));  // A stopped search never resumes on its own.

    search.Reset(AccelType::kUniformGrid, 10.0f);
    HAPPYRAY_CHECK(search.Stopped());  // Stopped state persists across Reset for the same type...
    HAPPYRAY_CHECK(!search.CanRebuild(32));
    search.Reset(AccelType::kTwoLevelGrid, 10.0f);
    HAPPYRAY_CHECK(!search.Stopped());  // ...but not for a different one.
    HAPPYRAY_CHECK(search.CanRebuild(32));

    for (AccelType type : {AccelType::kTwoLevelGrid, AccelType::kIrregularGrid}) {
        search.Reset(type, 10.0f);
        HAPPYRAY_CHECK(!search.Stopped());
        for (int i = 0; i < 10; ++i) {
            HAPPYRAY_CHECK(search.CanRebuild(32));
            HAPPYRAY_CHECK(!search.Evaluate(20.0f));  // Strictly worse than the 10.0f baseline.
        }
        HAPPYRAY_CHECK(search.Stopped());
        search.Reset(AccelType::kUniformGrid, 10.0f);
        HAPPYRAY_CHECK(search.Stopped());  // kUniformGrid's own earlier stop, unaffected by `type`.
        search.Reset(type, 10.0f);
        HAPPYRAY_CHECK(search.Stopped());
        HAPPYRAY_CHECK(!search.CanRebuild(1000));
    }
}

void TestDensityGridRebuilds(const happyray::FlatScene& scene, const happyray::Camera& camera) {
    using namespace happyray;
    Renderer renderer(scene), reference(scene);
    renderer.ResizeFramebuffer(24, 16);
    reference.ResizeFramebuffer(24, 16);
    renderer.BuildUniformGrid();
    renderer.BuildTwoLevelGrid(0.0625f, 1.2f, 4);
    renderer.BuildIrregularGrid(0.12f, 2.4f, 4, 0.6f, 1);
    AOIntegrator ao;
    ao.max_distance = 1.5f;
    SimpleIntegrator simple;
    const AccelType types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid,
                              AccelType::kIrregularGrid};
    for (AccelType type : types) {
        renderer.SetAccelType(type);
        renderer.ResetAccumulation();
        reference.ResetAccumulation();
        renderer.RenderProgressiveFrame(camera, ao, simple);
        reference.RenderProgressiveFrame(camera, ao, simple);
        const GridDensities original = renderer.CurrentGridDensities();
        const GridDensities candidate{original.top_level * 2, original.second_level * 0.5f};
        auto check_pixels = [&] {
            HAPPYRAY_CHECK(renderer.SampleCount() == reference.SampleCount());
            for (size_t i = 0; i < reference.ResolvedPixels().size(); ++i)
                HAPPYRAY_CHECK(Length(renderer.ResolvedPixels()[i] - reference.ResolvedPixels()[i]) <
                           1e-3f);
        };
        for (bool keep : {false, true}) {
            const uint32_t samples = renderer.SampleCount();
            renderer.RebuildGridDensities(candidate);
            HAPPYRAY_CHECK(renderer.SampleCount() == samples);  // No hidden trial renders.
            check_pixels();
            HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == candidate.top_level);
            renderer.RenderProgressiveFrame(camera, ao, simple);
            reference.RenderProgressiveFrame(camera, ao, simple);
            check_pixels();
            if (!keep) renderer.RebuildGridDensities(original);
            check_pixels();
            const GridDensities expected = keep ? candidate : original;
            HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == expected.top_level);
            HAPPYRAY_CHECK(renderer.CurrentGridDensities().second_level == expected.second_level);
            renderer.RenderProgressiveFrame(camera, ao, simple);
            reference.RenderProgressiveFrame(camera, ao, simple);
            check_pixels();
        }
        renderer.RebuildGridDensities(original);
        renderer.SetAccelType(AccelType::kBruteForce);
        renderer.SetAccelType(type);
        renderer.RebuildGridDensities(candidate);
        HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == candidate.top_level);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string assets_dir = (argc > 1) ? argv[1] : "assets";

    TestRendererLifecycle();
    TestDensitySearchScheduling();
    happyray::Scene scene;
    HAPPYRAY_CHECK(happyray::LoadObj(assets_dir + "/box_on_floor.obj", &scene));
    const happyray::FlatScene flat_scene = happyray::FlattenScene(scene);
    const happyray::SceneView view = flat_scene.View();

    constexpr int kWidth = 96;
    constexpr int kHeight = 54;
    constexpr int kNumFrames = 8;

    happyray::Camera camera;
    camera.position = happyray::Vec3(0.0f, 1.8f, 3.5f);
    camera.yaw_deg = -90.0f;
    camera.pitch_deg = -25.0f;

    TestProgressiveBlend(flat_scene, camera);
    TestDensityGridRebuilds(flat_scene, camera);

    happyray::AOIntegrator integrator;
    integrator.samples_per_pixel_per_frame = 4;
    integrator.max_distance = 1.5f;

    const std::vector<happyray::Vec2> pixel_jitter = MakePixelJitter(kWidth, kHeight);
    const std::vector<happyray::Vec3> reference =
        RenderBruteForceReference(view, camera, integrator, kWidth, kHeight, pixel_jitter,
                                  kNumFrames, integrator.samples_per_pixel_per_frame);

    happyray::Renderer renderer(flat_scene);
    renderer.ResizeFramebuffer(kWidth, kHeight);

    renderer.SetAccelType(happyray::AccelType::kBruteForce);
    CompareRendererToReference("brute force", &renderer, camera, integrator, kNumFrames, reference);

    renderer.BuildUniformGrid();
    renderer.SetAccelType(happyray::AccelType::kUniformGrid);
    CompareRendererToReference("uniform grid", &renderer, camera, integrator, kNumFrames, reference);

    // Scene-loading subdivision increases triangle counts; force the sparse path explicitly.
    renderer.BuildTwoLevelGrid(0.0625f, 1.2f, std::numeric_limits<uint32_t>::max());
    renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
    CompareRendererToReference("two-level grid (degenerate)", &renderer, camera, integrator, kNumFrames,
                             reference);

    renderer.BuildIrregularGrid(0.12f, 2.4f, std::numeric_limits<uint32_t>::max());
    renderer.SetAccelType(happyray::AccelType::kIrregularGrid);
    CompareRendererToReference("irregular grid (degenerate)", &renderer, camera, integrator, kNumFrames,
                             reference);

    happyray::SimpleIntegrator simple_integrator;
    simple_integrator.light_direction = camera.Forward() * -1.0f;
    const auto simple_reference = RenderBruteForceReference(
        view, camera, simple_integrator, kWidth, kHeight, pixel_jitter, 1);
    const happyray::AccelType simple_accels[] = {happyray::AccelType::kBruteForce,
        happyray::AccelType::kUniformGrid, happyray::AccelType::kTwoLevelGrid,
        happyray::AccelType::kIrregularGrid};
    for (happyray::AccelType type : simple_accels) {
        renderer.SetAccelType(type);
        CompareRendererToReference("simple integrator", &renderer, camera, simple_integrator,
                                   2, simple_reference);
        HAPPYRAY_CHECK(renderer.SampleCount() == 2);
    }

    // The subdivided dense cube exercises real second-level refinement at default densities.
    happyray::Scene dense_scene;
    HAPPYRAY_CHECK(happyray::LoadObj(assets_dir + "/dense_cube.obj", &dense_scene));
    const happyray::FlatScene dense_flat_scene = happyray::FlattenScene(dense_scene);
    const happyray::SceneView dense_view = dense_flat_scene.View();

    happyray::Camera dense_camera;
    dense_camera.position = happyray::Vec3(1.6f, 1.2f, 2.2f);
    dense_camera.yaw_deg = -145.0f;
    dense_camera.pitch_deg = -20.0f;

    happyray::AOIntegrator dense_settings;
    dense_settings.samples_per_pixel_per_frame = 4;
    dense_settings.max_distance = 0.6f;

    const std::vector<happyray::Vec3> dense_reference =
        RenderBruteForceReference(dense_view, dense_camera, dense_settings, kWidth, kHeight,
                                  pixel_jitter, kNumFrames, dense_settings.samples_per_pixel_per_frame);

    happyray::Renderer dense_renderer(dense_flat_scene);
    dense_renderer.ResizeFramebuffer(kWidth, kHeight);
    dense_renderer.BuildTwoLevelGrid();
    dense_renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
    CompareRendererToReference("two-level grid (refined)", &dense_renderer, dense_camera,
                             dense_settings, kNumFrames, dense_reference);

    // Same scene, now checking the irregular grid: its per-top-level-cell local grids are dense
    // enough here to exercise real merge (collapsing empty/uniform regions) and expand
    // (growing cells into subset-compatible neighbors), not just the degenerate case above.
    dense_renderer.BuildIrregularGrid();
    dense_renderer.SetAccelType(happyray::AccelType::kIrregularGrid);
    CompareRendererToReference("irregular grid (merge+expand)", &dense_renderer, dense_camera,
                             dense_settings, kNumFrames, dense_reference);

    if (g_failures == 0) {
        std::printf("All renderer tests passed.\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d renderer test check(s) failed.\n", g_failures);
    return EXIT_FAILURE;
}
