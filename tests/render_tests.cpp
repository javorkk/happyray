// Automated regression test for the brute-force ambient-occlusion integrator
// (src/render/ao_integrator.h): checks that AO is measurably lower near the base of a box resting
// on a floor (where the box occludes part of the hemisphere) than on open floor far from any
// occluder, and that a ray with no hit returns the background color untouched. Also covers the
// Camera's quaternion-based orientation (src/render/camera.h).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "camera.h"
#include "scene.h"
#include "obj_loader.h"
#include "ao_integrator.h"
#include "simple_integrator.h"

namespace {

int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                              \
    do {                                                                              \
        if (!(cond)) {                                                               \
            std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__);  \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

bool NearlyEqual(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }
bool NearlyEqual(const happyray::Vec3& a, const happyray::Vec3& b, float eps = 1e-3f) {
    return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps);
}

// Shoots a straight-down primary ray from `origin_above` and averages many AO occlusion
// samples (not just the 1-per-frame the interactive viewer uses) for a low-noise estimate.
float MeasureAoAtPoint(const happyray::SceneView& scene, const happyray::Vec3& origin_above,
                        float ao_radius) {
    happyray::AOIntegrator integrator;
    integrator.samples_per_pixel_per_frame = 512;
    integrator.max_distance = ao_radius;

    const happyray::Ray down_ray(origin_above, happyray::Vec3(0.0f, -1.0f, 0.0f));
    const happyray::AccelStructures accel;  // Default: brute force.
    const happyray::Vec3 ao = integrator.Integrate(scene, accel, down_ray,
                                              /*sample_offset=*/0, happyray::Vec2(0.37f, 0.81f));
    return ao.x;  // Grayscale: x == y == z.
}

void TestAOIntegrator() {
    using namespace happyray;
    FlatScene scene;
    scene.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    scene.normals = {Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1)};
    scene.indices = {0, 1, 2};
    scene.material_ids = {0};
    scene.material_diffuse_colors = {Vec3(0.2f, 0.4f, 0.8f)};
    AOIntegrator integrator;
    integrator.samples_per_pixel_per_frame = 16;
    integrator.background_color = Vec3(0.1f, 0.3f, 0.6f);
    const AccelStructures accel;
    const Vec2 jitter(0.37f, 0.81f);
    const Ray hit(Vec3(0.25f, 0.25f, 1), Vec3(0, 0, -1));
    const Ray miss(Vec3(0.25f, 0.25f, 1), Vec3(0, 0, 1));
    HAPPYRAY_CHECK(NearlyEqual(integrator.Integrate(scene.View(), accel, hit, 32, jitter),
                           scene.material_diffuse_colors[0]));
    HAPPYRAY_CHECK(NearlyEqual(integrator.Integrate(scene.View(), accel, miss, 32, jitter),
                           integrator.background_color));
}

void TestSimpleIntegrator() {
    using namespace happyray;
    FlatScene scene;
    scene.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    scene.normals = {Vec3(0, 0, 3), Vec3(0, 0, 3), Vec3(0, 0, 3)};
    scene.indices = {0, 1, 2};
    scene.material_ids = {0};
    scene.material_diffuse_colors = {Vec3(1.0f, 0.4f, 0.8f)};
    SimpleIntegrator integrator;
    integrator.background_color = Vec3(0.2f, 0.3f, 0.4f);
    const AccelStructures accel;
    const Ray hit(Vec3(0.25f, 0.25f, 1), Vec3(0, 0, -1));
    const Ray miss(Vec3(0.25f, 0.25f, 1), Vec3(0, 0, 1));
    const float intensities[] = {-1, 0, 0.1f, 0.2499f, 0.25f, 0.2501f,
                                 0.5f, 0.9499f, 0.95f, 0.9501f, 1};
    for (float intensity : intensities) {
        integrator.light_direction = Vec3(std::sqrt(1 - intensity * intensity), 0, intensity) * 2;
        const float expected = intensity > 0.95f ? 1.2f :
                               (intensity > 0.25f ? intensity : 0.1f);
        HAPPYRAY_CHECK(NearlyEqual(integrator.Integrate(scene.View(), accel, hit, 0, Vec2()),
                               scene.material_diffuse_colors[0] * expected, 1e-5f));
        HAPPYRAY_CHECK(NearlyEqual(integrator.Integrate(scene.View(), accel, hit, 100, Vec2(0.3f, 0.7f)),
                               scene.material_diffuse_colors[0] * expected, 1e-5f));
    }
    HAPPYRAY_CHECK(NearlyEqual(integrator.Integrate(scene.View(), accel, miss, 0, Vec2()),
                           integrator.background_color));
}

// Checks quaternion camera directions against the original spherical-coordinates formula.
void TestCameraOrientation() {
    using namespace happyray;
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

    auto reference_forward = [](float yaw_deg, float pitch_deg) {
        const float yaw = yaw_deg * kDegToRad;
        const float pitch = pitch_deg * kDegToRad;
        return Vec3(std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw));
    };
    auto reference_right = [&](float yaw_deg, float pitch_deg) {
        return Normalize(Cross(reference_forward(yaw_deg, pitch_deg), Vec3(0.0f, 1.0f, 0.0f)));
    };
    auto reference_up = [&](float yaw_deg, float pitch_deg) {
        const Vec3 f = reference_forward(yaw_deg, pitch_deg);
        return Cross(reference_right(yaw_deg, pitch_deg), f);
    };

    const float test_angles[][2] = {{-90.0f, 0.0f}, {-90.0f, -10.0f}, {0.0f, 0.0f},
                                     {45.0f, 30.0f}, {-135.0f, 20.0f}, {30.0f, -40.0f}};
    for (const auto& angles : test_angles) {
        Camera camera;
        camera.yaw_deg = angles[0];
        camera.pitch_deg = angles[1];

        HAPPYRAY_CHECK(NearlyEqual(camera.Forward(), reference_forward(angles[0], angles[1])));
        HAPPYRAY_CHECK(NearlyEqual(camera.Right(), reference_right(angles[0], angles[1])));
        HAPPYRAY_CHECK(NearlyEqual(camera.Up(), reference_up(angles[0], angles[1])));

        // The basis should always be orthonormal, regardless of yaw/pitch.
        HAPPYRAY_CHECK(NearlyEqual(Length(camera.Forward()), 1.0f));
        HAPPYRAY_CHECK(NearlyEqual(Length(camera.Right()), 1.0f));
        HAPPYRAY_CHECK(NearlyEqual(Length(camera.Up()), 1.0f));
        HAPPYRAY_CHECK(NearlyEqual(Dot(camera.Forward(), camera.Right()), 0.0f));
        HAPPYRAY_CHECK(NearlyEqual(Dot(camera.Forward(), camera.Up()), 0.0f));
        HAPPYRAY_CHECK(NearlyEqual(Dot(camera.Right(), camera.Up()), 0.0f));
    }

    // Look() accumulates yaw/pitch deltas and clamps pitch to +/-89 degrees.
    Camera camera;
    camera.Look(10.0f, 5.0f);
    HAPPYRAY_CHECK(NearlyEqual(camera.yaw_deg, -80.0f));
    HAPPYRAY_CHECK(NearlyEqual(camera.pitch_deg, 5.0f));
    camera.Look(0.0f, 1000.0f);
    HAPPYRAY_CHECK(NearlyEqual(camera.pitch_deg, 89.0f));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string assets_dir = (argc > 1) ? argv[1] : "assets";

    TestCameraOrientation();
    TestAOIntegrator();
    TestSimpleIntegrator();

    happyray::Scene scene;
    HAPPYRAY_CHECK(happyray::LoadObj(assets_dir + "/box_on_floor.obj", &scene));
    const happyray::FlatScene flat_scene = happyray::FlattenScene(scene);
    const happyray::SceneView view = flat_scene.View();

    // Floor is at y = 0; a point just above the floor and right next to the 1x1x1 box (centered
    // at the origin, so its vertical faces sit at |x| = 0.5 / |z| = 0.5) should see significant
    // occlusion from the box within a 1.5-unit AO radius. A point far out on the open floor
    // should see almost none.
    const float ao_near_box = MeasureAoAtPoint(view, happyray::Vec3(0.55f, 0.01f, 0.0f), 1.5f);
    const float ao_far_from_box = MeasureAoAtPoint(view, happyray::Vec3(2.5f, 0.01f, 0.0f), 1.5f);

    std::printf("AO near box base: %.3f, AO far from box: %.3f\n", ao_near_box, ao_far_from_box);

    HAPPYRAY_CHECK(ao_near_box < ao_far_from_box - 0.1f);
    HAPPYRAY_CHECK(ao_far_from_box > 0.85f);  // Open floor should be nearly fully lit.
    HAPPYRAY_CHECK(ao_near_box < 0.85f);      // Contact shadow should be clearly darker.

    if (g_failures == 0) {
        std::printf("All render tests passed.\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d render test check(s) failed.\n", g_failures);
    return EXIT_FAILURE;
}
