// Lightweight, dependency-free unit tests for src/core (no third-party test framework, matching
// the project's "no non-essential dependencies" preference). Each check prints a message and
// the test binary returns non-zero on first failure; registered with CTest.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "bbox.h"
#include "scene.h"
#include "random.h"
#include "mat4.h"
#include "mesh.h"
#include "obj_loader.h"
#include "quaternion.h"
#include "ray.h"
#include "vec.h"

namespace {

int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                         \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                       \
        }                                                                        \
    } while (0)

bool NearlyEqual(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
bool NearlyEqual(const happyray::Vec3& a, const happyray::Vec3& b, float eps = 1e-4f) {
    return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps);
}

void CheckSubdivisionLimit(const happyray::Scene& scene) {
    const auto bounds = scene.WorldBounds();
    const double limit_squared =
        0.01 * happyray::SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
    for (const auto& mesh : scene.meshes) {
        HAPPYRAY_CHECK(mesh.positions.size() == mesh.attributes.size());
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            for (size_t edge = 0; edge < 3; ++edge)
                HAPPYRAY_CHECK(happyray::SceneEdgeLengthSquared(
                    mesh.positions[mesh.indices[i + edge]],
                    mesh.positions[mesh.indices[i + (edge + 1) % 3]]) <= limit_squared);
        }
    }
}

void TestTriangleSubdivision() {
    using namespace happyray;
    Scene scene;
    SubdivideLargeTriangles(&scene);
    HAPPYRAY_CHECK(scene.meshes.empty());
    Mesh mesh;
    mesh.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    mesh.attributes = {{Vec3(0, 0, 1), Vec2(0, 0)},
                       {Vec3(0, 0, 3), Vec2(1, 0)},
                       {Vec3(0, 0, 2), Vec2(0, 1)}};
    mesh.indices = {0, 1, 2};
    mesh.material_ids = {7};
    scene.meshes.push_back(mesh);
    // Exercise a second mesh and a degenerate long triangle with no material array.
    mesh.indices = {0, 1, 1};
    mesh.material_ids.clear();
    scene.meshes.push_back(mesh);
    const BBox bounds = scene.WorldBounds();
    SubdivideLargeTriangles(&scene);
    CheckSubdivisionLimit(scene);
    HAPPYRAY_CHECK(NearlyEqual(scene.WorldBounds().min_bound, bounds.min_bound));
    HAPPYRAY_CHECK(NearlyEqual(scene.WorldBounds().max_bound, bounds.max_bound));
    const Mesh& result = scene.meshes.front();
    HAPPYRAY_CHECK(result.TriangleCount() > 1);
    HAPPYRAY_CHECK(result.material_ids.size() == result.TriangleCount());
    double area = 0;
    for (size_t tri = 0; tri < result.indices.size(); tri += 3) {
        const Vec3 a = result.positions[result.indices[tri]];
        const Vec3 b = result.positions[result.indices[tri + 1]];
        const Vec3 c = result.positions[result.indices[tri + 2]];
        const Vec3 normal = Cross(b - a, c - a);
        HAPPYRAY_CHECK(normal.z > 0);
        area += Length(normal) * 0.5;
        HAPPYRAY_CHECK(result.material_ids[tri / 3] == 7);
    }
    HAPPYRAY_CHECK(std::fabs(area - 0.5) < 1e-6);
    for (size_t i = 0; i < result.positions.size(); ++i) {
        const Vec3& position = result.positions[i];
        const VertexAttributes& attributes = result.attributes[i];
        HAPPYRAY_CHECK(NearlyEqual(attributes.uv.x, position.x));
        HAPPYRAY_CHECK(NearlyEqual(attributes.uv.y, position.y));
        HAPPYRAY_CHECK(NearlyEqual(attributes.normal.z, 1 + 2 * position.x + position.y));
    }
    HAPPYRAY_CHECK(scene.meshes[1].material_ids.empty());
    const uint32_t count = scene.TriangleCount();
    const size_t vertices = result.positions.size();
    SubdivideLargeTriangles(&scene);
    HAPPYRAY_CHECK(scene.TriangleCount() == count);
    HAPPYRAY_CHECK(scene.meshes.front().positions.size() == vertices);
    HAPPYRAY_CHECK(scene.meshes.front().attributes.size() == vertices);

    Scene point;
    mesh.positions = {Vec3(1, 2, 3)};
    mesh.attributes = {{Vec3(), Vec2()}};
    mesh.indices = {0, 0, 0};
    point.meshes.push_back(mesh);
    SubdivideLargeTriangles(&point);
    HAPPYRAY_CHECK(point.TriangleCount() == 1);
}

void TestVecMath() {
    using namespace happyray;
    const Vec3 a(1.0f, 0.0f, 0.0f);
    const Vec3 b(0.0f, 1.0f, 0.0f);

    HAPPYRAY_CHECK(NearlyEqual(Dot(a, b), 0.0f));
    HAPPYRAY_CHECK(NearlyEqual(Cross(a, b), Vec3(0.0f, 0.0f, 1.0f)));
    HAPPYRAY_CHECK(NearlyEqual(Length(Vec3(3.0f, 4.0f, 0.0f)), 5.0f));
    HAPPYRAY_CHECK(NearlyEqual(Normalize(Vec3(5.0f, 0.0f, 0.0f)), Vec3(1.0f, 0.0f, 0.0f)));
    HAPPYRAY_CHECK(NearlyEqual(Lerp(a, b, 0.5f), Vec3(0.5f, 0.5f, 0.0f)));

    const Vec3 incident(1.0f, -1.0f, 0.0f);
    const Vec3 normal(0.0f, 1.0f, 0.0f);
    HAPPYRAY_CHECK(NearlyEqual(Reflect(incident, normal), Vec3(1.0f, 1.0f, 0.0f)));
}

void TestMat4() {
    using namespace happyray;
    const Mat4 t = Mat4::Translation(Vec3(1.0f, 2.0f, 3.0f));
    HAPPYRAY_CHECK(NearlyEqual(TransformPoint(t, Vec3(0.0f, 0.0f, 0.0f)), Vec3(1.0f, 2.0f, 3.0f)));
    HAPPYRAY_CHECK(NearlyEqual(TransformVector(t, Vec3(1.0f, 0.0f, 0.0f)), Vec3(1.0f, 0.0f, 0.0f)));

    const Mat4 s = Mat4::Scale(Vec3(2.0f, 2.0f, 2.0f));
    HAPPYRAY_CHECK(NearlyEqual(TransformPoint(s, Vec3(1.0f, 1.0f, 1.0f)), Vec3(2.0f, 2.0f, 2.0f)));

    const Mat4 combined = t * s;
    HAPPYRAY_CHECK(NearlyEqual(TransformPoint(combined, Vec3(1.0f, 1.0f, 1.0f)), Vec3(3.0f, 4.0f, 5.0f)));

    const Mat4 inv_t = Inverse(t);
    HAPPYRAY_CHECK(NearlyEqual(TransformPoint(inv_t, Vec3(1.0f, 2.0f, 3.0f)), Vec3(0.0f, 0.0f, 0.0f)));

    const Mat4 view = Mat4::LookAt(Vec3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, 0.0f),
                                   Vec3(0.0f, 1.0f, 0.0f));
    // The camera looks down -Z in its own space, so the world origin (5 units in front of the
    // eye along +Z) should map to camera-space (0, 0, -5).
    HAPPYRAY_CHECK(NearlyEqual(TransformPoint(view, Vec3(0.0f, 0.0f, 0.0f)), Vec3(0.0f, 0.0f, -5.0f)));
}

void TestBBox() {
    using namespace happyray;
    BBox box;
    box.Grow(Vec3(-1.0f, -1.0f, -1.0f));
    box.Grow(Vec3(1.0f, 1.0f, 1.0f));
    HAPPYRAY_CHECK(!box.IsEmpty());
    HAPPYRAY_CHECK(NearlyEqual(box.Center(), Vec3(0.0f, 0.0f, 0.0f)));

    const Ray ray(Vec3(-5.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    float t_near = 0.0f, t_far = 0.0f;
    HAPPYRAY_CHECK(IntersectRayBBox(ray, box, &t_near, &t_far));
    HAPPYRAY_CHECK(NearlyEqual(t_near, 4.0f));
    HAPPYRAY_CHECK(NearlyEqual(t_far, 6.0f));

    const Ray miss_ray(Vec3(-5.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    HAPPYRAY_CHECK(!IntersectRayBBox(miss_ray, box, &t_near, &t_far));
}

void TestBBoxExtractor() {
    using namespace happyray;
    const Triangle triangle{{Vec3(-2, 3, 1), Vec3(4, -1, 2), Vec3(0, 2, -3)}};
    const BBox triangle_bounds = BBoxExtractor<Triangle>::Get(triangle);
    HAPPYRAY_CHECK(NearlyEqual(triangle_bounds.min_bound, Vec3(-2, -1, -3)));
    HAPPYRAY_CHECK(NearlyEqual(triangle_bounds.max_bound, Vec3(4, 3, 2)));

    const Primitive<4> quad{{triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], Vec3(5, 6, 7)}};
    const BBox quad_bounds = BBoxExtractor<Primitive<4>>::Get(quad);
    HAPPYRAY_CHECK(NearlyEqual(quad_bounds.min_bound, Vec3(-2, -1, -3)));
    HAPPYRAY_CHECK(NearlyEqual(quad_bounds.max_bound, Vec3(5, 6, 7)));
}

void TestQuaternion() {
    using namespace happyray;
    constexpr float kPi = 3.14159265358979323846f;

    // Identity leaves every vector untouched.
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(Quaternion::Identity(), Vec3(1.0f, 2.0f, 3.0f)),
                            Vec3(1.0f, 2.0f, 3.0f)));

    // A 90-degree rotation around +Z takes +X to +Y.
    const Quaternion rot_z90 = Quaternion::FromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi * 0.5f);
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(rot_z90, Vec3(1.0f, 0.0f, 0.0f)), Vec3(0.0f, 1.0f, 0.0f),
                           1e-3f));

    // A 90-degree rotation around +Y takes +Z to +X.
    const Quaternion rot_y90 = Quaternion::FromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), kPi * 0.5f);
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(rot_y90, Vec3(0.0f, 0.0f, 1.0f)), Vec3(1.0f, 0.0f, 0.0f),
                           1e-3f));

    // Composition: a * b applies b first, then a.
    const Quaternion composed = rot_y90 * rot_z90;
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(composed, Vec3(1.0f, 0.0f, 0.0f)),
                           RotateVector(rot_y90, RotateVector(rot_z90, Vec3(1.0f, 0.0f, 0.0f))), 1e-3f));

    // Slerp endpoints reproduce the inputs, and the midpoint is still a unit quaternion.
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(Slerp(rot_z90, rot_y90, 0.0f), Vec3(1.0f, 0.0f, 0.0f)),
                           RotateVector(rot_z90, Vec3(1.0f, 0.0f, 0.0f)), 1e-3f));
    HAPPYRAY_CHECK(NearlyEqual(RotateVector(Slerp(rot_z90, rot_y90, 1.0f), Vec3(1.0f, 0.0f, 0.0f)),
                           RotateVector(rot_y90, Vec3(1.0f, 0.0f, 0.0f)), 1e-3f));
    const Quaternion mid = Slerp(rot_z90, rot_y90, 0.5f);
    HAPPYRAY_CHECK(NearlyEqual(Length(mid), 1.0f, 1e-3f));

    // ToMat4 should agree with RotateVector for the same rotation.
    const Mat4 rot_z90_mat = ToMat4(rot_z90);
    HAPPYRAY_CHECK(NearlyEqual(TransformVector(rot_z90_mat, Vec3(1.0f, 0.0f, 0.0f)),
                           RotateVector(rot_z90, Vec3(1.0f, 0.0f, 0.0f)), 1e-3f));
}

void TestHalton() {
    using namespace happyray;

    // Base-2 radical inverse: bit-reverses the index around the binary point.
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(1, 2), 0.5f));
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(2, 2), 0.25f));
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(3, 2), 0.75f));
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(4, 2), 0.125f));

    // Base-3 radical inverse.
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(1, 3), 1.0f / 3.0f));
    HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(3, 3), 1.0f / 9.0f));

    // Every sample stays inside [0, 1), for a reasonable range of indices.
    for (uint32_t i = 0; i < 1000; ++i) {
        const Vec2 sample = Halton2D(i);
        HAPPYRAY_CHECK(sample.x >= 0.0f && sample.x < 1.0f);
        HAPPYRAY_CHECK(sample.y >= 0.0f && sample.y < 1.0f);
    }

    // Low discrepancy: the first N samples' mean should be close to (0.5, 0.5), much closer
    // than a handful of independent random draws would typically land.
    Vec2 sum(0.0f, 0.0f);
    constexpr int kN = 256;
    for (int i = 1; i <= kN; ++i) {
        const Vec2 sample = Halton2D(static_cast<uint32_t>(i));
        sum += sample;
    }
    const Vec2 mean = sum / static_cast<float>(kN);
    HAPPYRAY_CHECK(NearlyEqual(mean.x, 0.5f, 0.05f));
    HAPPYRAY_CHECK(NearlyEqual(mean.y, 0.5f, 0.05f));

    // Cranley-Patterson rotation wraps around [0, 1) and is a pure offset (no-op at offset 0).
    HAPPYRAY_CHECK(NearlyEqual(CranleyPattersonRotate(0.7f, 0.0f), 0.7f));
    HAPPYRAY_CHECK(NearlyEqual(CranleyPattersonRotate(0.7f, 0.5f), 0.2f));
    const float rotated = CranleyPattersonRotate(0.9f, 0.9f);
    HAPPYRAY_CHECK(rotated >= 0.0f && rotated < 1.0f);
    HAPPYRAY_CHECK(NearlyEqual(rotated, 0.8f, 1e-4f));
}

void TestObjLoaderTriangleNoNormals(const std::string& assets_dir) {
    using namespace happyray;
    Scene scene;
    HAPPYRAY_CHECK(LoadObj(assets_dir + "/triangle.obj", &scene));
    HAPPYRAY_CHECK(scene.meshes.size() == 1);
    if (scene.meshes.empty()) {
        return;
    }
    const Mesh& mesh = scene.meshes[0];
    HAPPYRAY_CHECK(mesh.positions.size() > 3);
    HAPPYRAY_CHECK(mesh.TriangleCount() > 1);
    CheckSubdivisionLimit(scene);
    // No vn lines were present, so LoadObj must have derived a flat normal of +Z for every
    // vertex of this single, counter-clockwise-when-viewed-from+Z triangle.
    for (const VertexAttributes& attributes : mesh.attributes) {
        HAPPYRAY_CHECK(NearlyEqual(attributes.normal, Vec3(0.0f, 0.0f, 1.0f)));
        HAPPYRAY_CHECK(NearlyEqual(attributes.uv.x, 0.0f));
        HAPPYRAY_CHECK(NearlyEqual(attributes.uv.y, 0.0f));
    }

    // No usemtl/mtllib at all: Scene::materials stays empty, but FlattenScene must still supply
    // a valid (white, default) diffuse color at index 0 for the unassigned material id every
    // triangle implicitly uses.
    HAPPYRAY_CHECK(scene.materials.empty());
    const FlatScene flat = FlattenScene(scene);
    HAPPYRAY_CHECK(!flat.material_diffuse_colors.empty());
    HAPPYRAY_CHECK(NearlyEqual(flat.material_diffuse_colors[0], Vec3(1.0f, 1.0f, 1.0f)));
}

void TestObjLoaderAttributeSeam(const std::string& assets_dir) {
    using namespace happyray;
    Scene scene;
    HAPPYRAY_CHECK(LoadObj(assets_dir + "/attribute_seam.obj", &scene));
    HAPPYRAY_CHECK(scene.meshes.size() == 1);
    if (scene.meshes.empty()) return;
    const Mesh& mesh = scene.meshes.front();
    CheckSubdivisionLimit(scene);
    HAPPYRAY_CHECK(mesh.positions.size() >= 6);
    if (mesh.positions.size() < 6 || mesh.attributes.size() < 6) return;
    const Vec2 expected_uvs[] = {Vec2(0, 0), Vec2(1, 0), Vec2(0, 1),
                                Vec2(1, 1), Vec2(0, 1), Vec2(1, 0)};
    for (size_t i = 0; i < 6; ++i) {
        HAPPYRAY_CHECK(NearlyEqual(mesh.positions[i], mesh.positions[i % 3]));
        HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].normal, Vec3(0, 0, i < 3 ? 1 : -1)));
        HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].uv.x, expected_uvs[i].x));
        HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].uv.y, expected_uvs[i].y));
    }
}

void TestDefaultMaxDistance() {
    happyray::Scene scene;
    happyray::Mesh mesh;
    mesh.positions = {happyray::Vec3(-2, 1, 0), happyray::Vec3(8, 5, 3)};
    scene.meshes.push_back(mesh);
    HAPPYRAY_CHECK(NearlyEqual(happyray::DefaultMaxDistance(scene), 1.0f));

    HAPPYRAY_CHECK(NearlyEqual(happyray::DefaultMaxDistance(happyray::Scene()), 1.0f));
}

void TestFlattenScene() {
    using namespace happyray;
    Scene scene;
    HAPPYRAY_CHECK(FlattenScene(scene).positions.empty());
    for (int i = 0; i < 2; ++i) {
        Mesh mesh;
        mesh.positions = {Vec3(i, 0, 0), Vec3(i, 1, 0), Vec3(i, 0, 1)};
        mesh.attributes = {{Vec3(1, i, 0), Vec2(0, 0)},
                           {Vec3(0, 1, i), Vec2(1, 0)},
                           {Vec3(i, 0, 1), Vec2(0, 1)}};
        mesh.indices = {2, 0, 1};
        mesh.material_ids = {0};
        scene.meshes.push_back(mesh);
    }
    const FlatScene flat = FlattenScene(scene);
    HAPPYRAY_CHECK(flat.positions.size() == 6);
    HAPPYRAY_CHECK(flat.normals.size() == 6);
    HAPPYRAY_CHECK(flat.indices == std::vector<uint32_t>({2, 0, 1, 5, 3, 4}));
    HAPPYRAY_CHECK(flat.material_ids == std::vector<uint32_t>({0, 0}));
    for (size_t m = 0; m < scene.meshes.size(); ++m) {
        for (size_t i = 0; i < 3; ++i) {
            HAPPYRAY_CHECK(NearlyEqual(flat.positions[m * 3 + i], scene.meshes[m].positions[i]));
            HAPPYRAY_CHECK(NearlyEqual(flat.normals[m * 3 + i], scene.meshes[m].attributes[i].normal));
        }
    }
    HAPPYRAY_CHECK(flat.View().positions == flat.positions.data());
    HAPPYRAY_CHECK(flat.View().normals == flat.normals.data());
    HAPPYRAY_CHECK(flat.View().triangle_count == 2);
}

void TestObjLoaderCubeWithMaterialsAndQuads(const std::string& assets_dir) {
    using namespace happyray;
    Scene scene;
    HAPPYRAY_CHECK(LoadObj(assets_dir + "/cube.obj", &scene));
    HAPPYRAY_CHECK(scene.meshes.size() == 1);
    HAPPYRAY_CHECK(scene.materials.size() == 2);
    if (scene.meshes.empty()) {
        return;
    }
    const Mesh& mesh = scene.meshes[0];
    // The 12 original triangles and their children retain their face materials.
    HAPPYRAY_CHECK(mesh.TriangleCount() > 12);
    HAPPYRAY_CHECK(mesh.material_ids.size() == mesh.TriangleCount());
    CheckSubdivisionLimit(scene);
    const uint32_t red_id = (scene.materials[0].name == "red") ? 0 : 1;
    HAPPYRAY_CHECK(mesh.material_ids[0] == red_id);
    HAPPYRAY_CHECK(mesh.material_ids[3] == red_id);
    HAPPYRAY_CHECK(mesh.material_ids[4] != red_id);
    double red_area = 0, blue_area = 0;
    for (size_t tri = 0; tri < mesh.indices.size(); tri += 3) {
        const Vec3 a = mesh.positions[mesh.indices[tri]];
        const Vec3 b = mesh.positions[mesh.indices[tri + 1]];
        const Vec3 c = mesh.positions[mesh.indices[tri + 2]];
        const double area = 0.5 * Length(Cross(b - a, c - a));
        if (mesh.material_ids[tri / 3] == red_id) red_area += area;
        else blue_area += area;
    }
    HAPPYRAY_CHECK(std::fabs(red_area - 2) < 1e-6);
    HAPPYRAY_CHECK(std::fabs(blue_area - 4) < 1e-6);

    // cube.mtl's Kd lines should have been parsed into the matching materials via mtllib.
    const uint32_t blue_id = red_id == 0 ? 1 : 0;
    HAPPYRAY_CHECK(NearlyEqual(scene.materials[red_id].diffuse_color, Vec3(0.8f, 0.1f, 0.1f)));
    HAPPYRAY_CHECK(NearlyEqual(scene.materials[blue_id].diffuse_color, Vec3(0.1f, 0.1f, 0.8f)));

    const BBox bounds = scene.WorldBounds();
    HAPPYRAY_CHECK(NearlyEqual(bounds.Extent(), Vec3(1.0f, 1.0f, 1.0f)));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string assets_dir = (argc > 1) ? argv[1] : "assets";

    TestVecMath();
    TestMat4();
    TestBBox();
    TestBBoxExtractor();
    TestQuaternion();
    TestHalton();
    TestTriangleSubdivision();
    TestDefaultMaxDistance();
    TestFlattenScene();
    TestObjLoaderTriangleNoNormals(assets_dir);
    TestObjLoaderAttributeSeam(assets_dir);
    TestObjLoaderCubeWithMaterialsAndQuads(assets_dir);

    if (g_failures == 0) {
        std::printf("All core tests passed.\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d core test check(s) failed.\n", g_failures);
    return EXIT_FAILURE;
}
