// Scene views, host-owned flattened geometry, subdivision, and camera/AO helpers.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../render/camera.h"
#include "mesh.h"
#include "primitive.h"
#include "vec.h"

namespace happyray {

// Non-owning triangle-soup view over either host or device memory.
struct SceneView {
    const Vec3* positions = nullptr;
    const Vec3* normals = nullptr;
    const uint32_t* indices = nullptr;       // 3 per triangle, indexing positions/normals.
    const uint32_t* material_ids = nullptr;  // 1 per triangle.
    // FlattenScene supplies a default-white entry for scenes with no materials.
    const Vec3* material_diffuse_colors = nullptr;
    uint32_t triangle_count = 0;
};

HAPPYRAY_HD inline Triangle FetchTriangle(const SceneView& scene, uint32_t tri) {
    Triangle t;
    t.vtx[0] = scene.positions[scene.indices[tri * 3 + 0]];
    t.vtx[1] = scene.positions[scene.indices[tri * 3 + 1]];
    t.vtx[2] = scene.positions[scene.indices[tri * 3 + 2]];
    return t;
}

// Host-owned geometry shared by CPU rendering and GPU upload.
struct FlatScene {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> material_ids;
    std::vector<Vec3> material_diffuse_colors;

    SceneView View() const {
        SceneView view;
        view.positions = positions.data();
        view.normals = normals.data();
        view.indices = indices.data();
        view.material_ids = material_ids.data();
        view.material_diffuse_colors = material_diffuse_colors.data();
        view.triangle_count = static_cast<uint32_t>(indices.size() / 3);
        return view;
    }
};

// Concatenates meshes, offsetting indices by the running vertex count.
inline FlatScene FlattenScene(const Scene& scene) {
    FlatScene flat;
    uint32_t vertex_offset = 0;
    for (const Mesh& mesh : scene.meshes) {
        flat.positions.insert(flat.positions.end(), mesh.positions.begin(), mesh.positions.end());
        for (const VertexAttributes& attributes : mesh.attributes) {
            flat.normals.push_back(attributes.normal);
        }
        for (uint32_t index : mesh.indices) {
            flat.indices.push_back(index + vertex_offset);
        }
        for (uint32_t material_id : mesh.material_ids) {
            flat.material_ids.push_back(material_id);
        }
        vertex_offset += static_cast<uint32_t>(mesh.positions.size());
    }

    for (const Material& material : scene.materials) {
        flat.material_diffuse_colors.push_back(material.diffuse_color);
    }
    if (flat.material_diffuse_colors.empty()) {
        // Material-less OBJ faces still reference material id 0.
        flat.material_diffuse_colors.push_back(Vec3(1.0f, 1.0f, 1.0f));
    }
    return flat;
}

inline double SceneEdgeLengthSquared(const Vec3& a, const Vec3& b) {
    const double x = static_cast<double>(a.x) - b.x;
    const double y = static_cast<double>(a.y) - b.y;
    const double z = static_cast<double>(a.z) - b.z;
    return x * x + y * y + z * z;
}

// Keep the original scene-wide limit fixed throughout longest-edge bisection.
inline void SubdivideLargeTriangles(Scene* scene) {
    for (const Mesh& mesh : scene->meshes) {
        for (const Vec3& p : mesh.positions) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                throw std::runtime_error("Triangle subdivision requires finite vertex positions");
        }
    }
    const BBox bounds = scene->WorldBounds();
    if (bounds.IsEmpty()) return;
    const double limit_squared =
        0.01 * SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
    if (!std::isfinite(limit_squared))
        throw std::runtime_error("Triangle subdivision requires finite scene bounds");
    if (limit_squared == 0) return;

    for (Mesh& mesh : scene->meshes) {
        for (size_t tri = 0; tri < mesh.indices.size() / 3;) {
            const uint32_t indices[3] = {mesh.indices[tri * 3], mesh.indices[tri * 3 + 1],
                                         mesh.indices[tri * 3 + 2]};
            int edge = 0;
            double longest = 0;
            for (int i = 0; i < 3; ++i) {
                const double length = SceneEdgeLengthSquared(
                    mesh.positions[indices[i]], mesh.positions[indices[(i + 1) % 3]]);
                if (length > longest) {
                    longest = length;
                    edge = i;
                }
            }
            if (longest <= limit_squared) {
                ++tri;
                continue;
            }

            const uint32_t a = indices[edge], b = indices[(edge + 1) % 3];
            const uint32_t c = indices[(edge + 2) % 3];
            const Vec3 midpoint_position = mesh.positions[a] * 0.5f + mesh.positions[b] * 0.5f;
            const VertexAttributes& va = mesh.attributes[a];
            const VertexAttributes& vb = mesh.attributes[b];
            VertexAttributes midpoint;
            midpoint.normal = va.normal * 0.5f + vb.normal * 0.5f;
            midpoint.uv = va.uv * 0.5f + vb.uv * 0.5f;
            if (SceneEdgeLengthSquared(midpoint_position, mesh.positions[a]) == 0 ||
                SceneEdgeLengthSquared(midpoint_position, mesh.positions[b]) == 0)
                throw std::runtime_error("Triangle subdivision cannot represent an edge midpoint");
            if (mesh.positions.size() >= std::numeric_limits<uint32_t>::max() ||
                mesh.indices.size() / 3 >= std::numeric_limits<uint32_t>::max())
                throw std::overflow_error("Triangle subdivision exceeds 32-bit mesh limits");
            const uint32_t mid = static_cast<uint32_t>(mesh.positions.size());
            mesh.positions.push_back(midpoint_position);
            mesh.attributes.push_back(midpoint);
            mesh.indices[tri * 3] = a;
            mesh.indices[tri * 3 + 1] = mid;
            mesh.indices[tri * 3 + 2] = c;
            mesh.indices.insert(mesh.indices.end(), {mid, b, c});
            if (!mesh.material_ids.empty()) {
                const uint32_t material = mesh.material_ids[tri];
                mesh.material_ids.push_back(material);
            }
        }
    }
}

// Frames the camera on the scene's bounding box so freshly-loaded OBJ files are visible
// without manual positioning.
inline Camera MakeDefaultCamera(const Scene& scene) {
    Camera camera;
    const BBox bounds = scene.WorldBounds();
    if (bounds.IsEmpty()) {
        return camera;
    }
    const Vec3 center = bounds.Center();
    const Vec3 extent = bounds.Extent();
    const float radius = 0.5f * Length(extent);
    camera.position = center + Vec3(0.0f, radius * 0.4f, radius * 2.2f + 1e-3f);
    camera.yaw_deg = -90.0f;
    camera.pitch_deg = -10.0f;
    return camera;
}

// The default ambient-occlusion occlusion-ray reach for a scene of a given size: 0.1x the
// longest side of its world-space bounding box. See AOIntegrator::max_distance
// (src/render/ao_integrator.h) for where this gets applied.
inline float DefaultMaxDistance(const Scene& scene) {
    const BBox bounds = scene.WorldBounds();
    if (bounds.IsEmpty()) {
        return 1.0f;
    }
    const Vec3 extent = bounds.Extent();
    const float longest_side = std::fmax(extent.x, std::fmax(extent.y, extent.z));
    return longest_side * 0.1f;
}

// A scene.obj's optional sibling camera config, e.g. "assets/ogre.obj" ->
// "assets/ogre.camera.cfg". Doesn't check the file exists -- LoadCameraConfig does that.
inline std::string CameraConfigPathFor(const std::string& scene_path) {
    const std::string suffix = ".obj";
    if (scene_path.size() >= suffix.size() &&
        scene_path.compare(scene_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return scene_path.substr(0, scene_path.size() - suffix.size()) + ".camera.cfg";
    }
    return scene_path + ".camera.cfg";
}

// Loads a happyray-style camera.cfg ("position x y z", "orientation x y z" (the forward/view
// direction, not necessarily normalized), "up x y z", "FOV deg", "resX"/"resY" -- "rotation" is
// accepted but ignored, since HappyRay has no initial-rotation-then-reset step). Returns false
// (leaving `camera` untouched) if the file doesn't exist.
//
// HappyRay's Camera only supports yaw/pitch (no roll -- its "up" is always derived from yaw/pitch
// applied to world-+Y, see camera.h), so only the forward direction's yaw/pitch is reproduced
// exactly; an "up" vector that isn't already close to world +Y (i.e. a camera.cfg with
// intentional roll/bank) can't be represented and is otherwise ignored.
//
// If `out_width`/`out_height` are non-null and the file has a "resX"/"resY" line, they're set to
// those values (left untouched otherwise); the caller decides how to combine that with any other
// resolution source (e.g. a command-line override should still win). Pass nullptr (the default)
// to ignore resX/resY entirely, as benchmark.cpp does (its resolution comes from --width/--height
// instead).
inline bool LoadCameraConfig(const std::string& path, Camera* camera, int* out_width = nullptr,
                              int* out_height = nullptr) {
    std::FILE* file = std::fopen(path.c_str(), "r");
    if (!file) {
        return false;
    }

    bool have_orientation = false;
    Vec3 orientation(1.0f, 0.0f, 0.0f);
    char line_buffer[256];
    while (std::fgets(line_buffer, sizeof(line_buffer), file)) {
        std::istringstream line(line_buffer);
        std::string keyword;
        line >> keyword;
        if (keyword == "position") {
            line >> camera->position.x >> camera->position.y >> camera->position.z;
        } else if (keyword == "orientation") {
            line >> orientation.x >> orientation.y >> orientation.z;
            have_orientation = true;
        } else if (keyword == "FOV" || keyword == "fov") {
            line >> camera->vertical_fov_deg;
        } else if (keyword == "resX" && out_width) {
            line >> *out_width;
        } else if (keyword == "resY" && out_height) {
            line >> *out_height;
        }
        // "up"/"rotation" (and anything else) are intentionally ignored -- see this function's
        // header comment.
    }
    std::fclose(file);

    if (have_orientation) {
        const Vec3 forward = Normalize(orientation);
        constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
        camera->pitch_deg = std::asin(forward.y) * kRadToDeg;
        camera->yaw_deg = std::atan2(forward.z, forward.x) * kRadToDeg;
    }
    return true;
}

}  // namespace happyray
