// In-memory scene representation produced by the OBJ loader (and, later, other importers like
// USD). Positions and vertex attributes live in separate, equally sized buffers sharing
// the same triangle indices.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bbox.h"
#include "vec.h"

namespace happyray {

struct VertexAttributes {
    Vec3 normal;
    Vec2 uv;
};

struct Material {
    std::string name;
    // Flat Lambertian diffuse color, parsed from the OBJ's mtllib ("Kd r g b"). Defaults to
    // white (full albedo) when no material/Kd is available, so AO shading multiplied by this
    // color is a no-op for un-textured/material-less geometry.
    Vec3 diffuse_color = Vec3(1.0f, 1.0f, 1.0f);
};

struct Mesh {
    std::string name;
    std::vector<Vec3> positions;
    std::vector<VertexAttributes> attributes;  // One entry per position, including at seams.
    std::vector<uint32_t> indices;       // 3 per triangle, indexing positions and attributes.
    std::vector<uint32_t> material_ids;  // One entry per triangle, indexing into Scene::materials.

    uint32_t TriangleCount() const { return static_cast<uint32_t>(indices.size() / 3); }
};

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;

    BBox WorldBounds() const {
        BBox bounds;
        for (const Mesh& mesh : meshes) {
            for (const Vec3& position : mesh.positions) {
                bounds.Grow(position);
            }
        }
        return bounds;
    }

    uint32_t TriangleCount() const {
        uint32_t total = 0;
        for (const Mesh& mesh : meshes) {
            total += mesh.TriangleCount();
        }
        return total;
    }
};

}  // namespace happyray
