// Shared host/device scene intersection and occlusion queries, independent of the integrator.
// Brute force is the correctness oracle for every acceleration structure.
#pragma once

#include "primitive.h"
#include "ray.h"
#include "scene.h"
#include "triangle_intersect.h"
#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"
#include "vec.h"

namespace happyray {

// Finds the closest hit across every triangle in `scene`. Returns false if the ray hits
// nothing.
HAPPYRAY_HD inline bool IntersectSceneBruteForce(const SceneView& scene, Ray ray, HitRecord* out_hit) {
    bool hit_anything = false;
    for (uint32_t tri = 0; tri < scene.triangle_count; ++tri) {
        const Triangle triangle = FetchTriangle(scene, tri);
        const uint32_t i0 = scene.indices[tri * 3 + 0];
        const uint32_t i1 = scene.indices[tri * 3 + 1];
        const uint32_t i2 = scene.indices[tri * 3 + 2];
        float t, u, v;
        if (IntersectRayTriangle(ray, triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], &t, &u,
                                 &v)) {
            ray.t_max = t;
            out_hit->t = t;
            out_hit->position = ray.At(t);
            // Barycentric-interpolated shading normal.
            out_hit->normal = Normalize(scene.normals[i0] * (1.0f - u - v) +
                                        scene.normals[i1] * u + scene.normals[i2] * v);
            out_hit->material_id = scene.material_ids[tri];
            hit_anything = true;
        }
    }
    return hit_anything;
}

// Returns true if any triangle occludes `ray` (no need for the closest hit -- used for AO
// occlusion tests, where only a boolean "something is in the way" result is needed).
HAPPYRAY_HD inline bool IsOccludedBruteForce(const SceneView& scene, const Ray& ray) {
    for (uint32_t tri = 0; tri < scene.triangle_count; ++tri) {
        const Triangle triangle = FetchTriangle(scene, tri);
        float t, u, v;
        if (IntersectRayTriangle(ray, triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], &t, &u,
                                 &v)) {
            return true;
        }
    }
    return false;
}

// Selects which acceleration structure (if any) IntersectScene/IsOccluded
// trace against. Each later phase adds one more AccelType enum value and one more (optional)
// *View member here -- the dispatch in IntersectScene/IsOccluded stays a simple switch, same
// cost on CPU and GPU.
enum class AccelType { kBruteForce, kUniformGrid, kTwoLevelGrid, kIrregularGrid };

struct AccelStructures {
    AccelType type = AccelType::kBruteForce;
    UniformGridView uniform_grid;      // Valid/meaningful only when type == kUniformGrid.
    TwoLevelGridView two_level_grid;   // Valid/meaningful only when type == kTwoLevelGrid.
    IrregularGridView irregular_grid;  // Valid/meaningful only when type == kIrregularGrid.
};

HAPPYRAY_HD inline bool IntersectScene(const SceneView& scene, const AccelStructures& accel, Ray ray,
                                   HitRecord* out_hit) {
    switch (accel.type) {
        case AccelType::kUniformGrid:
            return IntersectSceneGrid(scene, accel.uniform_grid, ray, out_hit);
        case AccelType::kTwoLevelGrid:
            return IntersectSceneTwoLevelGrid(scene, accel.two_level_grid, ray, out_hit);
        case AccelType::kIrregularGrid:
            return IntersectSceneIrregularGrid(scene, accel.irregular_grid, ray, out_hit);
        case AccelType::kBruteForce:
        default:
            return IntersectSceneBruteForce(scene, ray, out_hit);
    }
}

HAPPYRAY_HD inline bool IsOccluded(const SceneView& scene, const AccelStructures& accel,
                               const Ray& ray) {
    switch (accel.type) {
        case AccelType::kUniformGrid:
            return IsOccludedGrid(scene, accel.uniform_grid, ray);
        case AccelType::kTwoLevelGrid:
            return IsOccludedTwoLevelGrid(scene, accel.two_level_grid, ray);
        case AccelType::kIrregularGrid:
            return IsOccludedIrregularGrid(scene, accel.irregular_grid, ray);
        case AccelType::kBruteForce:
        default:
            return IsOccludedBruteForce(scene, ray);
    }
}

}  // namespace happyray
