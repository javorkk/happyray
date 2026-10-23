// Ray/triangle intersection primitives, shared by brute-force scene queries (src/accel/
// scene_intersect.h) and every acceleration structure's traversal code (src/accel/*): both need the
// same HitRecord/IntersectRayTriangle, and accel structures must not depend on render (to avoid
// a dependency cycle), so this lives in core.
#pragma once

#include <cstdint>

#include "ray.h"
#include "vec.h"

namespace happyray {

struct HitRecord {
    float t = 1e30f;
    Vec3 position;
    Vec3 normal;
    uint32_t material_id = 0;
};

// Moller-Trumbore ray/triangle intersection.
HAPPYRAY_HD inline bool IntersectRayTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1,
                                         const Vec3& v2, float* out_t, float* out_u,
                                         float* out_v) {
    const Vec3 edge1 = v1 - v0;
    const Vec3 edge2 = v2 - v0;
    const Vec3 pvec = Cross(ray.direction, edge2);
    const float det = Dot(edge1, pvec);
    if (det > -1e-8f && det < 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 tvec = ray.origin - v0;
    const float u = Dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 qvec = Cross(tvec, edge1);
    const float v = Dot(ray.direction, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float t = Dot(edge2, qvec) * inv_det;
    if (t < ray.t_min || t > ray.t_max) {
        return false;
    }
    *out_t = t;
    *out_u = u;
    *out_v = v;
    return true;
}

}  // namespace happyray
