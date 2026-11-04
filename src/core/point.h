#pragma once

#include "bbox.h"
#include "primitive.h"
#include "ray.h"

namespace happyray {

struct Point : Primitive<1> {
    HAPPYRAY_HD Point() = default;
    HAPPYRAY_HD explicit Point(const Vec3& position) { vtx[0] = position; }
};

constexpr float kPointRayTolerance = 1e-4f;

HAPPYRAY_HD inline bool IntersectRayPoint(const Ray& ray, const Point& point, float* out_t,
                                           float tolerance = kPointRayTolerance) {
    const float direction_length_squared = LengthSquared(ray.direction);
    if (direction_length_squared == 0.0f || tolerance < 0.0f) return false;
    const float t = Dot(point.vtx[0] - ray.origin, ray.direction) / direction_length_squared;
    if (t < ray.t_min || t > ray.t_max) return false;
    const Vec3 delta = ray.At(t) - point.vtx[0];
    if (LengthSquared(delta) > tolerance * tolerance) return false;
    *out_t = t;
    return true;
}

HAPPYRAY_HD inline bool Intersects(const Point& point, const BBox& box) {
    return point.vtx[0].x >= box.min_bound.x && point.vtx[0].x <= box.max_bound.x &&
           point.vtx[0].y >= box.min_bound.y && point.vtx[0].y <= box.max_bound.y &&
           point.vtx[0].z >= box.min_bound.z && point.vtx[0].z <= box.max_bound.z;
}

}  // namespace happyray