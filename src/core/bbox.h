// Axis-aligned bounding box, used both as a general geometric utility and as the building
// block for every acceleration structure (uniform grid, two-level grid, irregular grid).
// BBoxExtractor computes bounds for any Primitive<N> (see primitive.h).
#pragma once

#include "primitive.h"
#include "ray.h"
#include "vec.h"

namespace happyray {

struct BBox {
    Vec3 min_bound = Vec3(1e30f, 1e30f, 1e30f);
    Vec3 max_bound = Vec3(-1e30f, -1e30f, -1e30f);

    HAPPYRAY_HD BBox() = default;
    HAPPYRAY_HD BBox(const Vec3& min_b, const Vec3& max_b) : min_bound(min_b), max_bound(max_b) {}

    HAPPYRAY_HD bool IsEmpty() const {
        return min_bound.x > max_bound.x || min_bound.y > max_bound.y ||
               min_bound.z > max_bound.z;
    }

    HAPPYRAY_HD Vec3 Extent() const { return max_bound - min_bound; }
    HAPPYRAY_HD Vec3 Center() const { return (min_bound + max_bound) * 0.5f; }

    HAPPYRAY_HD void Grow(const Vec3& p) {
        min_bound = Min(min_bound, p);
        max_bound = Max(max_bound, p);
    }

    HAPPYRAY_HD void Grow(const BBox& other) {
        min_bound = Min(min_bound, other.min_bound);
        max_bound = Max(max_bound, other.max_bound);
    }
};

template <typename TPrimitive>
struct BBoxExtractor {
    HAPPYRAY_HD static BBox Get(const TPrimitive& primitive) {
        BBox box;
        for (int i = 0; i < TPrimitive::kNumVertices; ++i) {
            box.Grow(primitive.vtx[i]);
        }
        return box;
    }
};

// Slab-method ray/box intersection. On a hit, writes the entry/exit distances along the ray
// (clipped to the ray's own [t_min, t_max]) to `out_t_near` / `out_t_far`.
HAPPYRAY_HD inline bool IntersectRayBBox(const Ray& ray, const BBox& box, float* out_t_near,
                                     float* out_t_far) {
    float t_near = ray.t_min;
    float t_far = ray.t_max;

    const float* ray_origin = &ray.origin.x;
    const float* ray_dir = &ray.direction.x;
    const float* box_min = &box.min_bound.x;
    const float* box_max = &box.max_bound.x;

    for (int axis = 0; axis < 3; ++axis) {
        if (ray_dir[axis] == 0.0f) {
            if (ray_origin[axis] < box_min[axis] || ray_origin[axis] > box_max[axis]) {
                return false;
            }
            continue;
        }
        const float inv_dir = 1.0f / ray_dir[axis];
        float t0 = (box_min[axis] - ray_origin[axis]) * inv_dir;
        float t1 = (box_max[axis] - ray_origin[axis]) * inv_dir;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_near = t0 > t_near ? t0 : t_near;
        t_far = t1 < t_far ? t1 : t_far;
        if (t_near > t_far) {
            return false;
        }
    }

    *out_t_near = t_near;
    *out_t_far = t_far;
    return true;
}

}  // namespace happyray
