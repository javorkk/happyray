// Triangle geometry shared by brute-force scene queries and acceleration structures.
#pragma once

#include <cstdint>
#include <cmath>

#include "bbox.h"
#include "primitive.h"
#include "ray.h"
#include "vec.h"

namespace happyray {

using Triangle = Primitive<3>;

struct HitRecord {
    float t = 1e30f;
    Vec3f position;
    Vec3f normal;
    uint32_t material_id = 0;
};

// Moller-Trumbore ray/triangle intersection.
HAPPYRAY_HD inline bool IntersectRayTriangle(const Ray& ray, const Vec3f& v0, const Vec3f& v1,
                                         const Vec3f& v2, float* out_t, float* out_u,
                                         float* out_v) {
    const Vec3f edge1 = v1 - v0;
    const Vec3f edge2 = v2 - v0;
    const Vec3f pvec = Cross(ray.direction, edge2);
    const float det = Dot(edge1, pvec);
    if (det > -1e-8f && det < 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3f tvec = ray.origin - v0;
    const float u = Dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3f qvec = Cross(tvec, edge1);
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

struct TriangleBoxIntersection {
    Vec3f a, b, c;
    Vec3f edges[3];
    Vec3f normal;

    HAPPYRAY_HD explicit TriangleBoxIntersection(const Triangle& triangle)
        : a(triangle.vtx[0]), b(triangle.vtx[1]), c(triangle.vtx[2]),
          edges{b - a, c - b, a - c}, normal(Cross(edges[0], c - a)) {}

    HAPPYRAY_HD static bool AxisTest(const Vec3f& axis, const Vec3f& a, const Vec3f& b,
                                     const Vec3f& c, const Vec3f& half) {
        const float p = Dot(axis, a), q = Dot(axis, b), r = Dot(axis, c);
        const float radius = std::fabs(axis.x) * half.x + std::fabs(axis.y) * half.y +
                             std::fabs(axis.z) * half.z;
        return !(std::fmin(p, std::fmin(q, r)) > radius + 1e-5f ||
                 std::fmax(p, std::fmax(q, r)) < -radius - 1e-5f);
    }

    HAPPYRAY_HD bool operator()(const BBox& box) const {
        const Vec3f center = box.Center();
        const Vec3f v0 = a - center, v1 = b - center, v2 = c - center;
        const Vec3f half = box.Extent() * 0.5f;
        if (!AxisTest(Vec3f(1, 0, 0), v0, v1, v2, half) ||
            !AxisTest(Vec3f(0, 1, 0), v0, v1, v2, half) ||
            !AxisTest(Vec3f(0, 0, 1), v0, v1, v2, half) ||
            !AxisTest(normal, v0, v1, v2, half))
            return false;
        const Vec3f opposite[3] = {v2, v0, v1};
        const Vec3f endpoints[3] = {v0, v1, v2};
        for (int i = 0; i < 3; ++i) {
            const Vec3f& edge = edges[i];
            const Vec3f& p = endpoints[i];
            const Vec3f& q = opposite[i];
            if (!AxisTest(Vec3f(0, edge.z, -edge.y), p, q, q, half) ||
                !AxisTest(Vec3f(-edge.z, 0, edge.x), p, q, q, half) ||
                !AxisTest(Vec3f(edge.y, -edge.x, 0), p, q, q, half))
                return false;
        }
        return true;
    }
};

namespace detail {

HAPPYRAY_HD inline int PlaneSide(float distance, float tolerance) {
    return distance > tolerance ? 1 : (distance < -tolerance ? -1 : 0);
}

HAPPYRAY_HD inline int TrianglePlaneSlice(const Vec3f triangle[3], const float distance[3],
                                           float tolerance, Vec3f points[6]) {
    int count = 0;
    for (int i = 0; i < 3; ++i) {
        const int next = (i + 1) % 3;
        const int side = PlaneSide(distance[i], tolerance);
        const int next_side = PlaneSide(distance[next], tolerance);
        if (side == 0) points[count++] = triangle[i];
        if (side * next_side < 0) {
            const float t = distance[i] / (distance[i] - distance[next]);
            points[count++] = Lerp(triangle[i], triangle[next], t);
        }
    }
    return count;
}

HAPPYRAY_HD inline bool CoplanarTrianglesOverlap(const Vec3f a[3], const Vec3f b[3],
                                                 const Vec3f& normal, float tolerance) {
    int drop_axis = 0;
    if (std::fabs(normal.y) > std::fabs(normal.x)) drop_axis = 1;
    if (std::fabs(normal.z) > (drop_axis == 0 ? std::fabs(normal.x) : std::fabs(normal.y)))
        drop_axis = 2;

    for (int triangle = 0; triangle < 2; ++triangle) {
        const Vec3f* vertices = triangle == 0 ? a : b;
        for (int edge_index = 0; edge_index < 3; ++edge_index) {
            const Vec3f& start = vertices[edge_index];
            const Vec3f& end = vertices[(edge_index + 1) % 3];
            const Vec3f edge = end - start;
            Vec3f axis;
            if (drop_axis == 0) axis = Vec3f(0, -edge.z, edge.y);
            else if (drop_axis == 1) axis = Vec3f(-edge.z, 0, edge.x);
            else axis = Vec3f(-edge.y, edge.x, 0);
            if (LengthSquared(axis) == 0.0f) continue;

            float a_min = Dot(axis, a[0]), a_max = a_min;
            float b_min = Dot(axis, b[0]), b_max = b_min;
            for (int i = 1; i < 3; ++i) {
                const float pa = Dot(axis, a[i]);
                const float pb = Dot(axis, b[i]);
                a_min = fminf(a_min, pa);
                a_max = fmaxf(a_max, pa);
                b_min = fminf(b_min, pb);
                b_max = fmaxf(b_max, pb);
            }
            const float axis_tolerance = tolerance * std::sqrt(LengthSquared(axis));
            if (a_max < b_min - axis_tolerance || b_max < a_min - axis_tolerance)
                return false;
        }
    }
    return true;
}

HAPPYRAY_HD inline bool DegenerateTrianglesOverlap(const Vec3f a[3], const Vec3f b[3],
                                                   const Vec3f& normal_a,
                                                   const Vec3f& normal_b,
                                                   float tolerance) {
    Vec3f axes[32];
    int axis_count = 0;
    axes[axis_count++] = normal_a;
    axes[axis_count++] = normal_b;
    for (int i = 0; i < 3; ++i) {
        const Vec3f edge_a = a[(i + 1) % 3] - a[i];
        const Vec3f edge_b = b[(i + 1) % 3] - b[i];
        axes[axis_count++] = Cross(normal_a, edge_a);
        axes[axis_count++] = Cross(normal_b, edge_b);
        for (int j = 0; j < 3; ++j) {
            axes[axis_count++] = Cross(edge_a, b[(j + 1) % 3] - b[j]);
            axes[axis_count++] = a[i] - b[j];
        }
    }
    for (int axis_index = 0; axis_index < axis_count; ++axis_index) {
        const Vec3f& axis = axes[axis_index];
        const float axis_length_squared = LengthSquared(axis);
        if (axis_length_squared <= 1e-20f) continue;
        float a_min = Dot(axis, a[0]), a_max = a_min;
        float b_min = Dot(axis, b[0]), b_max = b_min;
        for (int i = 1; i < 3; ++i) {
            const float pa = Dot(axis, a[i]);
            const float pb = Dot(axis, b[i]);
            a_min = fminf(a_min, pa);
            a_max = fmaxf(a_max, pa);
            b_min = fminf(b_min, pb);
            b_max = fmaxf(b_max, pb);
        }
        const float axis_tolerance = tolerance * std::sqrt(axis_length_squared);
        if (a_max < b_min - axis_tolerance || b_max < a_min - axis_tolerance) return false;
    }
    return true;
}

}  // namespace detail

HAPPYRAY_HD inline bool Intersects(const Triangle& triangle, const BBox& box) {
    return TriangleBoxIntersection(triangle)(box);
}

// Moller-style triangle/triangle test; tolerance is a world-space distance.
HAPPYRAY_HD inline bool IntersectTriangles(const Triangle& triangle_a,
                                           const Triangle& triangle_b,
                                           float tolerance = 1e-6f) {
    if (tolerance < 0.0f) return false;
    const Vec3f a[3] = {triangle_a.vtx[0], triangle_a.vtx[1], triangle_a.vtx[2]};
    const Vec3f b[3] = {triangle_b.vtx[0], triangle_b.vtx[1], triangle_b.vtx[2]};
    const Vec3f normal_a = Cross(a[1] - a[0], a[2] - a[0]);
    const Vec3f normal_b = Cross(b[1] - b[0], b[2] - b[0]);
    const float normal_a_length = std::sqrt(LengthSquared(normal_a));
    const float normal_b_length = std::sqrt(LengthSquared(normal_b));
    if (normal_a_length <= 1e-12f || normal_b_length <= 1e-12f)
        return detail::DegenerateTrianglesOverlap(a, b, normal_a, normal_b, tolerance);

    float distance_b[3], distance_a[3];
    int side_b[3], side_a[3];
    bool b_coplanar = true, a_coplanar = true;
    for (int i = 0; i < 3; ++i) {
        distance_b[i] = Dot(normal_a, b[i] - a[0]) / normal_a_length;
        distance_a[i] = Dot(normal_b, a[i] - b[0]) / normal_b_length;
        side_b[i] = detail::PlaneSide(distance_b[i], tolerance);
        side_a[i] = detail::PlaneSide(distance_a[i], tolerance);
        b_coplanar = b_coplanar && side_b[i] == 0;
        a_coplanar = a_coplanar && side_a[i] == 0;
    }
    if ((side_b[0] > 0 && side_b[1] > 0 && side_b[2] > 0) ||
        (side_b[0] < 0 && side_b[1] < 0 && side_b[2] < 0) ||
        (side_a[0] > 0 && side_a[1] > 0 && side_a[2] > 0) ||
        (side_a[0] < 0 && side_a[1] < 0 && side_a[2] < 0))
        return false;
    if (b_coplanar && a_coplanar)
        return detail::CoplanarTrianglesOverlap(a, b, normal_a, tolerance);

    Vec3f slice_a[6], slice_b[6];
    const int count_a = detail::TrianglePlaneSlice(a, distance_a, tolerance, slice_a);
    const int count_b = detail::TrianglePlaneSlice(b, distance_b, tolerance, slice_b);
    if (count_a == 0 || count_b == 0) return false;

    const Vec3f line = Cross(normal_a, normal_b);
    int projection_axis = 0;
    if (std::fabs(line.y) > std::fabs(line.x)) projection_axis = 1;
    if (std::fabs(line.z) > (projection_axis == 0 ? std::fabs(line.x) : std::fabs(line.y)))
        projection_axis = 2;
    float a_min = projection_axis == 0 ? slice_a[0].x
                  : projection_axis == 1 ? slice_a[0].y : slice_a[0].z;
    float a_max = a_min;
    float b_min = projection_axis == 0 ? slice_b[0].x
                  : projection_axis == 1 ? slice_b[0].y : slice_b[0].z;
    float b_max = b_min;
    for (int i = 1; i < count_a; ++i) {
        const float value = projection_axis == 0 ? slice_a[i].x
                            : projection_axis == 1 ? slice_a[i].y : slice_a[i].z;
        a_min = fminf(a_min, value);
        a_max = fmaxf(a_max, value);
    }
    for (int i = 1; i < count_b; ++i) {
        const float value = projection_axis == 0 ? slice_b[i].x
                            : projection_axis == 1 ? slice_b[i].y : slice_b[i].z;
        b_min = fminf(b_min, value);
        b_max = fmaxf(b_max, value);
    }
    return a_min <= b_max + tolerance && b_min <= a_max + tolerance;
}

}  // namespace happyray