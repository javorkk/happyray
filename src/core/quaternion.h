// Host/device quaternion math, used by Camera for gimbal/roll-safe rotation and (later) for
// keyframed mesh/camera animation, where Slerp is the natural way to interpolate orientations.
#pragma once

#include "mat4.h"
#include "vec.h"

namespace happyray {

struct Quaternion {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;

    HAPPYRAY_HD Quaternion() = default;
    HAPPYRAY_HD Quaternion(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    HAPPYRAY_HD static Quaternion Identity() { return Quaternion(0.0f, 0.0f, 0.0f, 1.0f); }

    // `axis` need not be normalized; `angle_rad` is a right-handed rotation around it.
    HAPPYRAY_HD static Quaternion FromAxisAngle(const Vec3& axis, float angle_rad) {
        const Vec3 n = Normalize(axis);
        const float half_angle = angle_rad * 0.5f;
        const float s = std::sin(half_angle);
        return Quaternion(n.x * s, n.y * s, n.z * s, std::cos(half_angle));
    }
};

// Hamilton product: composes rotations so that RotateVector(a * b, v) == RotateVector(a,
// RotateVector(b, v)) (i.e. `b` is applied first, then `a`).
HAPPYRAY_HD inline Quaternion operator*(const Quaternion& a, const Quaternion& b) {
    return Quaternion(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                       a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                       a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                       a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

HAPPYRAY_HD inline float Dot(const Quaternion& a, const Quaternion& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
HAPPYRAY_HD inline float Length(const Quaternion& q) { return std::sqrt(Dot(q, q)); }

HAPPYRAY_HD inline Quaternion Normalize(const Quaternion& q) {
    const float len = Length(q);
    return len > 0.0f ? Quaternion(q.x / len, q.y / len, q.z / len, q.w / len)
                       : Quaternion::Identity();
}

HAPPYRAY_HD inline Quaternion Conjugate(const Quaternion& q) {
    return Quaternion(-q.x, -q.y, -q.z, q.w);
}

// Rotates `v` by unit quaternion `q` (v' = q * v * q^-1), using Fabian Giesen's optimized form
// that avoids constructing a pure-vector quaternion intermediate.
HAPPYRAY_HD inline Vec3 RotateVector(const Quaternion& q, const Vec3& v) {
    const Vec3 qv(q.x, q.y, q.z);
    const Vec3 t = Cross(qv, v) * 2.0f;
    return v + t * q.w + Cross(qv, t);
}

// Spherical linear interpolation between two orientations, t in [0, 1]. Used for camera/mesh
// keyframe animation.
HAPPYRAY_HD inline Quaternion Slerp(const Quaternion& a, const Quaternion& b, float t) {
    Quaternion end = b;
    float cos_half_theta = Dot(a, b);
    if (cos_half_theta < 0.0f) {
        end = Quaternion(-b.x, -b.y, -b.z, -b.w);
        cos_half_theta = -cos_half_theta;
    }
    // Nearly parallel: fall back to a normalized linear blend to avoid dividing by ~0.
    if (cos_half_theta > 0.9995f) {
        return Normalize(Quaternion(a.x + (end.x - a.x) * t, a.y + (end.y - a.y) * t,
                                     a.z + (end.z - a.z) * t, a.w + (end.w - a.w) * t));
    }
    const float half_theta = std::acos(cos_half_theta);
    const float sin_half_theta = std::sin(half_theta);
    const float weight_a = std::sin((1.0f - t) * half_theta) / sin_half_theta;
    const float weight_b = std::sin(t * half_theta) / sin_half_theta;
    return Quaternion(a.x * weight_a + end.x * weight_b, a.y * weight_a + end.y * weight_b,
                       a.z * weight_a + end.z * weight_b, a.w * weight_a + end.w * weight_b);
}

// Embeds the rotation as the linear part of an affine Mat4 (translation left at zero).
HAPPYRAY_HD inline Mat4 ToMat4(const Quaternion& q) {
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    Mat4 result;
    result.m[0][0] = 1.0f - 2.0f * (yy + zz);
    result.m[0][1] = 2.0f * (xy - wz);
    result.m[0][2] = 2.0f * (xz + wy);

    result.m[1][0] = 2.0f * (xy + wz);
    result.m[1][1] = 1.0f - 2.0f * (xx + zz);
    result.m[1][2] = 2.0f * (yz - wx);

    result.m[2][0] = 2.0f * (xz - wy);
    result.m[2][1] = 2.0f * (yz + wx);
    result.m[2][2] = 1.0f - 2.0f * (xx + yy);
    return result;
}

}  // namespace happyray
