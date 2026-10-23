// A 4x4 affine-capable matrix (row-major storage) used for camera transforms and, later, mesh
// instance transforms. Inverse() assumes the common affine case (bottom row [0,0,0,1]), which
// covers every matrix this project constructs (translation/rotation/scale/look-at).
#pragma once

#include "vec.h"

namespace happyray {

struct Mat4 {
    // m[row][col]
    float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

    HAPPYRAY_HD Mat4() = default;

    HAPPYRAY_HD static Mat4 Identity() { return Mat4(); }

    HAPPYRAY_HD static Mat4 Translation(const Vec3& t) {
        Mat4 result;
        result.m[0][3] = t.x;
        result.m[1][3] = t.y;
        result.m[2][3] = t.z;
        return result;
    }

    HAPPYRAY_HD static Mat4 Scale(const Vec3& s) {
        Mat4 result;
        result.m[0][0] = s.x;
        result.m[1][1] = s.y;
        result.m[2][2] = s.z;
        return result;
    }

    // Right-handed look-at: builds the world-to-camera (view) matrix for a camera at `eye`
    // looking towards `target` with the given `up` hint.
    HAPPYRAY_HD static Mat4 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        const Vec3 forward = Normalize(target - eye);
        const Vec3 right = Normalize(Cross(forward, up));
        const Vec3 camera_up = Cross(right, forward);

        Mat4 result;
        result.m[0][0] = right.x;
        result.m[0][1] = right.y;
        result.m[0][2] = right.z;
        result.m[0][3] = -Dot(right, eye);

        result.m[1][0] = camera_up.x;
        result.m[1][1] = camera_up.y;
        result.m[1][2] = camera_up.z;
        result.m[1][3] = -Dot(camera_up, eye);

        result.m[2][0] = -forward.x;
        result.m[2][1] = -forward.y;
        result.m[2][2] = -forward.z;
        result.m[2][3] = Dot(forward, eye);

        result.m[3][0] = 0.0f;
        result.m[3][1] = 0.0f;
        result.m[3][2] = 0.0f;
        result.m[3][3] = 1.0f;
        return result;
    }
};

HAPPYRAY_HD inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[row][k] * b.m[k][col];
            }
            result.m[row][col] = sum;
        }
    }
    return result;
}

// Transforms a point (implicit w = 1); the result's w is discarded (assumes an affine matrix).
HAPPYRAY_HD inline Vec3 TransformPoint(const Mat4& m, const Vec3& p) {
    return Vec3(m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3],
                m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3],
                m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3]);
}

// Transforms a direction (implicit w = 0); unaffected by translation.
HAPPYRAY_HD inline Vec3 TransformVector(const Mat4& m, const Vec3& v) {
    return Vec3(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
                m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
                m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z);
}

// Inverse of an affine matrix (bottom row assumed to be [0,0,0,1]): inverts the 3x3 linear part
// and recomputes the translation as -R^-1 * t.
HAPPYRAY_HD inline Mat4 Inverse(const Mat4& a) {
    const float a00 = a.m[0][0], a01 = a.m[0][1], a02 = a.m[0][2];
    const float a10 = a.m[1][0], a11 = a.m[1][1], a12 = a.m[1][2];
    const float a20 = a.m[2][0], a21 = a.m[2][1], a22 = a.m[2][2];

    const float cofactor00 = a11 * a22 - a12 * a21;
    const float cofactor01 = a12 * a20 - a10 * a22;
    const float cofactor02 = a10 * a21 - a11 * a20;

    const float det = a00 * cofactor00 + a01 * cofactor01 + a02 * cofactor02;
    const float inv_det = (det != 0.0f) ? (1.0f / det) : 0.0f;

    Mat4 result;
    result.m[0][0] = cofactor00 * inv_det;
    result.m[1][0] = cofactor01 * inv_det;
    result.m[2][0] = cofactor02 * inv_det;

    result.m[0][1] = (a02 * a21 - a01 * a22) * inv_det;
    result.m[1][1] = (a00 * a22 - a02 * a20) * inv_det;
    result.m[2][1] = (a01 * a20 - a00 * a21) * inv_det;

    result.m[0][2] = (a01 * a12 - a02 * a11) * inv_det;
    result.m[1][2] = (a02 * a10 - a00 * a12) * inv_det;
    result.m[2][2] = (a00 * a11 - a01 * a10) * inv_det;

    const Vec3 t(a.m[0][3], a.m[1][3], a.m[2][3]);
    const Vec3 inv_t = TransformVector(result, -t);
    result.m[0][3] = inv_t.x;
    result.m[1][3] = inv_t.y;
    result.m[2][3] = inv_t.z;

    result.m[3][0] = 0.0f;
    result.m[3][1] = 0.0f;
    result.m[3][2] = 0.0f;
    result.m[3][3] = 1.0f;
    return result;
}

}  // namespace happyray
