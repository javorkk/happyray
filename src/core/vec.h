// Host/device vector math shared by the CPU and CUDA rendering back ends. Every function is
// a small inline so the same header works whether it's pulled into a plain .cpp translation
// unit (host-only) or a .cu translation unit compiled by nvcc (host + device).
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define HAPPYRAY_HD __host__ __device__
#else
#define HAPPYRAY_HD
#endif

namespace happyray {

HAPPYRAY_HD inline int ClampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    HAPPYRAY_HD Vec2() = default;
    HAPPYRAY_HD Vec2(float x_, float y_) : x(x_), y(y_) {}
};

HAPPYRAY_HD inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x + b.x, a.y + b.y); }
HAPPYRAY_HD inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x - b.x, a.y - b.y); }
HAPPYRAY_HD inline Vec2 operator*(const Vec2& a, float s) { return Vec2(a.x * s, a.y * s); }
HAPPYRAY_HD inline Vec2 operator/(const Vec2& a, float s) { return a * (1.0f / s); }
HAPPYRAY_HD inline Vec2& operator+=(Vec2& a, const Vec2& b) { a = a + b; return a; }

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    HAPPYRAY_HD Vec3() = default;
    HAPPYRAY_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    HAPPYRAY_HD explicit Vec3(float s) : x(s), y(s), z(s) {}
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    HAPPYRAY_HD Vec4() = default;
    HAPPYRAY_HD Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    HAPPYRAY_HD Vec4(const Vec3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

// --- Vec3 arithmetic ---

HAPPYRAY_HD inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
HAPPYRAY_HD inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
HAPPYRAY_HD inline Vec3 operator-(const Vec3& a) { return Vec3(-a.x, -a.y, -a.z); }
HAPPYRAY_HD inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
HAPPYRAY_HD inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
HAPPYRAY_HD inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
HAPPYRAY_HD inline Vec3 operator/(const Vec3& a, float s) { return a * (1.0f / s); }
HAPPYRAY_HD inline Vec3& operator+=(Vec3& a, const Vec3& b) { a = a + b; return a; }
HAPPYRAY_HD inline Vec3& operator-=(Vec3& a, const Vec3& b) { a = a - b; return a; }
HAPPYRAY_HD inline Vec3& operator*=(Vec3& a, float s) { a = a * s; return a; }

HAPPYRAY_HD inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
HAPPYRAY_HD inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HAPPYRAY_HD inline float LengthSquared(const Vec3& a) { return Dot(a, a); }
HAPPYRAY_HD inline float Length(const Vec3& a) { return std::sqrt(LengthSquared(a)); }
HAPPYRAY_HD inline Vec3 Normalize(const Vec3& a) {
    const float len = Length(a);
    return len > 0.0f ? a / len : Vec3(0.0f, 0.0f, 0.0f);
}
HAPPYRAY_HD inline Vec3 Min(const Vec3& a, const Vec3& b) {
    return Vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}
HAPPYRAY_HD inline Vec3 Max(const Vec3& a, const Vec3& b) {
    return Vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}
HAPPYRAY_HD inline Vec3 Lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }
HAPPYRAY_HD inline Vec3 Reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0f * Dot(incident, normal));
}

}  // namespace happyray
