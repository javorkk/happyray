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

template <typename T> struct Vec2 {
    T x = T{};
    T y = T{};

    HAPPYRAY_HD Vec2() = default;
    HAPPYRAY_HD Vec2(T x_, T y_) : x(x_), y(y_) {}
};

template <typename T> HAPPYRAY_HD inline Vec2<T> operator+(const Vec2<T>& a, const Vec2<T>& b) { return Vec2<T>(a.x + b.x, a.y + b.y); }
template <typename T> HAPPYRAY_HD inline Vec2<T> operator-(const Vec2<T>& a, const Vec2<T>& b) { return Vec2<T>(a.x - b.x, a.y - b.y); }
template <typename T, typename S> HAPPYRAY_HD inline Vec2<T> operator*(const Vec2<T>& a, S s) { return Vec2<T>(a.x * s, a.y * s); }
template <typename T, typename S> HAPPYRAY_HD inline Vec2<T> operator/(const Vec2<T>& a, S s) { return Vec2<T>(a.x / s, a.y / s); }
template <typename T> HAPPYRAY_HD inline Vec2<T>& operator+=(Vec2<T>& a, const Vec2<T>& b) { a = a + b; return a; }

template <typename T> struct Vec3 {
    T x = T{};
    T y = T{};
    T z = T{};

    HAPPYRAY_HD Vec3() = default;
    HAPPYRAY_HD Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    HAPPYRAY_HD explicit Vec3(T s) : x(s), y(s), z(s) {}
};

template <typename T> struct Vec4 {
    T x = T{};
    T y = T{};
    T z = T{};
    T w = T{};

    HAPPYRAY_HD Vec4() = default;
    HAPPYRAY_HD Vec4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
    HAPPYRAY_HD Vec4(const Vec3<T>& v, T w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

using Vec2f = Vec2<float>;
using Vec2i = Vec2<int>;
using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec4f = Vec4<float>;

// --- Vec3f arithmetic ---

template <typename T> HAPPYRAY_HD inline Vec3<T> operator+(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}
template <typename T> HAPPYRAY_HD inline Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}
template <typename T> HAPPYRAY_HD inline Vec3<T> operator-(const Vec3<T>& a) { return Vec3<T>(-a.x, -a.y, -a.z); }
template <typename T> HAPPYRAY_HD inline Vec3<T> operator*(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}
template <typename T, typename S> HAPPYRAY_HD inline Vec3<T> operator*(const Vec3<T>& a, S s) { return Vec3<T>(a.x * s, a.y * s, a.z * s); }
template <typename S, typename T> HAPPYRAY_HD inline Vec3<T> operator*(S s, const Vec3<T>& a) { return a * s; }
template <typename T, typename S> HAPPYRAY_HD inline Vec3<T> operator/(const Vec3<T>& a, S s) { return Vec3<T>(a.x / s, a.y / s, a.z / s); }
template <typename T> HAPPYRAY_HD inline Vec3<T>& operator+=(Vec3<T>& a, const Vec3<T>& b) { a = a + b; return a; }
template <typename T> HAPPYRAY_HD inline Vec3<T>& operator-=(Vec3<T>& a, const Vec3<T>& b) { a = a - b; return a; }
template <typename T, typename S> HAPPYRAY_HD inline Vec3<T>& operator*=(Vec3<T>& a, S s) { a = a * s; return a; }

template <typename T> HAPPYRAY_HD inline T Dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T> HAPPYRAY_HD inline Vec3<T> Cross(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HAPPYRAY_HD inline float LengthSquared(const Vec3f& a) { return Dot(a, a); }
HAPPYRAY_HD inline float Length(const Vec3f& a) { return std::sqrt(LengthSquared(a)); }
HAPPYRAY_HD inline Vec3f Normalize(const Vec3f& a) {
    const float len = Length(a);
    return len > 0.0f ? a / len : Vec3f(0.0f, 0.0f, 0.0f);
}
template <typename T> HAPPYRAY_HD inline Vec3<T> Min(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}
template <typename T> HAPPYRAY_HD inline Vec3<T> Max(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}
HAPPYRAY_HD inline Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t) { return a + (b - a) * t; }
HAPPYRAY_HD inline Vec3f Reflect(const Vec3f& incident, const Vec3f& normal) {
    return incident - normal * (2.0f * Dot(incident, normal));
}

}  // namespace happyray
