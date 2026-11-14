// Host/device vector math.
#pragma once

#include <cmath>
#include <cstring>

#if defined(__CUDACC__)
#define HAPPYRAY_HD __host__ __device__
#else
#define HAPPYRAY_HD
#endif

namespace happyray
{

	HAPPYRAY_HD inline int ClampInt(int v, int lo, int hi)
	{
		return v < lo ? lo : (v > hi ? hi : v);
	}

	template <typename T> struct Vec2
	{
		T x = T{};
		T y = T{};

		HAPPYRAY_HD Vec2() = default;
		HAPPYRAY_HD Vec2(T x_, T y_) : x(x_), y(y_)
		{
		}
	};

	template <typename T> HAPPYRAY_HD inline Vec2<T> operator+(const Vec2<T> &a, const Vec2<T> &b)
	{
		return Vec2<T>(a.x + b.x, a.y + b.y);
	}
	template <typename T> HAPPYRAY_HD inline Vec2<T> operator-(const Vec2<T> &a, const Vec2<T> &b)
	{
		return Vec2<T>(a.x - b.x, a.y - b.y);
	}
	template <typename T, typename S> HAPPYRAY_HD inline Vec2<T> operator*(const Vec2<T> &a, S s)
	{
		return Vec2<T>(a.x * s, a.y * s);
	}
	template <typename T, typename S> HAPPYRAY_HD inline Vec2<T> operator/(const Vec2<T> &a, S s)
	{
		return Vec2<T>(a.x / s, a.y / s);
	}
	template <typename T> HAPPYRAY_HD inline Vec2<T> &operator+=(Vec2<T> &a, const Vec2<T> &b)
	{
		a = a + b;
		return a;
	}

	template <typename T> struct Vec3
	{
		T x = T{};
		T y = T{};
		T z = T{};

		HAPPYRAY_HD Vec3() = default;
		HAPPYRAY_HD Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_)
		{
		}
		HAPPYRAY_HD explicit Vec3(T s) : x(s), y(s), z(s)
		{
		}
	};

	template <typename T> struct Vec4
	{
		T x = T{};
		T y = T{};
		T z = T{};
		T w = T{};

		HAPPYRAY_HD Vec4() = default;
		HAPPYRAY_HD Vec4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_)
		{
		}
		HAPPYRAY_HD Vec4(const Vec3<T> &v, T w_) : x(v.x), y(v.y), z(v.z), w(w_)
		{
		}
	};

	using Vec2f = Vec2<float>;
	using Vec2i = Vec2<int>;
	using Vec3f = Vec3<float>;
	using Vec3i = Vec3<int>;
	using Vec4f = Vec4<float>;


	template <typename T> HAPPYRAY_HD inline Vec3<T> operator+(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> operator-(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> operator-(const Vec3<T> &a)
	{
		return Vec3<T>(-a.x, -a.y, -a.z);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> operator*(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
	}
	template <typename T, typename S> HAPPYRAY_HD inline Vec3<T> operator*(const Vec3<T> &a, S s)
	{
		return Vec3<T>(a.x * s, a.y * s, a.z * s);
	}
	template <typename S, typename T> HAPPYRAY_HD inline Vec3<T> operator*(S s, const Vec3<T> &a)
	{
		return a * s;
	}
	template <typename T, typename S> HAPPYRAY_HD inline Vec3<T> operator/(const Vec3<T> &a, S s)
	{
		return Vec3<T>(a.x / s, a.y / s, a.z / s);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> &operator+=(Vec3<T> &a, const Vec3<T> &b)
	{
		a = a + b;
		return a;
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> &operator-=(Vec3<T> &a, const Vec3<T> &b)
	{
		a = a - b;
		return a;
	}
	template <typename T, typename S> HAPPYRAY_HD inline Vec3<T> &operator*=(Vec3<T> &a, S s)
	{
		a = a * s;
		return a;
	}

	template <typename T> HAPPYRAY_HD inline T Dot(const Vec3<T> &a, const Vec3<T> &b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> Cross(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}
	HAPPYRAY_HD inline float LengthSquared(const Vec3f &a)
	{
		return Dot(a, a);
	}
	HAPPYRAY_HD inline float Length(const Vec3f &a)
	{
		return std::sqrt(LengthSquared(a));
	}
	HAPPYRAY_HD inline Vec3f Normalize(const Vec3f &a)
	{
		const float len = Length(a);
		return len > 0.0f ? a / len : Vec3f(0.0f, 0.0f, 0.0f);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> Min(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
	}
	template <typename T> HAPPYRAY_HD inline Vec3<T> Max(const Vec3<T> &a, const Vec3<T> &b)
	{
		return Vec3<T>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
	}
	HAPPYRAY_HD inline Vec3f Lerp(const Vec3f &a, const Vec3f &b, float t)
	{
		return a + (b - a) * t;
	}
	HAPPYRAY_HD inline Vec3f Reflect(const Vec3f &incident, const Vec3f &normal)
	{
		return incident - normal * (2.0f * Dot(incident, normal));
	}

	// Loads via the read-only data path (__ldg) for 4/8/16-byte (or 16-byte multiple) types; a plain
	// load otherwise and on the host. Only for memory no thread writes during the kernel.
	template <typename T> HAPPYRAY_HD inline T LoadRO(const T *p)
	{
#if defined(__CUDA_ARCH__)
		if constexpr (sizeof(T) % 16 == 0 && alignof(T) >= 16)
		{
			T value;
			for (size_t chunk = 0; chunk < sizeof(T) / 16; ++chunk)
			{
				const int4 raw = __ldg(reinterpret_cast<const int4 *>(p) + chunk);
				memcpy(reinterpret_cast<char *>(&value) + chunk * 16, &raw, 16);
			}
			return value;
		}
		else if constexpr (sizeof(T) == 8 && alignof(T) >= 8)
		{
			const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(p));
			T			value;
			memcpy(&value, &raw, sizeof(T));
			return value;
		}
		else if constexpr (sizeof(T) == 4 && alignof(T) >= 4)
		{
			const unsigned raw = __ldg(reinterpret_cast<const unsigned *>(p));
			T			   value;
			memcpy(&value, &raw, sizeof(T));
			return value;
		}
		else
			return *p;
#else
		return *p;
#endif
	}

} // namespace happyray
