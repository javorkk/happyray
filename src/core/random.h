// Host/device PCG32, Halton sequences, and cosine-weighted hemisphere sampling.
#pragma once

#include <cstdint>

#include "vec.h"

namespace happyray
{

	struct Rng
	{
		uint64_t state = 0;
		uint64_t inc   = 0;

		HAPPYRAY_HD Rng() = default;
		HAPPYRAY_HD Rng(uint64_t seed, uint64_t sequence)
		{
			Seed(seed, sequence);
		}

		HAPPYRAY_HD void Seed(uint64_t seed, uint64_t sequence)
		{
			state = 0;
			inc	  = (sequence << 1u) | 1u;
			NextUint32();
			state += seed;
			NextUint32();
		}

		HAPPYRAY_HD uint32_t NextUint32()
		{
			const uint64_t old_state  = state;
			state					  = old_state * 6364136223846793005ULL + inc;
			const uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
			const uint32_t rot		  = static_cast<uint32_t>(old_state >> 59u);
			return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
		}

		HAPPYRAY_HD float NextFloat()
		{
			return static_cast<float>(NextUint32()) * (1.0f / 4294967296.0f);
		}
	};

	// Decorrelated seed per (pixel, frame, sample) tuple.
	HAPPYRAY_HD inline uint64_t HashSeed(uint32_t a, uint32_t b, uint32_t c)
	{
		uint64_t h = (static_cast<uint64_t>(a) * 2654435761u) ^ (static_cast<uint64_t>(b) * 2246822519u) ^
					 (static_cast<uint64_t>(c) * 3266489917u);
		h ^= h >> 33;
		h *= 0xff51afd7ed558ccdULL;
		h ^= h >> 33;
		return h;
	}

	// Reflects the base-`base` digits of `index` around the radix point.
	HAPPYRAY_HD inline float HaltonRadicalInverse(uint32_t index, uint32_t base)
	{
		float result   = 0.0f;
		float inv_base = 1.0f / static_cast<float>(base);
		float fraction = inv_base;
		while (index > 0)
		{
			result += static_cast<float>(index % base) * fraction;
			index /= base;
			fraction *= inv_base;
		}
		return result;
	}

	// Bases 2 and 3.
	HAPPYRAY_HD inline Vec2f Halton2D(uint32_t index)
	{
		return Vec2f(HaltonRadicalInverse(index, 2u), HaltonRadicalInverse(index, 3u));
	}

	HAPPYRAY_HD inline float Frac(float v)
	{
		return v - static_cast<float>(static_cast<int>(v)) + (v < 0.0f ? 1.0f : 0.0f);
	}

	// Cranley-Patterson rotation: a toroidal per-pixel shift that decorrelates pixels while keeping
	// the low-discrepancy progression.
	HAPPYRAY_HD inline float CranleyPattersonRotate(float value, float offset)
	{
		return Frac(value + offset);
	}

	HAPPYRAY_HD inline Vec2f CranleyPattersonRotate(const Vec2f &value, const Vec2f &offset)
	{
		return Vec2f(CranleyPattersonRotate(value.x, offset.x), CranleyPattersonRotate(value.y, offset.y));
	}

	HAPPYRAY_HD inline Vec2f Halton2D(uint32_t index, const Vec2f &pixel_offset)
	{
		return CranleyPattersonRotate(Halton2D(index), pixel_offset);
	}

	// Duff et al., "Building an Orthonormal Basis, Revisited".
	HAPPYRAY_HD inline void MakeOrthonormalBasis(const Vec3f &n, Vec3f *tangent, Vec3f *bitangent)
	{
		const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
		const float a	 = -1.0f / (sign + n.z);
		const float b	 = n.x * n.y * a;
		*tangent		 = Vec3f(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
		*bitangent		 = Vec3f(b, sign + n.y * n.y * a, -n.y);
	}

	HAPPYRAY_HD inline Vec3f SampleCosineHemisphere(const Vec3f &n, float u1, float u2)
	{
		const float r		= std::sqrt(u1);
		const float theta	= 2.0f * 3.14159265358979323846f * u2;
		const float local_x = r * std::cos(theta);
		const float local_y = r * std::sin(theta);
		const float local_z = std::sqrt(1.0f - u1);

		Vec3f tangent, bitangent;
		MakeOrthonormalBasis(n, &tangent, &bitangent);
		return tangent * local_x + bitangent * local_y + n * local_z;
	}

} // namespace happyray
