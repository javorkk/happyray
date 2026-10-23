// Host/device-compatible PCG32, Halton low-discrepancy sequences, and cosine-weighted
// hemisphere sampling for the ambient-occlusion integrator.
#pragma once

#include <cstdint>

#include "vec.h"

namespace happyray {

struct Rng {
    uint64_t state = 0;
    uint64_t inc = 0;

    HAPPYRAY_HD Rng() = default;
    HAPPYRAY_HD Rng(uint64_t seed, uint64_t sequence) { Seed(seed, sequence); }

    HAPPYRAY_HD void Seed(uint64_t seed, uint64_t sequence) {
        state = 0;
        inc = (sequence << 1u) | 1u;
        NextUint32();
        state += seed;
        NextUint32();
    }

    HAPPYRAY_HD uint32_t NextUint32() {
        const uint64_t old_state = state;
        state = old_state * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
        const uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform float in [0, 1).
    HAPPYRAY_HD float NextFloat() {
        return static_cast<float>(NextUint32()) * (1.0f / 4294967296.0f);
    }
};

// Combines a few integer coordinates (pixel x/y, frame index, sample index) into a seed, so
// every (pixel, frame, sample) tuple gets its own decorrelated RNG stream.
HAPPYRAY_HD inline uint64_t HashSeed(uint32_t a, uint32_t b, uint32_t c) {
    uint64_t h = (static_cast<uint64_t>(a) * 2654435761u) ^ (static_cast<uint64_t>(b) * 2246822519u) ^
                 (static_cast<uint64_t>(c) * 3266489917u);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

// Radical inverse of `index` in the given prime `base`: reflects the digits of `index` (written
// in that base) around the "decimal" point, e.g. base 2: 1 -> 0.5, 2 -> 0.25, 3 -> 0.75, ...
HAPPYRAY_HD inline float HaltonRadicalInverse(uint32_t index, uint32_t base) {
    float result = 0.0f;
    float inv_base = 1.0f / static_cast<float>(base);
    float fraction = inv_base;
    while (index > 0) {
        result += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction *= inv_base;
    }
    return result;
}

// The 2D Halton sample at `index`, using the standard base-2/base-3 pair (the lowest two
// primes, which is sufficient for a 2D hemisphere-direction sample).
HAPPYRAY_HD inline Vec2 Halton2D(uint32_t index) {
    return Vec2(HaltonRadicalInverse(index, 2u), HaltonRadicalInverse(index, 3u));
}

HAPPYRAY_HD inline float Frac(float v) { return v - static_cast<float>(static_cast<int>(v)) + (v < 0.0f ? 1.0f : 0.0f); }

// Cranley-Patterson rotation: toroidally shifts a [0,1) low-discrepancy sample by a per-pixel
// random `offset` (also in [0,1)). This decorrelates the (otherwise identical) Halton sequence
// across pixels while preserving its low-discrepancy progression as `index` increases, which is
// exactly what a progressive per-pixel accumulator needs.
HAPPYRAY_HD inline float CranleyPattersonRotate(float value, float offset) { return Frac(value + offset); }

HAPPYRAY_HD inline Vec2 CranleyPattersonRotate(const Vec2& value, const Vec2& offset) {
    return Vec2(CranleyPattersonRotate(value.x, offset.x), CranleyPattersonRotate(value.y, offset.y));
}

// Convenience: the Cranley-Patterson-rotated 2D Halton sample at `index`, decorrelated per
// pixel by `pixel_offset`.
HAPPYRAY_HD inline Vec2 Halton2D(uint32_t index, const Vec2& pixel_offset) {
    return CranleyPattersonRotate(Halton2D(index), pixel_offset);
}

// Builds an orthonormal basis around unit vector `n` (Duff et al., "Building an Orthonormal
// Basis, Revisited"), used to transform hemisphere samples from local space into world space.
HAPPYRAY_HD inline void MakeOrthonormalBasis(const Vec3& n, Vec3* tangent, Vec3* bitangent) {
    const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    *tangent = Vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    *bitangent = Vec3(b, sign + n.y * n.y * a, -n.y);
}

// Cosine-weighted sample over the hemisphere around normal `n`.
HAPPYRAY_HD inline Vec3 SampleCosineHemisphere(const Vec3& n, float u1, float u2) {
    const float r = std::sqrt(u1);
    const float theta = 2.0f * 3.14159265358979323846f * u2;
    const float local_x = r * std::cos(theta);
    const float local_y = r * std::sin(theta);
    const float local_z = std::sqrt(1.0f - u1);

    Vec3 tangent, bitangent;
    MakeOrthonormalBasis(n, &tangent, &bitangent);
    return tangent * local_x + bitangent * local_y + n * local_z;
}

}  // namespace happyray
