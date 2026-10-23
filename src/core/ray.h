#pragma once

#include "vec.h"

namespace happyray {

struct Ray {
    Vec3 origin;
    Vec3 direction;  // Not required to be normalized; callers decide the convention.
    float t_min = 1e-4f;
    float t_max = 1e30f;

    HAPPYRAY_HD Ray() = default;
    HAPPYRAY_HD Ray(const Vec3& o, const Vec3& d, float tmin = 1e-4f, float tmax = 1e30f)
        : origin(o), direction(d), t_min(tmin), t_max(tmax) {}

    HAPPYRAY_HD Vec3 At(float t) const { return origin + direction * t; }
};

}  // namespace happyray
