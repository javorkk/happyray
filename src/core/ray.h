#pragma once

#include "vec.h"

namespace happyray
{

	struct Ray
	{
		Vec3f origin;
		Vec3f direction; // Not required to be normalized; callers decide the convention.
		float t_min = 1e-4f;
		float t_max = 1e30f;

		HAPPYRAY_HD Ray() = default;
		HAPPYRAY_HD Ray(const Vec3f &o, const Vec3f &d, float tmin = 1e-4f, float tmax = 1e30f)
			: origin(o), direction(d), t_min(tmin), t_max(tmax)
		{
		}

		HAPPYRAY_HD Vec3f At(float t) const
		{
			return origin + direction * t;
		}
	};

} // namespace happyray
