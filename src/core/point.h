#pragma once

#include "bbox.h"
#include "primitive.h"
#include "ray.h"

namespace happyray
{

	// A ray-facing disc of `radius` (the hit tolerance when zero).
	struct Point : Primitive<1>
	{
		float radius = 0.0f;

		HAPPYRAY_HD Point() = default;
		HAPPYRAY_HD explicit Point(const Vec3f &position, float radius_ = 0.0f) : radius(radius_)
		{
			vtx[0] = position;
		}
	};

	constexpr float kPointRayTolerance = 1e-4f;

	template <> struct BBoxExtractor<Point>
	{
		HAPPYRAY_HD static BBox Get(const Point &point)
		{
			const Vec3f extent(point.radius > 0.0f ? point.radius : 0.0f);
			return BBox(point.vtx[0] - extent, point.vtx[0] + extent);
		}
	};

	HAPPYRAY_HD inline bool
	IntersectRayPoint(const Ray &ray, const Point &point, float *out_t, float tolerance = kPointRayTolerance)
	{
		const float direction_length_squared = LengthSquared(ray.direction);
		const float radius					 = point.radius > 0.0f ? point.radius : tolerance;
		if (direction_length_squared == 0.0f || radius < 0.0f)
			return false;
		// Rays leaving a disc (e.g. AO rays) never hit it.
		if (LengthSquared(ray.origin - point.vtx[0]) < radius * radius)
			return false;
		const float t = Dot(point.vtx[0] - ray.origin, ray.direction) / direction_length_squared;
		if (t < ray.t_min || t > ray.t_max)
			return false;
		const Vec3f delta = ray.At(t) - point.vtx[0];
		if (LengthSquared(delta) > radius * radius)
			return false;
		*out_t = t;
		return true;
	}

	// Conservative: tests the disc's bounding sphere.
	HAPPYRAY_HD inline bool Intersects(const Point &point, const BBox &box)
	{
		if (box.IsEmpty())
			return false;
		const Vec3f &c = point.vtx[0];
		const Vec3f	 closest(fminf(fmaxf(c.x, box.min_bound.x), box.max_bound.x),
							 fminf(fmaxf(c.y, box.min_bound.y), box.max_bound.y),
							 fminf(fmaxf(c.z, box.min_bound.z), box.max_bound.z));
		const float	 radius = point.radius > 0.0f ? point.radius : 0.0f;
		return LengthSquared(c - closest) <= radius * radius;
	}

} // namespace happyray
