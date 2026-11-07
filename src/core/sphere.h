#pragma once

#include "bbox.h"
#include "primitive.h"
#include "ray.h"

namespace happyray
{

	struct Sphere : Primitive<1>
	{
		float radius = 0.0f;

		HAPPYRAY_HD Sphere() = default;
		HAPPYRAY_HD Sphere(const Vec3f &center, float radius_) : radius(radius_)
		{
			vtx[0] = center;
		}
	};

	template <> struct BBoxExtractor<Sphere>
	{
		HAPPYRAY_HD static BBox Get(const Sphere &sphere)
		{
			const float radius = sphere.radius > 0.0f ? sphere.radius : 0.0f;
			const Vec3f extent(radius);
			return BBox(sphere.vtx[0] - extent, sphere.vtx[0] + extent);
		}
	};

	HAPPYRAY_HD inline bool IntersectRaySphere(const Ray &ray, const Sphere &sphere, float *out_t)
	{
		if (sphere.radius < 0.0f)
			return false;
		const float direction_length_squared = LengthSquared(ray.direction);
		if (direction_length_squared == 0.0f)
			return false;
		const Vec3f offset		 = ray.origin - sphere.vtx[0];
		const float half_b		 = Dot(offset, ray.direction);
		const float c			 = LengthSquared(offset) - sphere.radius * sphere.radius;
		const float discriminant = half_b * half_b - direction_length_squared * c;
		if (discriminant < 0.0f)
			return false;
		const float root   = std::sqrt(discriminant);
		const float near_t = (-half_b - root) / direction_length_squared;
		const float far_t  = (-half_b + root) / direction_length_squared;
		if (near_t >= ray.t_min && near_t <= ray.t_max)
		{
			*out_t = near_t;
			return true;
		}
		if (far_t >= ray.t_min && far_t <= ray.t_max)
		{
			*out_t = far_t;
			return true;
		}
		return false;
	}

	HAPPYRAY_HD inline bool Intersects(const Sphere &sphere, const BBox &box)
	{
		if (sphere.radius < 0.0f || box.IsEmpty())
			return false;
		const Vec3f closest(sphere.vtx[0].x < box.min_bound.x	? box.min_bound.x
							: sphere.vtx[0].x > box.max_bound.x ? box.max_bound.x
																: sphere.vtx[0].x,
							sphere.vtx[0].y < box.min_bound.y	? box.min_bound.y
							: sphere.vtx[0].y > box.max_bound.y ? box.max_bound.y
																: sphere.vtx[0].y,
							sphere.vtx[0].z < box.min_bound.z	? box.min_bound.z
							: sphere.vtx[0].z > box.max_bound.z ? box.max_bound.z
																: sphere.vtx[0].z);
		return LengthSquared(sphere.vtx[0] - closest) <= sphere.radius * sphere.radius;
	}

} // namespace happyray