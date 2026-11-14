// Axis-aligned bounding box: grid building block and the solid box primitive (PrimitiveType::kBox).
#pragma once

#include <cmath>

#include "primitive.h"
#include "ray.h"
#include "vec.h"

namespace happyray
{

	struct BBox
	{
		Vec3f min_bound = Vec3f(1e30f, 1e30f, 1e30f);
		Vec3f max_bound = Vec3f(-1e30f, -1e30f, -1e30f);

		HAPPYRAY_HD BBox() = default;
		HAPPYRAY_HD BBox(const Vec3f &min_b, const Vec3f &max_b) : min_bound(min_b), max_bound(max_b)
		{
		}

		HAPPYRAY_HD bool IsEmpty() const
		{
			return min_bound.x > max_bound.x || min_bound.y > max_bound.y || min_bound.z > max_bound.z;
		}

		HAPPYRAY_HD Vec3f Extent() const
		{
			return max_bound - min_bound;
		}
		HAPPYRAY_HD Vec3f Center() const
		{
			return (min_bound + max_bound) * 0.5f;
		}

		HAPPYRAY_HD void Grow(const Vec3f &p)
		{
			min_bound = Min(min_bound, p);
			max_bound = Max(max_bound, p);
		}

		HAPPYRAY_HD void Grow(const BBox &other)
		{
			min_bound = Min(min_bound, other.min_bound);
			max_bound = Max(max_bound, other.max_bound);
		}
	};

	template <typename TPrimitive> struct BBoxExtractor
	{
		HAPPYRAY_HD static BBox Get(const TPrimitive &primitive)
		{
			BBox box;
			for (int i = 0; i < TPrimitive::kNumVertices; ++i)
			{
				box.Grow(primitive.vtx[i]);
			}
			return box;
		}
	};

	// A box is its own bounds when used as a primitive (PrimitiveType::kBox).
	template <> struct BBoxExtractor<BBox>
	{
		HAPPYRAY_HD static BBox Get(const BBox &box)
		{
			return box;
		}
	};

	// Slab test; entry/exit distances are clipped to the ray's [t_min, t_max].
	HAPPYRAY_HD inline bool IntersectRayBBox(const Ray &ray, const BBox &box, float *out_t_near, float *out_t_far)
	{
		float t_near = ray.t_min;
		float t_far	 = ray.t_max;

		const float *ray_origin = &ray.origin.x;
		const float *ray_dir	= &ray.direction.x;
		const float *box_min	= &box.min_bound.x;
		const float *box_max	= &box.max_bound.x;

		for (int axis = 0; axis < 3; ++axis)
		{
			if (ray_dir[axis] == 0.0f)
			{
				if (ray_origin[axis] < box_min[axis] || ray_origin[axis] > box_max[axis])
				{
					return false;
				}
				continue;
			}
			const float inv_dir = 1.0f / ray_dir[axis];
			float		t0		= (box_min[axis] - ray_origin[axis]) * inv_dir;
			float		t1		= (box_max[axis] - ray_origin[axis]) * inv_dir;
			if (t0 > t1)
			{
				const float tmp = t0;
				t0				= t1;
				t1				= tmp;
			}
			t_near = t0 > t_near ? t0 : t_near;
			t_far  = t1 < t_far ? t1 : t_far;
			if (t_near > t_far)
			{
				return false;
			}
		}

		*out_t_near = t_near;
		*out_t_far	= t_far;
		return true;
	}

	HAPPYRAY_HD inline bool Overlaps(const BBox &a, const BBox &b)
	{
		return a.min_bound.x <= b.max_bound.x && a.max_bound.x >= b.min_bound.x && a.min_bound.y <= b.max_bound.y &&
			   a.max_bound.y >= b.min_bound.y && a.min_bound.z <= b.max_bound.z && a.max_bound.z >= b.min_bound.z;
	}

	// Box primitive / cell overlap (PrimitiveTraits<BBox>::CellOverlap).
	HAPPYRAY_HD inline bool Intersects(const BBox &primitive, const BBox &box)
	{
		return !primitive.IsEmpty() && !box.IsEmpty() && Overlaps(primitive, box);
	}

	// Ray hit on the surface of a solid box: the entry point, or the exit point if the ray starts
	// inside. Writes the hit distance and the axis (0..2) of the face that was hit.
	HAPPYRAY_HD inline bool IntersectRayBoxSurface(const Ray &ray, const BBox &box, float *out_t, int *out_axis)
	{
		if (box.IsEmpty())
			return false;
		const float *ray_origin = &ray.origin.x;
		const float *ray_dir	= &ray.direction.x;
		const float *box_min	= &box.min_bound.x;
		const float *box_max	= &box.max_bound.x;
		float		 t_near = -INFINITY, t_far = INFINITY;
		int			 near_axis = 0, far_axis = 0;
		for (int axis = 0; axis < 3; ++axis)
		{
			if (ray_dir[axis] == 0.0f)
			{
				if (ray_origin[axis] < box_min[axis] || ray_origin[axis] > box_max[axis])
					return false;
				continue;
			}
			const float inv_dir = 1.0f / ray_dir[axis];
			const float t0		= (box_min[axis] - ray_origin[axis]) * inv_dir;
			const float t1		= (box_max[axis] - ray_origin[axis]) * inv_dir;
			const float entry = t0 < t1 ? t0 : t1, exit = t0 < t1 ? t1 : t0;
			if (entry > t_near)
			{
				t_near	  = entry;
				near_axis = axis;
			}
			if (exit < t_far)
			{
				t_far	 = exit;
				far_axis = axis;
			}
		}
		if (t_near > t_far)
			return false;
		if (t_near >= ray.t_min && t_near <= ray.t_max)
		{
			*out_t	  = t_near;
			*out_axis = near_axis;
			return true;
		}
		if (t_far >= ray.t_min && t_far <= ray.t_max)
		{
			*out_t	  = t_far;
			*out_axis = far_axis;
			return true;
		}
		return false;
	}

	// Whether segment [a, b] intersects `box`.
	HAPPYRAY_HD inline bool SegmentIntersectsBBox(const Vec3f &a, const Vec3f &b, const BBox &box)
	{
		float t_near, t_far;
		return IntersectRayBBox(Ray(a, b - a, 0.0f, 1.0f), box, &t_near, &t_far);
	}

	namespace detail
	{
		// +/-infinity for zero (HAGrid's safe_rcp), so traversal multiplies instead of divides.
		HAPPYRAY_HD inline float SafeRcp(float x)
		{
			return x != 0.0f ? 1.0f / x : std::copysign(INFINITY, x);
		}

		HAPPYRAY_HD inline Vec3f SafeRcp(const Vec3f &d)
		{
			return Vec3f(SafeRcp(d.x), SafeRcp(d.y), SafeRcp(d.z));
		}

		// Zero direction components are handled explicitly (no 0 * inf NaNs).
		HAPPYRAY_HD inline bool IntersectRayBoxRcp(const Vec3f &origin,
												   const Vec3f &dir,
												   const Vec3f &inv_dir,
												   const Vec3f &box_min,
												   const Vec3f &box_max,
												   float		t_min,
												   float		t_max,
												   float	   *out_t_near,
												   float	   *out_t_far)
		{
			float t_near = t_min, t_far = t_max;
#define HAPPYRAY_SLAB(a)                                                                                               \
	if (dir.a != 0.0f)                                                                                                 \
	{                                                                                                                  \
		const float t0 = (box_min.a - origin.a) * inv_dir.a;                                                           \
		const float t1 = (box_max.a - origin.a) * inv_dir.a;                                                           \
		t_near		   = std::fmax(t_near, std::fmin(t0, t1));                                                         \
		t_far		   = std::fmin(t_far, std::fmax(t0, t1));                                                          \
	}                                                                                                                  \
	else if (origin.a < box_min.a || origin.a > box_max.a)                                                             \
	{                                                                                                                  \
		return false;                                                                                                  \
	}
			HAPPYRAY_SLAB(x)
			HAPPYRAY_SLAB(y)
			HAPPYRAY_SLAB(z)
#undef HAPPYRAY_SLAB
			*out_t_near = t_near;
			*out_t_far	= t_far;
			return t_near <= t_far;
		}
	} // namespace detail

} // namespace happyray
