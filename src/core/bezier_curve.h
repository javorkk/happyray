#pragma once

#include "bbox.h"
#include "primitive.h"
#include "ray.h"
#include "sphere.h"

namespace happyray
{

	struct BezierCurve : Primitive<4>
	{
		float radius = 0.0f;

		HAPPYRAY_HD BezierCurve() = default;
		HAPPYRAY_HD BezierCurve(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2, const Vec3f &p3, float radius_)
			: radius(radius_)
		{
			vtx[0] = p0;
			vtx[1] = p1;
			vtx[2] = p2;
			vtx[3] = p3;
		}
	};

	template <> struct BBoxExtractor<BezierCurve>
	{
		HAPPYRAY_HD static BBox Get(const BezierCurve &curve)
		{
			BBox box;
			for (int i = 0; i < BezierCurve::kNumVertices; ++i)
				box.Grow(curve.vtx[i]);
			const float radius = curve.radius > 0.0f ? curve.radius : 0.0f;
			const Vec3f extent(radius);
			box.min_bound -= extent;
			box.max_bound += extent;
			return box;
		}
	};

	constexpr float kBezierRayTolerance		   = 1e-3f;
	constexpr int	kBezierMaxSubdivisionDepth = 16;

	namespace detail
	{

		HAPPYRAY_HD inline float DistanceSquaredToSegment(const Vec3f &point, const Vec3f &a, const Vec3f &b)
		{
			const Vec3f edge				= b - a;
			const float edge_length_squared = LengthSquared(edge);
			const float t		  = edge_length_squared > 0.0f ? Dot(point - a, edge) / edge_length_squared : 0.0f;
			const float clamped_t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
			return LengthSquared(point - (a + edge * clamped_t));
		}

		HAPPYRAY_HD inline bool
		IntersectRayCapsule(const Ray &ray, const Vec3f &a, const Vec3f &b, float radius, float *out_t)
		{
			if (radius < 0.0f)
				return false;
			const Vec3f axis				= b - a;
			const float axis_length_squared = LengthSquared(axis);
			if (axis_length_squared == 0.0f)
				return IntersectRaySphere(ray, Sphere{a, radius}, out_t);

			bool		found					 = false;
			float		closest_t				 = ray.t_max;
			const Vec3f offset					 = ray.origin - a;
			const float direction_length_squared = LengthSquared(ray.direction);
			const float axis_direction			 = Dot(axis, ray.direction);
			const float axis_offset				 = Dot(axis, offset);
			const float direction_offset		 = Dot(ray.direction, offset);
			const float offset_length_squared	 = LengthSquared(offset);
			const float qa = axis_length_squared * direction_length_squared - axis_direction * axis_direction;
			const float qb = axis_length_squared * direction_offset - axis_offset * axis_direction;
			const float qc = axis_length_squared * offset_length_squared - axis_offset * axis_offset -
							 radius * radius * axis_length_squared;
			const float discriminant = qb * qb - qa * qc;
			if (qa > 0.0f && discriminant >= 0.0f)
			{
				const float root	 = std::sqrt(discriminant);
				const float roots[2] = {(-qb - root) / qa, (-qb + root) / qa};
				for (int i = 0; i < 2; ++i)
				{
					const float t		   = roots[i];
					const float along_axis = axis_offset + t * axis_direction;
					if (t >= ray.t_min && t <= ray.t_max && along_axis >= 0.0f && along_axis <= axis_length_squared &&
						(!found || t < closest_t))
					{
						closest_t = t;
						found	  = true;
					}
				}
			}

			float cap_t;
			if (IntersectRaySphere(ray, Sphere{a, radius}, &cap_t) && (!found || cap_t < closest_t))
			{
				closest_t = cap_t;
				found	  = true;
			}
			if (IntersectRaySphere(ray, Sphere{b, radius}, &cap_t) && (!found || cap_t < closest_t))
			{
				closest_t = cap_t;
				found	  = true;
			}
			if (found)
				*out_t = closest_t;
			return found;
		}

		struct BezierSegment
		{
			Vec3f control[4];
			int	  depth;
		};

		HAPPYRAY_HD inline void
		SplitBezierSegment(const BezierSegment &segment, BezierSegment *left, BezierSegment *right)
		{
			const Vec3f p01		 = Lerp(segment.control[0], segment.control[1], 0.5f);
			const Vec3f p12		 = Lerp(segment.control[1], segment.control[2], 0.5f);
			const Vec3f p23		 = Lerp(segment.control[2], segment.control[3], 0.5f);
			const Vec3f p012	 = Lerp(p01, p12, 0.5f);
			const Vec3f p123	 = Lerp(p12, p23, 0.5f);
			const Vec3f midpoint = Lerp(p012, p123, 0.5f);
			*left				 = {
				   {segment.control[0], p01, p012, midpoint},
				   segment.depth + 1
			};
			*right = {
				{midpoint, p123, p23, segment.control[3]},
				segment.depth + 1
			 };
		}

	} // namespace detail

	HAPPYRAY_HD inline bool IntersectRayBezierCurve(const Ray		  &ray,
													const BezierCurve &curve,
													float			  *out_t,
													float			   tolerance = kBezierRayTolerance)
	{
		if (curve.radius < 0.0f || tolerance < 0.0f)
			return false;
		detail::BezierSegment stack[kBezierMaxSubdivisionDepth + 1];
		stack[0] = {
			{curve.vtx[0], curve.vtx[1], curve.vtx[2], curve.vtx[3]},
			0
		 };
		int			stack_size		  = 1;
		bool		found			  = false;
		float		closest_t		  = ray.t_max;
		const float tolerance_squared = tolerance * tolerance;

		while (stack_size > 0)
		{
			const detail::BezierSegment segment = stack[--stack_size];
			const float					flatness_squared =
				fmaxf(detail::DistanceSquaredToSegment(segment.control[1], segment.control[0], segment.control[3]),
					  detail::DistanceSquaredToSegment(segment.control[2], segment.control[0], segment.control[3]));
			if (flatness_squared > tolerance_squared && segment.depth < kBezierMaxSubdivisionDepth)
			{
				detail::BezierSegment left, right;
				detail::SplitBezierSegment(segment, &left, &right);
				stack[stack_size++] = right;
				stack[stack_size++] = left;
				continue;
			}

			const float proxy_radius = curve.radius + std::sqrt(flatness_squared) + tolerance;
			float		t;
			if (detail::IntersectRayCapsule(ray, segment.control[0], segment.control[3], proxy_radius, &t) &&
				(!found || t < closest_t))
			{
				closest_t = t;
				found	  = true;
			}
		}

		if (found)
			*out_t = closest_t;
		return found;
	}

	HAPPYRAY_HD inline bool Intersects(const BezierCurve &curve, const BBox &box)
	{
		if (curve.radius < 0.0f || box.IsEmpty())
			return false;
		const BBox curve_bounds = BBoxExtractor<BezierCurve>::Get(curve);
		return curve_bounds.min_bound.x <= box.max_bound.x && curve_bounds.max_bound.x >= box.min_bound.x &&
			   curve_bounds.min_bound.y <= box.max_bound.y && curve_bounds.max_bound.y >= box.min_bound.y &&
			   curve_bounds.min_bound.z <= box.max_bound.z && curve_bounds.max_bound.z >= box.min_bound.z;
	}

} // namespace happyray