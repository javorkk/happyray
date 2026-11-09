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

	constexpr float kBezierRayTolerance		   = 1e-3f;
	constexpr int	kBezierMaxSubdivisionDepth = 16;
	constexpr int	kBezierBoxSubdivisionDepth = 10;

	// Includes the slack of IntersectRayBezierCurve's capsule proxies.
	template <> struct BBoxExtractor<BezierCurve>
	{
		HAPPYRAY_HD static BBox Get(const BezierCurve &curve)
		{
			BBox box;
			for (int i = 0; i < BezierCurve::kNumVertices; ++i)
				box.Grow(curve.vtx[i]);
			const float radius = curve.radius > 0.0f ? curve.radius : 0.0f;
			const Vec3f extent(radius + 2.0f * kBezierRayTolerance);
			box.min_bound -= extent;
			box.max_bound += extent;
			return box;
		}
	};

	HAPPYRAY_HD inline Vec3f EvaluateBezierCurve(const BezierCurve &curve, float u)
	{
		const float s = 1.0f - u;
		return curve.vtx[0] * (s * s * s) + curve.vtx[1] * (3.0f * s * s * u) + curve.vtx[2] * (3.0f * s * u * u) +
			   curve.vtx[3] * (u * u * u);
	}

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

		// `out_s` receives the hit's fraction along [a, b].
		HAPPYRAY_HD inline bool
		IntersectRayCapsule(const Ray &ray, const Vec3f &a, const Vec3f &b, float radius, float *out_t, float *out_s)
		{
			if (radius < 0.0f)
				return false;
			const Vec3f axis				= b - a;
			const float axis_length_squared = LengthSquared(axis);
			if (axis_length_squared == 0.0f)
			{
				*out_s = 0.0f;
				return IntersectRaySphere(ray, Sphere{a, radius}, out_t);
			}

			bool		found					 = false;
			float		closest_t				 = ray.t_max;
			float		closest_s				 = 0.0f;
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
						closest_s = along_axis / axis_length_squared;
						found	  = true;
					}
				}
			}

			float cap_t;
			if (IntersectRaySphere(ray, Sphere{a, radius}, &cap_t) && (!found || cap_t < closest_t))
			{
				closest_t = cap_t;
				closest_s = 0.0f;
				found	  = true;
			}
			if (IntersectRaySphere(ray, Sphere{b, radius}, &cap_t) && (!found || cap_t < closest_t))
			{
				closest_t = cap_t;
				closest_s = 1.0f;
				found	  = true;
			}
			if (found)
			{
				*out_t = closest_t;
				*out_s = closest_s;
			}
			return found;
		}

		struct BezierSegment
		{
			Vec3f control[4];
			float u0, u1; // Curve parameter range.
			int	  depth;
		};

		HAPPYRAY_HD inline BezierSegment WholeBezierSegment(const BezierCurve &curve)
		{
			return {
				{curve.vtx[0], curve.vtx[1], curve.vtx[2], curve.vtx[3]},
				0.0f, 1.0f, 0
			};
		}

		// Max distance of the inner control points from the chord; bounds the curve's deviation.
		HAPPYRAY_HD inline float BezierFlatnessSquared(const BezierSegment &segment)
		{
			return fmaxf(DistanceSquaredToSegment(segment.control[1], segment.control[0], segment.control[3]),
						 DistanceSquaredToSegment(segment.control[2], segment.control[0], segment.control[3]));
		}

		HAPPYRAY_HD inline void
		SplitBezierSegment(const BezierSegment &segment, BezierSegment *left, BezierSegment *right)
		{
			const Vec3f p01		 = Lerp(segment.control[0], segment.control[1], 0.5f);
			const Vec3f p12		 = Lerp(segment.control[1], segment.control[2], 0.5f);
			const Vec3f p23		 = Lerp(segment.control[2], segment.control[3], 0.5f);
			const Vec3f p012	 = Lerp(p01, p12, 0.5f);
			const Vec3f p123	 = Lerp(p12, p23, 0.5f);
			const Vec3f midpoint = Lerp(p012, p123, 0.5f);
			const float u_mid	 = 0.5f * (segment.u0 + segment.u1);
			*left				 = {
				   {segment.control[0], p01, p012, midpoint},
				   segment.u0, u_mid, segment.depth + 1
			};
			*right = {
				{midpoint, p123, p23, segment.control[3]},
				u_mid, segment.u1, segment.depth + 1
			 };
		}

	} // namespace detail

	// `out_u` receives the curve parameter of the hit.
	HAPPYRAY_HD inline bool IntersectRayBezierCurve(const Ray		  &ray,
													const BezierCurve &curve,
													float			  *out_t,
													float			  *out_u,
													float			   tolerance = kBezierRayTolerance)
	{
		if (curve.radius < 0.0f || tolerance < 0.0f)
			return false;
		detail::BezierSegment stack[kBezierMaxSubdivisionDepth + 1];
		stack[0]					  = detail::WholeBezierSegment(curve);
		int			stack_size		  = 1;
		bool		found			  = false;
		float		closest_t		  = ray.t_max;
		float		closest_u		  = 0.0f;
		const float tolerance_squared = tolerance * tolerance;

		while (stack_size > 0)
		{
			const detail::BezierSegment segment			 = stack[--stack_size];
			const float					flatness_squared = detail::BezierFlatnessSquared(segment);
			if (flatness_squared > tolerance_squared && segment.depth < kBezierMaxSubdivisionDepth)
			{
				detail::BezierSegment left, right;
				detail::SplitBezierSegment(segment, &left, &right);
				stack[stack_size++] = right;
				stack[stack_size++] = left;
				continue;
			}

			const float proxy_radius = curve.radius + std::sqrt(flatness_squared) + tolerance;
			float		t, s;
			if (detail::IntersectRayCapsule(ray, segment.control[0], segment.control[3], proxy_radius, &t, &s) &&
				(!found || t < closest_t))
			{
				closest_t = t;
				closest_u = segment.u0 + s * (segment.u1 - segment.u0);
				found	  = true;
			}
		}

		if (found)
		{
			*out_t = closest_t;
			*out_u = closest_u;
		}
		return found;
	}

	HAPPYRAY_HD inline bool IntersectRayBezierCurve(const Ray		  &ray,
													const BezierCurve &curve,
													float			  *out_t,
													float			   tolerance = kBezierRayTolerance)
	{
		float u;
		return IntersectRayBezierCurve(ray, curve, out_t, &u, tolerance);
	}

	// Conservative w.r.t. IntersectRayBezierCurve's hits: subdivides the curve, culling sub-curves
	// whose hulls miss `box`, and tests the remaining chords against `box` grown by their capsule
	// radius.
	HAPPYRAY_HD inline bool Intersects(const BezierCurve &curve, const BBox &box)
	{
		if (curve.radius < 0.0f || box.IsEmpty())
			return false;
		const float slack	  = curve.radius + 2.0f * kBezierRayTolerance;
		const Vec3f extent	  = box.Extent();
		// Never finer than the ray test, so its leaves nest inside ours and `slack` covers them.
		const float tolerance =
			fmaxf(kBezierRayTolerance, 0.1f * fminf(extent.x, fminf(extent.y, extent.z)) + 0.5f * curve.radius);
		detail::BezierSegment stack[kBezierBoxSubdivisionDepth + 1];
		stack[0]	   = detail::WholeBezierSegment(curve);
		int stack_size = 1;
		while (stack_size > 0)
		{
			const detail::BezierSegment segment = stack[--stack_size];
			BBox						hull;
			for (int i = 0; i < 4; ++i)
				hull.Grow(segment.control[i]);
			if (!Overlaps(BBox(hull.min_bound - Vec3f(slack), hull.max_bound + Vec3f(slack)), box))
				continue;
			const float flatness = std::sqrt(detail::BezierFlatnessSquared(segment));
			if (flatness > tolerance && segment.depth < kBezierBoxSubdivisionDepth)
			{
				detail::BezierSegment left, right;
				detail::SplitBezierSegment(segment, &left, &right);
				stack[stack_size++] = right;
				stack[stack_size++] = left;
				continue;
			}
			const Vec3f grow(slack + flatness);
			if (SegmentIntersectsBBox(segment.control[0],
									  segment.control[3],
									  BBox(box.min_bound - grow, box.max_bound + grow)))
				return true;
		}
		return false;
	}

} // namespace happyray