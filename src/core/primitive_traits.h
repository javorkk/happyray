// Per-primitive-type interface used by the grid builders, traversal leaves, and collision detection.
#pragma once

#include <cstdint>
#include <type_traits>

#include "../app/scene.h"
#include "bezier_curve.h"
#include "point.h"
#include "primitive.h"
#include "sphere.h"
#include "triangle.h"

namespace happyray
{

	struct HitRecord
	{
		float	 t = 1e30f;
		Vec3f	 position;
		Vec3f	 normal;
		uint32_t material_id = 0;
	};

	// Closest-hit state carried through traversal; resolved into a HitRecord afterwards.
	struct ClosestHit
	{
		uint32_t prim = 0xffffffffu;
		float	 t = 0.0f, u = 0.0f, v = 0.0f;
	};

	// Closest-point state carried through a query; resolved into a ClosestPointRecord afterwards.
	struct ClosestPointHit
	{
		uint32_t prim			  = 0xffffffffu;
		float	 distance_squared = INFINITY;
		Vec3f	 point;
		float	 u = 0.0f, v = 0.0f;
	};

	// Normalized `d`, or +z when it is zero (the query point is a point primitive or on a curve axis).
	HAPPYRAY_HD inline Vec3f NormalizeOrZAxis(const Vec3f &d)
	{
		return LengthSquared(d) > 0.0f ? Normalize(d) : Vec3f(0.0f, 0.0f, 1.0f);
	}

	HAPPYRAY_HD inline Vec3f InterpolateTriangleNormal(const SceneView &scene, uint32_t prim, float u, float v)
	{
		const uint32_t i0 = scene.indices[prim * 3 + 0];
		const uint32_t i1 = scene.indices[prim * 3 + 1];
		const uint32_t i2 = scene.indices[prim * 3 + 2];
		return Normalize(scene.normals[i0] * (1.0f - u - v) + scene.normals[i1] * u + scene.normals[i2] * v);
	}

	// Selects the primitive type from SceneView::primitive_type at runtime.
	struct DynamicPrimitive
	{
	};

	template <typename P> struct PrimitiveTag
	{
		using type = P;
	};

	// Exact (conservative) primitive/box overlap via the primitive's Intersects(p, box).
	template <typename P> struct PrimitiveBoxIntersection
	{
		P primitive;

		HAPPYRAY_HD explicit PrimitiveBoxIntersection(const P &p) : primitive(p)
		{
		}

		HAPPYRAY_HD bool operator()(const BBox &box) const
		{
			return Intersects(primitive, box);
		}
	};

	// Specializations provide: Fetch, Intersect (t and up to two hit parameters), Normal, and
	// CellOverlap (constructed once per primitive, then tested against many boxes). All but
	// TriangleWoop also provide ClosestPoint (squared unsigned distance) and ClosestPointNormal.
	template <typename P> struct PrimitiveTraits;

	template <> struct PrimitiveTraits<Triangle>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kTriangle;
		using CellOverlap					 = TriangleBoxIntersection;

		HAPPYRAY_HD static Triangle Fetch(const SceneView &scene, uint32_t i)
		{
			return FetchTriangle(scene, i);
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const Triangle &p, float *t, float *u, float *v)
		{
			return IntersectRayTriangle(ray, p.vtx[0], p.vtx[1], p.vtx[2], t, u, v);
		}

		// Barycentric-interpolated shading normal.
		HAPPYRAY_HD static Vec3f Normal(const SceneView &scene, const Ray &, const ClosestHit &hit)
		{
			return InterpolateTriangleNormal(scene, hit.prim, hit.u, hit.v);
		}

		HAPPYRAY_HD static float
		ClosestPoint(const Triangle &p, const Vec3f &q, float, Vec3f *point, float *u, float *v)
		{
			return happyray::ClosestPoint(p, q, point, u, v);
		}

		HAPPYRAY_HD static Vec3f ClosestPointNormal(const SceneView &scene, const Vec3f &, const ClosestPointHit &hit)
		{
			return InterpolateTriangleNormal(scene, hit.prim, hit.u, hit.v);
		}
	};

	// Traversal-only view of a triangle scene through SceneView::triangle_woop (which must be set),
	// intersected with Woop's unit-triangle test: nearly the same hits as PrimitiveTraits<Triangle>
	// (they can differ on shared edges), fewer and wider loads. It has no CellOverlap, so grids are
	// built with Triangle and traced with TriangleWoop.
	template <> struct PrimitiveTraits<TriangleWoop>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kTriangle;

		HAPPYRAY_HD static TriangleWoop Fetch(const SceneView &scene, uint32_t i)
		{
			return LoadRO(scene.triangle_woop + i);
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const TriangleWoop &p, float *t, float *u, float *v)
		{
			return IntersectRayTriangleWoop(ray, p, t, u, v);
		}

		HAPPYRAY_HD static Vec3f Normal(const SceneView &scene, const Ray &ray, const ClosestHit &hit)
		{
			return PrimitiveTraits<Triangle>::Normal(scene, ray, hit);
		}
	};

	template <> struct PrimitiveTraits<Point>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kPoint;
		using CellOverlap					 = PrimitiveBoxIntersection<Point>;

		HAPPYRAY_HD static Point Fetch(const SceneView &scene, uint32_t i)
		{
			return scene.points[i];
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const Point &p, float *t, float *, float *)
		{
			return IntersectRayPoint(ray, p, t);
		}

		// Discs face the ray.
		HAPPYRAY_HD static Vec3f Normal(const SceneView &, const Ray &ray, const ClosestHit &)
		{
			return Normalize(ray.direction) * -1.0f;
		}

		HAPPYRAY_HD static float ClosestPoint(const Point &p, const Vec3f &q, float, Vec3f *point, float *, float *)
		{
			return happyray::ClosestPoint(p, q, point);
		}

		// Faces the query point.
		HAPPYRAY_HD static Vec3f ClosestPointNormal(const SceneView &, const Vec3f &q, const ClosestPointHit &hit)
		{
			return NormalizeOrZAxis(q - hit.point);
		}
	};

	template <> struct PrimitiveTraits<Sphere>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kSphere;
		using CellOverlap					 = PrimitiveBoxIntersection<Sphere>;

		HAPPYRAY_HD static Sphere Fetch(const SceneView &scene, uint32_t i)
		{
			return scene.spheres[i];
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const Sphere &p, float *t, float *, float *)
		{
			return IntersectRaySphere(ray, p, t);
		}

		HAPPYRAY_HD static Vec3f Normal(const SceneView &scene, const Ray &ray, const ClosestHit &hit)
		{
			return Normalize(ray.At(hit.t) - scene.spheres[hit.prim].vtx[0]);
		}

		HAPPYRAY_HD static float ClosestPoint(const Sphere &p, const Vec3f &q, float, Vec3f *point, float *, float *)
		{
			return happyray::ClosestPoint(p, q, point);
		}

		// Radial, also for query points inside the sphere.
		HAPPYRAY_HD static Vec3f ClosestPointNormal(const SceneView &scene, const Vec3f &q, const ClosestPointHit &hit)
		{
			return NormalizeOrZAxis(q - scene.spheres[hit.prim].vtx[0]);
		}
	};

	template <> struct PrimitiveTraits<BezierCurve>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kBezierCurve;
		using CellOverlap					 = PrimitiveBoxIntersection<BezierCurve>;

		HAPPYRAY_HD static BezierCurve Fetch(const SceneView &scene, uint32_t i)
		{
			return scene.curves[i];
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const BezierCurve &p, float *t, float *u, float *)
		{
			return IntersectRayBezierCurve(ray, p, t, u);
		}

		// Points away from the curve axis at the hit parameter.
		HAPPYRAY_HD static Vec3f Normal(const SceneView &scene, const Ray &ray, const ClosestHit &hit)
		{
			const Vec3f offset = ray.At(hit.t) - EvaluateBezierCurve(scene.curves[hit.prim], hit.u);
			return LengthSquared(offset) > 0.0f ? Normalize(offset) : Normalize(ray.direction) * -1.0f;
		}

		HAPPYRAY_HD static float
		ClosestPoint(const BezierCurve &p, const Vec3f &q, float bound_squared, Vec3f *point, float *u, float *)
		{
			return happyray::ClosestPoint(p, q, point, u, bound_squared);
		}

		// Away from the curve axis, also for query points inside the tube.
		HAPPYRAY_HD static Vec3f ClosestPointNormal(const SceneView &scene, const Vec3f &q, const ClosestPointHit &hit)
		{
			return NormalizeOrZAxis(q - EvaluateBezierCurve(scene.curves[hit.prim], hit.u));
		}
	};

	// Solid axis-aligned boxes; the hit face's axis travels in ClosestHit::u.
	template <> struct PrimitiveTraits<BBox>
	{
		static constexpr PrimitiveType kType = PrimitiveType::kBox;
		using CellOverlap					 = PrimitiveBoxIntersection<BBox>;

		HAPPYRAY_HD static BBox Fetch(const SceneView &scene, uint32_t i)
		{
			return scene.boxes[i];
		}

		HAPPYRAY_HD static bool Intersect(const Ray &ray, const BBox &p, float *t, float *u, float *)
		{
			int axis;
			if (!IntersectRayBoxSurface(ray, p, t, &axis))
				return false;
			*u = static_cast<float>(axis);
			return true;
		}

		// The hit face's normal, facing the ray (hits from inside see the face from behind).
		HAPPYRAY_HD static Vec3f Normal(const SceneView &, const Ray &ray, const ClosestHit &hit)
		{
			const int	axis	  = static_cast<int>(hit.u);
			const float component = (&ray.direction.x)[axis];
			Vec3f		normal(0.0f, 0.0f, 0.0f);
			(&normal.x)[axis] = component > 0.0f ? -1.0f : 1.0f;
			return normal;
		}

		HAPPYRAY_HD static float ClosestPoint(const BBox &p, const Vec3f &q, float, Vec3f *point, float *u, float *v)
		{
			return happyray::ClosestPoint(p, q, point, u, v);
		}

		// The face normal chosen by ClosestPoint (axis in u, sign in v).
		HAPPYRAY_HD static Vec3f ClosestPointNormal(const SceneView &, const Vec3f &, const ClosestPointHit &hit)
		{
			Vec3f normal(0.0f, 0.0f, 0.0f);
			(&normal.x)[static_cast<int>(hit.u)] = hit.v;
			return normal;
		}
	};

	template <typename P> HAPPYRAY_HD inline BBox ComputePrimitiveBBox(const SceneView &scene, uint32_t i)
	{
		return BBoxExtractor<P>::Get(PrimitiveTraits<P>::Fetch(scene, i));
	}

	// Calls `fn(PrimitiveTag<P>{})` for the primitive type `type`.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
	template <typename Fn> HAPPYRAY_HD inline decltype(auto) DispatchPrimitive(PrimitiveType type, Fn &&fn)
	{
		switch (type)
		{
		case PrimitiveType::kPoint:
			return fn(PrimitiveTag<Point>{});
		case PrimitiveType::kSphere:
			return fn(PrimitiveTag<Sphere>{});
		case PrimitiveType::kBezierCurve:
			return fn(PrimitiveTag<BezierCurve>{});
		case PrimitiveType::kBox:
			return fn(PrimitiveTag<BBox>{});
		default:
			return fn(PrimitiveTag<Triangle>{});
		}
	}

} // namespace happyray
