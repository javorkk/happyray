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
	// CellOverlap (constructed once per primitive, then tested against many boxes).
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
			const uint32_t i0 = scene.indices[hit.prim * 3 + 0];
			const uint32_t i1 = scene.indices[hit.prim * 3 + 1];
			const uint32_t i2 = scene.indices[hit.prim * 3 + 2];
			return Normalize(scene.normals[i0] * (1.0f - hit.u - hit.v) + scene.normals[i1] * hit.u +
							 scene.normals[i2] * hit.v);
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
		default:
			return fn(PrimitiveTag<Triangle>{});
		}
	}

} // namespace happyray
