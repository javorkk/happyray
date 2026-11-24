// Dependency-free unit tests for src/core; returns non-zero on the first failure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

#include "../src/app/animation.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/core/bbox.h"
#include "../src/core/bezier_curve.h"
#include "../src/core/mat4.h"
#include "../src/core/mesh.h"
#include "../src/core/point.h"
#include "../src/core/quaternion.h"
#include "../src/core/random.h"
#include "../src/core/ray.h"
#include "../src/core/sphere.h"
#include "../src/core/triangle.h"
#include "../src/core/vec.h"

namespace
{

	static_assert(happyray::Triangle::kNumVertices == 3);
	static_assert(std::is_base_of<happyray::Primitive<1>, happyray::Point>::value);
	static_assert(std::is_base_of<happyray::Primitive<1>, happyray::Sphere>::value);
	static_assert(std::is_base_of<happyray::Primitive<4>, happyray::BezierCurve>::value);

	int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                                                           \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
		{                                                                                                              \
			std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__);                                   \
			++g_failures;                                                                                              \
		}                                                                                                              \
	} while (0)

	bool NearlyEqual(float a, float b, float eps = 1e-4f)
	{
		return std::fabs(a - b) < eps;
	}
	bool NearlyEqual(const happyray::Vec3f &a, const happyray::Vec3f &b, float eps = 1e-4f)
	{
		return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps);
	}

	void CheckSubdivisionLimit(const happyray::Scene &scene)
	{
		const auto	 bounds		   = scene.WorldBounds();
		const double limit_squared = 0.01 * happyray::SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
		for (const auto &mesh : scene.meshes)
		{
			HAPPYRAY_CHECK(mesh.positions.size() == mesh.attributes.size());
			for (size_t i = 0; i < mesh.indices.size(); i += 3)
			{
				for (size_t edge = 0; edge < 3; ++edge)
					HAPPYRAY_CHECK(happyray::SceneEdgeLengthSquared(mesh.positions[mesh.indices[i + edge]],
																	mesh.positions[mesh.indices[i + (edge + 1) % 3]]) <=
								   limit_squared);
			}
		}
	}

	void TestTriangleSubdivision()
	{
		using namespace happyray;
		Scene scene;
		SubdivideLargeTriangles(&scene);
		HAPPYRAY_CHECK(scene.meshes.empty());
		Mesh mesh;
		mesh.positions	= {Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(0, 1, 0)};
		mesh.attributes = {
			{Vec3f(0, 0, 1), Vec2f(0, 0)},
			{Vec3f(0, 0, 3), Vec2f(1, 0)},
			{Vec3f(0, 0, 2), Vec2f(0, 1)}
		 };
		mesh.indices	  = {0, 1, 2};
		mesh.material_ids = {7};
		scene.meshes.push_back(mesh);
		// Exercise a second mesh and a degenerate long triangle with no material array.
		mesh.indices = {0, 1, 1};
		mesh.material_ids.clear();
		scene.meshes.push_back(mesh);
		const BBox bounds = scene.WorldBounds();
		SubdivideLargeTriangles(&scene);
		CheckSubdivisionLimit(scene);
		HAPPYRAY_CHECK(NearlyEqual(scene.WorldBounds().min_bound, bounds.min_bound));
		HAPPYRAY_CHECK(NearlyEqual(scene.WorldBounds().max_bound, bounds.max_bound));
		const Mesh &result = scene.meshes.front();
		HAPPYRAY_CHECK(result.TriangleCount() > 1);
		HAPPYRAY_CHECK(result.material_ids.size() == result.TriangleCount());
		double area = 0;
		for (size_t tri = 0; tri < result.indices.size(); tri += 3)
		{
			const Vec3f a	   = result.positions[result.indices[tri]];
			const Vec3f b	   = result.positions[result.indices[tri + 1]];
			const Vec3f c	   = result.positions[result.indices[tri + 2]];
			const Vec3f normal = Cross(b - a, c - a);
			HAPPYRAY_CHECK(normal.z > 0);
			area += Length(normal) * 0.5;
			HAPPYRAY_CHECK(result.material_ids[tri / 3] == 7);
		}
		HAPPYRAY_CHECK(std::fabs(area - 0.5) < 1e-6);
		for (size_t i = 0; i < result.positions.size(); ++i)
		{
			const Vec3f			   &position   = result.positions[i];
			const VertexAttributes &attributes = result.attributes[i];
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.x, position.x));
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.y, position.y));
			HAPPYRAY_CHECK(NearlyEqual(attributes.normal.z, 1 + 2 * position.x + position.y));
		}
		HAPPYRAY_CHECK(scene.meshes[1].material_ids.empty());
		const uint32_t count	= scene.TriangleCount();
		const size_t   vertices = result.positions.size();
		SubdivideLargeTriangles(&scene);
		HAPPYRAY_CHECK(scene.TriangleCount() == count);
		HAPPYRAY_CHECK(scene.meshes.front().positions.size() == vertices);
		HAPPYRAY_CHECK(scene.meshes.front().attributes.size() == vertices);

		Scene point;
		mesh.positions	= {Vec3f(1, 2, 3)};
		mesh.attributes = {
			{Vec3f(), Vec2f()}
		  };
		mesh.indices = {0, 0, 0};
		point.meshes.push_back(mesh);
		SubdivideLargeTriangles(&point);
		HAPPYRAY_CHECK(point.TriangleCount() == 1);
	}

	void TestVecMath()
	{
		using namespace happyray;
		const Vec3f a(1.0f, 0.0f, 0.0f);
		const Vec3f b(0.0f, 1.0f, 0.0f);

		HAPPYRAY_CHECK(NearlyEqual(Dot(a, b), 0.0f));
		HAPPYRAY_CHECK(NearlyEqual(Cross(a, b), Vec3f(0.0f, 0.0f, 1.0f)));
		HAPPYRAY_CHECK(NearlyEqual(Length(Vec3f(3.0f, 4.0f, 0.0f)), 5.0f));
		HAPPYRAY_CHECK(NearlyEqual(Normalize(Vec3f(5.0f, 0.0f, 0.0f)), Vec3f(1.0f, 0.0f, 0.0f)));
		HAPPYRAY_CHECK(NearlyEqual(Lerp(a, b, 0.5f), Vec3f(0.5f, 0.5f, 0.0f)));
		const Vec2i half2 = Vec2i(6, 8) / 2;
		const Vec3i half3 = Vec3i(6, 8, 10) / 2;
		HAPPYRAY_CHECK(half2.x == 3 && half2.y == 4);
		HAPPYRAY_CHECK(half3.x == 3 && half3.y == 4 && half3.z == 5);

		const Vec3f incident(1.0f, -1.0f, 0.0f);
		const Vec3f normal(0.0f, 1.0f, 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(Reflect(incident, normal), Vec3f(1.0f, 1.0f, 0.0f)));
	}

	void TestMat4()
	{
		using namespace happyray;
		const Mat4 t = Mat4::Translation(Vec3f(1.0f, 2.0f, 3.0f));
		HAPPYRAY_CHECK(NearlyEqual(TransformPoint(t, Vec3f(0.0f, 0.0f, 0.0f)), Vec3f(1.0f, 2.0f, 3.0f)));
		HAPPYRAY_CHECK(NearlyEqual(TransformVector(t, Vec3f(1.0f, 0.0f, 0.0f)), Vec3f(1.0f, 0.0f, 0.0f)));

		const Mat4 s = Mat4::Scale(Vec3f(2.0f, 2.0f, 2.0f));
		HAPPYRAY_CHECK(NearlyEqual(TransformPoint(s, Vec3f(1.0f, 1.0f, 1.0f)), Vec3f(2.0f, 2.0f, 2.0f)));

		const Mat4 combined = t * s;
		HAPPYRAY_CHECK(NearlyEqual(TransformPoint(combined, Vec3f(1.0f, 1.0f, 1.0f)), Vec3f(3.0f, 4.0f, 5.0f)));

		const Mat4 inv_t = Inverse(t);
		HAPPYRAY_CHECK(NearlyEqual(TransformPoint(inv_t, Vec3f(1.0f, 2.0f, 3.0f)), Vec3f(0.0f, 0.0f, 0.0f)));

		const Mat4 view = Mat4::LookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
		// The camera looks down -Z, so the origin 5 units ahead maps to (0, 0, -5).
		HAPPYRAY_CHECK(NearlyEqual(TransformPoint(view, Vec3f(0.0f, 0.0f, 0.0f)), Vec3f(0.0f, 0.0f, -5.0f)));
	}

	void TestBBox()
	{
		using namespace happyray;
		BBox box;
		box.Grow(Vec3f(-1.0f, -1.0f, -1.0f));
		box.Grow(Vec3f(1.0f, 1.0f, 1.0f));
		HAPPYRAY_CHECK(!box.IsEmpty());
		HAPPYRAY_CHECK(NearlyEqual(box.Center(), Vec3f(0.0f, 0.0f, 0.0f)));

		const Ray ray(Vec3f(-5.0f, 0.0f, 0.0f), Vec3f(1.0f, 0.0f, 0.0f));
		float	  t_near = 0.0f, t_far = 0.0f;
		HAPPYRAY_CHECK(IntersectRayBBox(ray, box, &t_near, &t_far));
		HAPPYRAY_CHECK(NearlyEqual(t_near, 4.0f));
		HAPPYRAY_CHECK(NearlyEqual(t_far, 6.0f));

		const Ray miss_ray(Vec3f(-5.0f, 5.0f, 0.0f), Vec3f(1.0f, 0.0f, 0.0f));
		HAPPYRAY_CHECK(!IntersectRayBBox(miss_ray, box, &t_near, &t_far));
	}

	void TestBBoxExtractor()
	{
		using namespace happyray;
		const Triangle triangle{
			{Vec3f(-2, 3, 1), Vec3f(4, -1, 2), Vec3f(0, 2, -3)}
		   };
		const BBox triangle_bounds = BBoxExtractor<Triangle>::Get(triangle);
		HAPPYRAY_CHECK(NearlyEqual(triangle_bounds.min_bound, Vec3f(-2, -1, -3)));
		HAPPYRAY_CHECK(NearlyEqual(triangle_bounds.max_bound, Vec3f(4, 3, 2)));

		const Primitive<4> quad{
			{triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], Vec3f(5, 6, 7)}
		   };
		const BBox quad_bounds = BBoxExtractor<Primitive<4>>::Get(quad);
		HAPPYRAY_CHECK(NearlyEqual(quad_bounds.min_bound, Vec3f(-2, -1, -3)));
		HAPPYRAY_CHECK(NearlyEqual(quad_bounds.max_bound, Vec3f(5, 6, 7)));
	}

	void TestAdditionalPrimitiveBounds()
	{
		using namespace happyray;
		const Point point{Vec3f(1, -2, 3)};
		const BBox	point_bounds = BBoxExtractor<Point>::Get(point);
		HAPPYRAY_CHECK(NearlyEqual(point_bounds.min_bound, point.vtx[0]));
		HAPPYRAY_CHECK(NearlyEqual(point_bounds.max_bound, point.vtx[0]));

		const Sphere sphere{Vec3f(1, 2, 3), 2.0f};
		const BBox	 sphere_bounds = BBoxExtractor<Sphere>::Get(sphere);
		HAPPYRAY_CHECK(NearlyEqual(sphere_bounds.min_bound, Vec3f(-1, 0, 1)));
		HAPPYRAY_CHECK(NearlyEqual(sphere_bounds.max_bound, Vec3f(3, 4, 5)));

		const BezierCurve curve(Vec3f(-2, 0, 1), Vec3f(1, 3, -1), Vec3f(4, -2, 2), Vec3f(0, 1, 5), 0.5f);
		const BBox		  curve_bounds = BBoxExtractor<BezierCurve>::Get(curve);
		HAPPYRAY_CHECK(NearlyEqual(curve_bounds.min_bound, Vec3f(-2.502f, -2.502f, -1.502f)));
		HAPPYRAY_CHECK(NearlyEqual(curve_bounds.max_bound, Vec3f(4.502f, 3.502f, 5.502f)));

		const BBox box(Vec3f(-1, 0, 2), Vec3f(3, 1, 4));
		HAPPYRAY_CHECK(NearlyEqual(BBoxExtractor<BBox>::Get(box).min_bound, box.min_bound));
		HAPPYRAY_CHECK(NearlyEqual(BBoxExtractor<BBox>::Get(box).max_bound, box.max_bound));
		HAPPYRAY_CHECK(std::string(PrimitiveTypeName(PrimitiveType::kBox)) == "box");
	}

	void TestPrimitiveIntersections()
	{
		using namespace happyray;
		float	  t = 0.0f;
		const Ray x_ray(Vec3f(0, 0, 0), Vec3f(1, 0, 0));
		HAPPYRAY_CHECK(IntersectRayPoint(x_ray, Point{Vec3f(2, 5e-5f, 0)}, &t));
		HAPPYRAY_CHECK(NearlyEqual(t, 2.0f));
		HAPPYRAY_CHECK(!IntersectRayPoint(x_ray, Point{Vec3f(2, 0.01f, 0)}, &t));

		HAPPYRAY_CHECK(IntersectRaySphere(x_ray, Sphere{Vec3f(4, 0, 0), 1.0f}, &t));
		HAPPYRAY_CHECK(NearlyEqual(t, 3.0f));
		const Ray tangent_ray(Vec3f(0, 1, 0), Vec3f(1, 0, 0));
		HAPPYRAY_CHECK(IntersectRaySphere(tangent_ray, Sphere{Vec3f(4, 0, 0), 1.0f}, &t));
		HAPPYRAY_CHECK(NearlyEqual(t, 4.0f));
		HAPPYRAY_CHECK(!IntersectRaySphere(x_ray, Sphere{Vec3f(4, 0, 0), -1.0f}, &t));

		const BezierCurve line_curve(Vec3f(4, 0, 0), Vec3f(5, 0, 0), Vec3f(6, 0, 0), Vec3f(7, 0, 0), 0.5f);
		const Ray		  tube_ray(Vec3f(5.5f, 0, -2), Vec3f(0, 0, 1));
		HAPPYRAY_CHECK(IntersectRayBezierCurve(tube_ray, line_curve, &t));
		HAPPYRAY_CHECK(t > 1.49f && t < 1.51f);
		const Ray tube_miss(Vec3f(5.5f, 0.7f, -2), Vec3f(0, 0, 1));
		HAPPYRAY_CHECK(!IntersectRayBezierCurve(tube_miss, line_curve, &t));
		const BezierCurve curved_tube(Vec3f(0, 0, 0), Vec3f(0, 1, 0), Vec3f(1, 1, 0), Vec3f(1, 0, 0), 0.05f);
		const Ray		  curved_tube_ray(Vec3f(0.5f, 0.75f, -1), Vec3f(0, 0, 1));
		HAPPYRAY_CHECK(IntersectRayBezierCurve(curved_tube_ray, curved_tube, &t));

		const BBox box(Vec3f(-1, -1, -1), Vec3f(1, 1, 1));
		HAPPYRAY_CHECK(Intersects(Point{Vec3f(1, 0, 0)}, box));
		HAPPYRAY_CHECK(!Intersects(Point{Vec3f(1.01f, 0, 0)}, box));
		HAPPYRAY_CHECK(Intersects(Sphere{Vec3f(1.5f, 0, 0), 0.5f}, box));
		HAPPYRAY_CHECK(!Intersects(Sphere{Vec3f(1.51f, 0, 0), 0.5f}, box));
		HAPPYRAY_CHECK(Intersects(line_curve, BBox(Vec3f(5, -0.5f, -0.5f), Vec3f(6, 0.5f, 0.5f))));

		// Point radius widens the hit disc and the conservative box test.
		HAPPYRAY_CHECK(IntersectRayPoint(x_ray, Point(Vec3f(2, 0.4f, 0), 0.5f), &t));
		HAPPYRAY_CHECK(!IntersectRayPoint(x_ray, Point(Vec3f(2, 0.6f, 0), 0.5f), &t));
		HAPPYRAY_CHECK(Intersects(Point(Vec3f(1.4f, 0, 0), 0.5f), box));
		HAPPYRAY_CHECK(
			NearlyEqual(BBoxExtractor<Point>::Get(Point(Vec3f(1, 2, 3), 0.5f)).max_bound, Vec3f(1.5f, 2.5f, 3.5f)));

		// Curve hits report their curve parameter; normals derive from it.
		float u = -1.0f;
		HAPPYRAY_CHECK(IntersectRayBezierCurve(tube_ray, line_curve, &t, &u));
		HAPPYRAY_CHECK(std::fabs(u - 0.5f) < 0.01f);
		HAPPYRAY_CHECK(NearlyEqual(EvaluateBezierCurve(line_curve, 0.5f), Vec3f(5.5f, 0, 0)));
		HAPPYRAY_CHECK(NearlyEqual(EvaluateBezierCurve(curved_tube, 0.5f), Vec3f(0.5f, 0.75f, 0)));
		// Boxes inside the curve's hull but away from the tube don't overlap it.
		HAPPYRAY_CHECK(!Intersects(curved_tube, BBox(Vec3f(0.45f, 0.15f, -0.05f), Vec3f(0.55f, 0.25f, 0.05f))));
		HAPPYRAY_CHECK(Intersects(curved_tube, BBox(Vec3f(0.45f, 0.7f, -0.05f), Vec3f(0.55f, 0.8f, 0.05f))));
		HAPPYRAY_CHECK(!Intersects(curved_tube, BBox(Vec3f(0.45f, 0.85f, -0.05f), Vec3f(0.55f, 0.95f, 0.05f))));
		// Rays starting inside a point's disc (AO rays) don't self-occlude.
		HAPPYRAY_CHECK(
			!IntersectRayPoint(Ray(Vec3f(2, 0.05f, 0.001f), Vec3f(0.3f, -0.1f, 1)), Point(Vec3f(2, 0, 0), 0.1f), &t));
		// Cells smaller than the ray tolerance still cover the ray test's capsule proxies.
		const BezierCurve flat_curve(Vec3f(0, 0.7e-3f, 5e-4f),
									 Vec3f(0.01f, 1.65e-3f, 5e-4f),
									 Vec3f(0.02f, 1.65e-3f, 5e-4f),
									 Vec3f(0.03f, 0.7e-3f, 5e-4f),
									 0.0f);
		HAPPYRAY_CHECK(IntersectRayBezierCurve(Ray(Vec3f(0.015f, -1.25e-3f, -1), Vec3f(0, 0, 1)), flat_curve, &t));
		HAPPYRAY_CHECK(Intersects(flat_curve,
								  BBox(Vec3f(0.014995f, -0.002005f, -5e-6f), Vec3f(0.016005f, -0.000995f, 0.001005f))));

		const Triangle flat{
			{Vec3f(0, 0, 0), Vec3f(2, 0, 0), Vec3f(0, 2, 0)}
		};
		const Triangle crossing{
			{Vec3f(0.5f, -1, -1), Vec3f(0.5f, 2, -1), Vec3f(0.5f, 0, 1)}
		};
		const Triangle separated{
			{Vec3f(3, 0, 0), Vec3f(4, 0, 0), Vec3f(3, 1, 0)}
		};
		const Triangle coplanar_overlap{
			{Vec3f(0.5f, 0.5f, 0), Vec3f(2, 0.5f, 0), Vec3f(0.5f, 2, 0)}
		};
		const Triangle coplanar_separated{
			{Vec3f(3, 3, 0), Vec3f(4, 3, 0), Vec3f(3, 4, 0)}
		};
		const Triangle degenerate_inside{
			{Vec3f(0.5f, 0.5f, 0), Vec3f(0.5f, 0.5f, 0), Vec3f(0.5f, 0.5f, 0)}
		  };
		const Triangle degenerate_outside{
			{Vec3f(3, 3, 0), Vec3f(3, 3, 0), Vec3f(3, 3, 0)}
		};
		// Solid boxes: entry face from outside, exit face from inside, touching boxes overlap.
		const BBox solid(Vec3f(1, -1, -1), Vec3f(3, 1, 1));
		int		   axis = -1;
		HAPPYRAY_CHECK(IntersectRayBoxSurface(x_ray, solid, &t, &axis));
		HAPPYRAY_CHECK(NearlyEqual(t, 1.0f) && axis == 0);
		HAPPYRAY_CHECK(IntersectRayBoxSurface(Ray(Vec3f(2, 0, 0), Vec3f(0, 0.5f, 0)), solid, &t, &axis));
		HAPPYRAY_CHECK(NearlyEqual(t, 2.0f) && axis == 1);
		HAPPYRAY_CHECK(!IntersectRayBoxSurface(Ray(Vec3f(4, 0, 0), Vec3f(1, 0, 0)), solid, &t, &axis));
		HAPPYRAY_CHECK(!IntersectRayBoxSurface(Ray(Vec3f(0, 2, 0), Vec3f(1, 0, 0)), solid, &t, &axis));
		HAPPYRAY_CHECK(!IntersectRayBoxSurface(Ray(Vec3f(0, 0, 0), Vec3f(1, 0, 0), 1e-4f, 0.5f), solid, &t, &axis));
		HAPPYRAY_CHECK(!IntersectRayBoxSurface(x_ray, BBox(), &t, &axis));
		HAPPYRAY_CHECK(Intersects(solid, BBox(Vec3f(3, 1, 1), Vec3f(4, 2, 2))));
		HAPPYRAY_CHECK(!Intersects(solid, BBox(Vec3f(3.01f, 0, 0), Vec3f(4, 1, 1))));
		HAPPYRAY_CHECK(!Intersects(BBox(), solid));

		HAPPYRAY_CHECK(IntersectTriangles(flat, crossing));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, separated));
		HAPPYRAY_CHECK(IntersectTriangles(flat, coplanar_overlap));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, coplanar_separated));
		HAPPYRAY_CHECK(IntersectTriangles(flat, degenerate_inside));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, degenerate_outside));
	}

	// The ray test's capsules, by splitting segments on an explicit stack (de Casteljau).
	bool IntersectRayBezierCurveReference(const happyray::Ray &ray, const happyray::BezierCurve &curve, float *out_t)
	{
		using namespace happyray;
		detail::BezierSegment stack[kBezierMaxSubdivisionDepth + 1];
		stack[0]			  = detail::WholeBezierSegment(curve);
		int			size	  = 1;
		bool		found	  = false;
		float		closest_t = ray.t_max;
		const float tol2	  = kBezierRayTolerance * kBezierRayTolerance;
		while (size > 0)
		{
			const detail::BezierSegment segment			 = stack[--size];
			const float					flatness_squared = detail::BezierFlatnessSquared(segment);
			if (flatness_squared > tol2 && segment.depth < kBezierMaxSubdivisionDepth)
			{
				detail::SplitBezierSegment(segment, &stack[size + 1], &stack[size]);
				size += 2;
				continue;
			}
			float t, s;
			if (detail::IntersectRayCapsule(ray,
											segment.control[0],
											segment.control[3],
											curve.radius + std::sqrt(flatness_squared) + kBezierRayTolerance,
											&t,
											&s) &&
				(!found || t < closest_t))
			{
				closest_t = t;
				found	  = true;
			}
		}
		*out_t = closest_t;
		return found;
	}

	// The stackless subdivision walk finds the same capsules as splitting on a stack.
	void TestClosestPointPrimitives()
	{
		using namespace happyray;
		Vec3f point;
		float u = -1.0f, v = -1.0f;

		Triangle triangle;
		triangle.vtx[0] = Vec3f(0, 0, 0);
		triangle.vtx[1] = Vec3f(1, 0, 0);
		triangle.vtx[2] = Vec3f(0, 1, 0);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(triangle, Vec3f(0.25f, 0.25f, 2), &point, &u, &v), 4.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(0.25f, 0.25f, 0)) && NearlyEqual(u, 0.25f) && NearlyEqual(v, 0.25f));
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(triangle, Vec3f(-1, -1, 0), &point, &u, &v), 2.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(0, 0, 0)) && u == 0.0f && v == 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(triangle, Vec3f(2, -1, 0), &point, &u, &v), 2.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1, 0, 0)) && u == 1.0f && v == 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(triangle, Vec3f(1, 1, 0), &point, &u, &v), 0.5f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(0.5f, 0.5f, 0)) && NearlyEqual(u, 0.5f) && NearlyEqual(v, 0.5f));
		Triangle collinear;
		collinear.vtx[0] = Vec3f(0, 0, 0);
		collinear.vtx[1] = Vec3f(1, 0, 0);
		collinear.vtx[2] = Vec3f(2, 0, 0);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(collinear, Vec3f(1.5f, 1, 0), &point, &u, &v), 1.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1.5f, 0, 0)));

		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(Point(Vec3f(1, 2, 3), 0.5f), Vec3f(1, 2, 5), &point), 4.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1, 2, 3)));

		const Sphere sphere(Vec3f(1, 0, 0), 0.5f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(sphere, Vec3f(3, 0, 0), &point), 1.5f * 1.5f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1.5f, 0, 0)));
		HAPPYRAY_CHECK(ClosestPoint(sphere, Vec3f(1.2f, 0, 0), &point) == 0.0f &&
					   NearlyEqual(point, Vec3f(1.2f, 0, 0)));
		HAPPYRAY_CHECK(ClosestPoint(Sphere(Vec3f(0), -1.0f), Vec3f(3, 0, 0), &point) == INFINITY);

		const BBox box(Vec3f(-1, -1, -1), Vec3f(1, 1, 1));
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(box, Vec3f(3, 0.5f, 0), &point, &u, &v), 4.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1, 0.5f, 0)) && u == 0.0f && v == 1.0f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(box, Vec3f(2, -3, 2), &point, &u, &v), 6.0f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(1, -1, 1)) && u == 1.0f && v == -1.0f);
		HAPPYRAY_CHECK(ClosestPoint(box, Vec3f(0.2f, 0.1f, -0.9f), &point, &u, &v) == 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(0.2f, 0.1f, -0.9f)) && u == 2.0f && v == -1.0f);
		HAPPYRAY_CHECK(DistanceSquaredToBBox(Vec3f(0.5f), box) == 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(DistanceSquaredToBBox(Vec3f(2, 3, 0), box), 5.0f));

		const BezierCurve line_curve(Vec3f(4, 0, 0), Vec3f(5, 0, 0), Vec3f(6, 0, 0), Vec3f(7, 0, 0), 0.5f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(line_curve, Vec3f(5.5f, 2, 0), &point, &u), 1.5f * 1.5f));
		HAPPYRAY_CHECK(NearlyEqual(point, Vec3f(5.5f, 0.5f, 0)) && NearlyEqual(u, 0.5f));
		HAPPYRAY_CHECK(ClosestPoint(line_curve, Vec3f(6, 0.3f, 0), &point, &u) == 0.0f);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(line_curve, Vec3f(9, 0, 0), &point, &u), 1.5f * 1.5f) && u == 1.0f);
		HAPPYRAY_CHECK(ClosestPoint(line_curve, Vec3f(5.5f, 2, 0), &point, &u, 1.0f) == INFINITY);
		HAPPYRAY_CHECK(NearlyEqual(ClosestPoint(line_curve, Vec3f(5.5f, 2, 0), &point, &u, 1.5f * 1.5f), 1.5f * 1.5f));
	}

	// No dense sample of a random triangle or curve may beat ClosestPoint, which must report a point
	// on the primitive within sampling error of the best sample.
	void TestClosestPointMatchesSampling()
	{
		using namespace happyray;
		Rng	 rng(11, 5);
		auto random_point = [&](float scale)
		{ return (Vec3f(rng.NextFloat(), rng.NextFloat(), rng.NextFloat()) - Vec3f(0.5f)) * scale; };
		int failures = 0;
		for (int c = 0; c < 200; ++c)
		{
			Triangle triangle;
			for (int i = 0; i < 3; ++i)
				triangle.vtx[i] = random_point(2.0f);
			const Vec3f query = random_point(4.0f);
			Vec3f		point;
			float		u, v;
			const float distance	= std::sqrt(ClosestPoint(triangle, query, &point, &u, &v));
			const Vec3f on_triangle = triangle.vtx[0] * (1.0f - u - v) + triangle.vtx[1] * u + triangle.vtx[2] * v;
			float		sampled		= INFINITY;
			for (int i = 0; i <= 256; ++i)
				for (int j = 0; i + j <= 256; ++j)
				{
					const float su = i / 256.0f, sv = j / 256.0f;
					const Vec3f sample =
						triangle.vtx[0] * (1.0f - su - sv) + triangle.vtx[1] * su + triangle.vtx[2] * sv;
					sampled = std::fmin(sampled, Length(query - sample));
				}
			failures += !(u >= -1e-5f && v >= -1e-5f && u + v <= 1.0f + 1e-5f && NearlyEqual(point, on_triangle) &&
						  distance <= sampled + 1e-5f && distance >= sampled - 1e-2f);
		}
		HAPPYRAY_CHECK(failures == 0);

		failures = 0;
		for (int c = 0; c < 200; ++c)
		{
			const float		  bend = c % 3 == 0 ? 0.01f : 1.5f;
			const Vec3f		  p0 = random_point(2.0f), p3 = random_point(2.0f);
			const BezierCurve curve(p0,
									Lerp(p0, p3, 1.0f / 3.0f) + random_point(bend),
									Lerp(p0, p3, 2.0f / 3.0f) + random_point(bend),
									p3,
									c % 4 == 0 ? 0.0f : 0.05f * rng.NextFloat());
			const Vec3f		  query = random_point(4.0f);
			Vec3f			  point;
			float			  u;
			const float		  distance = std::sqrt(ClosestPoint(curve, query, &point, &u));
			float			  sampled  = INFINITY;
			for (int i = 0; i <= 8192; ++i)
				sampled = std::fmin(sampled, Length(query - EvaluateBezierCurve(curve, i / 8192.0f)));
			sampled						 = std::fmax(sampled - curve.radius, 0.0f);
			const float axis_distance	 = Length(query - EvaluateBezierCurve(curve, u));
			const float surface_distance = std::fmax(axis_distance - curve.radius, 0.0f);
			failures += !(u >= 0.0f && u <= 1.0f && NearlyEqual(distance, surface_distance) &&
						  NearlyEqual(Length(query - point), distance) && distance <= sampled + 1e-5f &&
						  distance >= sampled - 1e-3f);
		}
		HAPPYRAY_CHECK(failures == 0);
	}

	void TestBezierCurveMatchesStackSubdivision()
	{
		using namespace happyray;
		Rng	 rng(7, 3);
		auto random_point = [&](float scale)
		{ return Vec3f(rng.NextFloat(), rng.NextFloat(), rng.NextFloat()) * scale; };
		int rays = 0, hits = 0, mismatches = 0;
		for (int c = 0; c < 400; ++c)
		{
			// From nearly straight to strongly bent curves.
			const float		  bend = c % 4 == 0 ? 0.002f : (c % 4 == 1 ? 0.05f : 1.0f);
			const Vec3f		  p0 = random_point(1.0f), p3 = p0 + Vec3f(0.3f, 0.1f, -0.2f) + random_point(0.2f);
			const BezierCurve curve(p0,
									Lerp(p0, p3, 1.0f / 3.0f) + random_point(bend),
									Lerp(p0, p3, 2.0f / 3.0f) + random_point(bend),
									p3,
									0.005f + 0.05f * rng.NextFloat());
			for (int r = 0; r < 64; ++r, ++rays)
			{
				const Vec3f target = EvaluateBezierCurve(curve, rng.NextFloat()) + random_point(5.0f * curve.radius) -
									 Vec3f(2.5f * curve.radius);
				const Vec3f origin = target + (random_point(2.0f) - Vec3f(1.0f)) * 3.0f;
				const Ray	ray(origin, target - origin);
				float		t = 0.0f, t_reference = 0.0f, u = -1.0f;
				const bool	hit			  = IntersectRayBezierCurve(ray, curve, &t, &u);
				const bool	hit_reference = IntersectRayBezierCurveReference(ray, curve, &t_reference);
				hits += hit;
				// Rounding can flip which capsules grazing rays touch.
				if (hit != hit_reference || (hit && std::fabs(t - t_reference) > 1e-3f * std::fmax(1.0f, t_reference)))
				{
					++mismatches;
					continue;
				}
				if (!hit)
					continue;
				HAPPYRAY_CHECK(u >= 0.0f && u <= 1.0f);
			}
		}
		HAPPYRAY_CHECK(hits > rays / 4 && hits < rays * 3 / 4);
		HAPPYRAY_CHECK(mismatches * 1000 <= rays);
	}

	// Rays aimed just beside shared edges of small, centered triangle fans must not slip through.
	void TestTriangleWoopCenteredGaps()
	{
		using namespace happyray;
		const Vec3f offset(-78.0f, -112.0f, -18.0f);
		const Vec3f eye		= Vec3f(95.9467f, 119.6175f, 45.7446f) + offset;
		const int	kSpokes = 8;
		Rng			rng(7, 3);
		int			misses = 0, rays = 0;
		for (int fan = 0; fan < 200; ++fan)
		{
			const Vec3f center = Vec3f(80.0f, 115.0f, 15.0f) + offset +
								 Vec3f(rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f) * 10.0f;
			const Vec3f normal = Normalize(
				eye - center + Vec3f(rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f, rng.NextFloat()) * 20.0f);
			const Vec3f tangent	  = Normalize(Cross(normal, Vec3f(0.3f, 1.0f, 0.2f)));
			const Vec3f bitangent = Cross(normal, tangent);
			Vec3f		rim[kSpokes];
			for (int i = 0; i < kSpokes; ++i)
			{
				const float angle  = 6.2831853f * (i + 0.4f * rng.NextFloat()) / kSpokes;
				const float radius = 0.02f + 0.02f * rng.NextFloat();
				rim[i]			   = center + (tangent * std::cos(angle) + bitangent * std::sin(angle)) * radius +
									 normal * (0.004f * (rng.NextFloat() - 0.5f));
			}
			TriangleWoop woop[kSpokes];
			for (int i = 0; i < kSpokes; ++i)
			{
				Triangle triangle;
				triangle.vtx[0] = center;
				triangle.vtx[1] = rim[i];
				triangle.vtx[2] = rim[(i + 1) % kSpokes];
				woop[i]			= TriangleWoop::FromTriangle(triangle);
			}
			for (int r = 0; r < 64; ++r)
			{
				const Vec3f spoke  = rim[r % kSpokes] - center;
				const Vec3f beside = Normalize(Cross(normal, spoke)) * (rng.NextFloat() < 0.5f ? -1e-5f : 1e-5f);
				const Ray	ray(eye, Lerp(center, rim[r % kSpokes], 0.05f + 0.9f * rng.NextFloat()) + beside - eye);
				bool		hit = false;
				for (int i = 0; i < kSpokes && !hit; ++i)
				{
					float t, u, v;
					hit = IntersectRayTriangleWoop(ray, woop[i], &t, &u, &v);
				}
				misses += !hit;
				++rays;
			}
		}
		HAPPYRAY_CHECK(rays == 200 * 64);
		HAPPYRAY_CHECK(misses == 0);
	}

	void TestCenterScene()
	{
		using namespace happyray;
		Scene scene;
		Mesh  mesh;
		mesh.positions = {Vec3f(100, 200, 300), Vec3f(102, 200, 300), Vec3f(100, 204, 300)};
		mesh.attributes.resize(3);
		mesh.indices = {0, 1, 2};
		scene.meshes.push_back(mesh);
		const Vec3f offset = CenterScene(&scene);
		HAPPYRAY_CHECK(NearlyEqual(offset, Vec3f(-101, -202, -300)));
		HAPPYRAY_CHECK(NearlyEqual(scene.WorldBounds().Center(), Vec3f(0, 0, 0)));
		HAPPYRAY_CHECK(NearlyEqual(scene.meshes[0].positions[1], Vec3f(1, -2, 0)));

		FlatScene first = FlattenScene(scene), second = first;
		second.positions[0] += Vec3f(2, 0, 0);
		KeyframeAnimation animation({first, second}, 0.5f);
		animation.Translate(Vec3f(1, 1, 1));
		HAPPYRAY_CHECK(NearlyEqual(animation.InterpolatedScene().positions[0], Vec3f(0, -1, 1)));
		animation.Advance();
		HAPPYRAY_CHECK(NearlyEqual(animation.InterpolatedScene().positions[0], Vec3f(1, -1, 1)));

		Scene empty;
		HAPPYRAY_CHECK(NearlyEqual(CenterScene(&empty), Vec3f(0, 0, 0)));
	}

	// Far from the ray origin, hits are still precise enough that AO rays leaving a hit (offset
	// 1e-3 along the normal, as AOIntegrator does) don't hit the same curve right away.
	void TestBezierCurveHitsFarFromOrigin()
	{
		using namespace happyray;
		Rng rng(11, 5);
		int self_hits = 0, hits = 0;
		for (int c = 0; c < 200; ++c)
		{
			const Vec3f center(20.0f * rng.NextFloat() - 10.0f, 20.0f * rng.NextFloat() - 10.0f, 40.0f);
			const Vec3f axis = Normalize(Vec3f(rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f));
			const Vec3f bend = Vec3f(rng.NextFloat(), rng.NextFloat(), rng.NextFloat()) * 0.3f;
			const BezierCurve curve(center - axis,
									center - axis * 0.3f + bend,
									center + axis * 0.3f - bend,
									center + axis,
									0.24f);
			const Vec3f		  eye(0.0f, 0.0f, -35.0f);
			for (int r = 0; r < 32; ++r)
			{
				const Vec3f target = EvaluateBezierCurve(curve, rng.NextFloat()) +
									 Vec3f(rng.NextFloat() - 0.5f, rng.NextFloat() - 0.5f, 0.0f) * 0.4f;
				const Ray	primary(eye, target - eye);
				float		t = 0.0f, u = 0.0f;
				if (!IntersectRayBezierCurve(primary, curve, &t, &u))
					continue;
				++hits;
				const Vec3f position = primary.At(t);
				const Vec3f normal	 = Normalize(position - EvaluateBezierCurve(curve, u));
				for (int i = 0; i < 8; ++i)
				{
					const Vec2f sample = Halton2D(static_cast<uint32_t>(i));
					const Ray occlusion(position + normal * 1e-3f, SampleCosineHemisphere(normal, sample.x, sample.y));
					float	  t_occlusion;
					// Strongly bent curves can genuinely occlude themselves further away.
					self_hits += IntersectRayBezierCurve(occlusion, curve, &t_occlusion) && t_occlusion < curve.radius;
				}
			}
		}
		HAPPYRAY_CHECK(hits > 1000);
		HAPPYRAY_CHECK(self_hits == 0);
	}

	void TestQuaternion()
	{
		using namespace happyray;
		constexpr float kPi = 3.14159265358979323846f;

		HAPPYRAY_CHECK(
			NearlyEqual(RotateVector(Quaternion::Identity(), Vec3f(1.0f, 2.0f, 3.0f)), Vec3f(1.0f, 2.0f, 3.0f)));

		// A 90-degree rotation around +Z takes +X to +Y.
		const Quaternion rot_z90 = Quaternion::FromAxisAngle(Vec3f(0.0f, 0.0f, 1.0f), kPi * 0.5f);
		HAPPYRAY_CHECK(NearlyEqual(RotateVector(rot_z90, Vec3f(1.0f, 0.0f, 0.0f)), Vec3f(0.0f, 1.0f, 0.0f), 1e-3f));

		// A 90-degree rotation around +Y takes +Z to +X.
		const Quaternion rot_y90 = Quaternion::FromAxisAngle(Vec3f(0.0f, 1.0f, 0.0f), kPi * 0.5f);
		HAPPYRAY_CHECK(NearlyEqual(RotateVector(rot_y90, Vec3f(0.0f, 0.0f, 1.0f)), Vec3f(1.0f, 0.0f, 0.0f), 1e-3f));

		// Composition: a * b applies b first, then a.
		const Quaternion composed = rot_y90 * rot_z90;
		HAPPYRAY_CHECK(NearlyEqual(RotateVector(composed, Vec3f(1.0f, 0.0f, 0.0f)),
								   RotateVector(rot_y90, RotateVector(rot_z90, Vec3f(1.0f, 0.0f, 0.0f))),
								   1e-3f));

		// Slerp endpoints reproduce the inputs, and the midpoint is still a unit quaternion.
		HAPPYRAY_CHECK(NearlyEqual(RotateVector(Slerp(rot_z90, rot_y90, 0.0f), Vec3f(1.0f, 0.0f, 0.0f)),
								   RotateVector(rot_z90, Vec3f(1.0f, 0.0f, 0.0f)),
								   1e-3f));
		HAPPYRAY_CHECK(NearlyEqual(RotateVector(Slerp(rot_z90, rot_y90, 1.0f), Vec3f(1.0f, 0.0f, 0.0f)),
								   RotateVector(rot_y90, Vec3f(1.0f, 0.0f, 0.0f)),
								   1e-3f));
		const Quaternion mid = Slerp(rot_z90, rot_y90, 0.5f);
		HAPPYRAY_CHECK(NearlyEqual(Length(mid), 1.0f, 1e-3f));

		// ToMat4 should agree with RotateVector for the same rotation.
		const Mat4 rot_z90_mat = ToMat4(rot_z90);
		HAPPYRAY_CHECK(NearlyEqual(TransformVector(rot_z90_mat, Vec3f(1.0f, 0.0f, 0.0f)),
								   RotateVector(rot_z90, Vec3f(1.0f, 0.0f, 0.0f)),
								   1e-3f));
	}

	void TestHalton()
	{
		using namespace happyray;

		// Base-2 radical inverse: bit-reverses the index around the binary point.
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(1, 2), 0.5f));
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(2, 2), 0.25f));
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(3, 2), 0.75f));
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(4, 2), 0.125f));

		// Base-3 radical inverse.
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(1, 3), 1.0f / 3.0f));
		HAPPYRAY_CHECK(NearlyEqual(HaltonRadicalInverse(3, 3), 1.0f / 9.0f));

		// Even and odd indices (alternate frames, see framebuffer.h) each fill every quarter of
		// both dimensions about equally.
		for (uint32_t parity = 0; parity < 2; ++parity)
		{
			int quarters[2][4] = {};
			for (uint32_t i = parity; i < 1024; i += 2)
			{
				const Vec2f sample = Halton2D(i);
				++quarters[0][static_cast<int>(sample.x * 4.0f)];
				++quarters[1][static_cast<int>(sample.y * 4.0f)];
			}
			for (const auto &dimension : quarters)
				for (int count : dimension)
					HAPPYRAY_CHECK(count >= 112 && count <= 144); // 128 each.
		}

		// Every sample stays inside [0, 1), for a reasonable range of indices.
		for (uint32_t i = 0; i < 1000; ++i)
		{
			const Vec2f sample = Halton2D(i);
			HAPPYRAY_CHECK(sample.x >= 0.0f && sample.x < 1.0f);
			HAPPYRAY_CHECK(sample.y >= 0.0f && sample.y < 1.0f);
		}

		// Low discrepancy: the first N samples' mean should be close to (0.5, 0.5), much closer
		// than a handful of independent random draws would typically land.
		Vec2f		  sum(0.0f, 0.0f);
		constexpr int kN = 256;
		for (int i = 1; i <= kN; ++i)
		{
			const Vec2f sample = Halton2D(static_cast<uint32_t>(i));
			sum += sample;
		}
		const Vec2f mean = sum / static_cast<float>(kN);
		HAPPYRAY_CHECK(NearlyEqual(mean.x, 0.5f, 0.05f));
		HAPPYRAY_CHECK(NearlyEqual(mean.y, 0.5f, 0.05f));

		// Cranley-Patterson rotation wraps around [0, 1) and is a pure offset (no-op at offset 0).
		HAPPYRAY_CHECK(NearlyEqual(CranleyPattersonRotate(0.7f, 0.0f), 0.7f));
		HAPPYRAY_CHECK(NearlyEqual(CranleyPattersonRotate(0.7f, 0.5f), 0.2f));
		const float rotated = CranleyPattersonRotate(0.9f, 0.9f);
		HAPPYRAY_CHECK(rotated >= 0.0f && rotated < 1.0f);
		HAPPYRAY_CHECK(NearlyEqual(rotated, 0.8f, 1e-4f));
	}

	void TestObjLoaderTriangleNoNormals(const std::string &assets_dir)
	{
		using namespace happyray;
		Scene scene;
		HAPPYRAY_CHECK(LoadObj(assets_dir + "/triangle.obj", &scene));
		HAPPYRAY_CHECK(scene.meshes.size() == 1);
		if (scene.meshes.empty())
		{
			return;
		}
		const Mesh &mesh = scene.meshes[0];
		HAPPYRAY_CHECK(mesh.positions.size() > 3);
		HAPPYRAY_CHECK(mesh.TriangleCount() > 1);
		CheckSubdivisionLimit(scene);
		// No vn lines, so LoadObj derives +Z normals for this counter-clockwise triangle.
		for (const VertexAttributes &attributes : mesh.attributes)
		{
			HAPPYRAY_CHECK(NearlyEqual(attributes.normal, Vec3f(0.0f, 0.0f, 1.0f)));
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.x, 0.0f));
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.y, 0.0f));
		}

		// Without materials, FlattenScene still supplies a white default for material id 0.
		HAPPYRAY_CHECK(scene.materials.empty());
		const FlatScene flat = FlattenScene(scene);
		HAPPYRAY_CHECK(!flat.material_diffuse_colors.empty());
		HAPPYRAY_CHECK(NearlyEqual(flat.material_diffuse_colors[0], Vec3f(1.0f, 1.0f, 1.0f)));
	}

	void TestObjLoaderAttributeSeam(const std::string &assets_dir)
	{
		using namespace happyray;
		Scene scene;
		HAPPYRAY_CHECK(LoadObj(assets_dir + "/attribute_seam.obj", &scene));
		HAPPYRAY_CHECK(scene.meshes.size() == 1);
		if (scene.meshes.empty())
			return;
		const Mesh &mesh = scene.meshes.front();
		CheckSubdivisionLimit(scene);
		HAPPYRAY_CHECK(mesh.positions.size() >= 6);
		if (mesh.positions.size() < 6 || mesh.attributes.size() < 6)
			return;
		const Vec2f expected_uvs[] = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(0, 1), Vec2f(1, 0)};
		for (size_t i = 0; i < 6; ++i)
		{
			HAPPYRAY_CHECK(NearlyEqual(mesh.positions[i], mesh.positions[i % 3]));
			HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].normal, Vec3f(0, 0, i < 3 ? 1 : -1)));
			HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].uv.x, expected_uvs[i].x));
			HAPPYRAY_CHECK(NearlyEqual(mesh.attributes[i].uv.y, expected_uvs[i].y));
		}
	}

	void TestDefaultMaxDistance()
	{
		happyray::Scene scene;
		happyray::Mesh	mesh;
		mesh.positions = {happyray::Vec3f(-2, 1, 0), happyray::Vec3f(8, 5, 3)};
		scene.meshes.push_back(mesh);
		HAPPYRAY_CHECK(NearlyEqual(happyray::DefaultMaxDistance(scene), 1.0f));

		HAPPYRAY_CHECK(NearlyEqual(happyray::DefaultMaxDistance(happyray::Scene()), 1.0f));
	}

	void TestFlattenScene()
	{
		using namespace happyray;
		Scene scene;
		HAPPYRAY_CHECK(FlattenScene(scene).positions.empty());
		for (int i = 0; i < 2; ++i)
		{
			Mesh mesh;
			mesh.positions	= {Vec3f(i, 0, 0), Vec3f(i, 1, 0), Vec3f(i, 0, 1)};
			mesh.attributes = {
				{Vec3f(1, i, 0), Vec2f(0, 0)},
				{Vec3f(0, 1, i), Vec2f(1, 0)},
				{Vec3f(i, 0, 1), Vec2f(0, 1)}
			 };
			mesh.indices	  = {2, 0, 1};
			mesh.material_ids = {0};
			scene.meshes.push_back(mesh);
		}
		const FlatScene flat = FlattenScene(scene);
		HAPPYRAY_CHECK(flat.positions.size() == 6);
		HAPPYRAY_CHECK(flat.normals.size() == 6);
		HAPPYRAY_CHECK(flat.indices == std::vector<uint32_t>({2, 0, 1, 5, 3, 4}));
		HAPPYRAY_CHECK(flat.material_ids == std::vector<uint32_t>({0, 0}));
		for (size_t m = 0; m < scene.meshes.size(); ++m)
		{
			for (size_t i = 0; i < 3; ++i)
			{
				HAPPYRAY_CHECK(NearlyEqual(flat.positions[m * 3 + i], scene.meshes[m].positions[i]));
				HAPPYRAY_CHECK(NearlyEqual(flat.normals[m * 3 + i], scene.meshes[m].attributes[i].normal));
			}
		}
		HAPPYRAY_CHECK(flat.View().positions == flat.positions.data());
		HAPPYRAY_CHECK(flat.View().normals == flat.normals.data());
		HAPPYRAY_CHECK(flat.View().primitive_count == 2);
	}

	// Swept boxes bound each triangle at both ends of its motion and keep its ids.
	void TestSweptTriangleBoxes()
	{
		using namespace happyray;
		FlatScene a;
		a.positions				  = {Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(0, 1, 0)};
		a.indices				  = {0, 1, 2};
		a.material_ids			  = {3};
		a.material_diffuse_colors = {Vec3f(1, 0, 0)};
		a.object_ids			  = {7};
		FlatScene b				  = a;
		b.positions				  = {Vec3f(2, -1, 0), Vec3f(3, 0, 1), Vec3f(2, 1, 0)};
		const FlatScene boxes	  = MakeSweptTriangleBoxes(a, b, 0.5f, 0.6f);
		HAPPYRAY_CHECK(boxes.primitive_type == PrimitiveType::kBox);
		HAPPYRAY_CHECK(boxes.PrimitiveCount() == 1);
		HAPPYRAY_CHECK(boxes.View().boxes == boxes.boxes.data());
		HAPPYRAY_CHECK(boxes.View().primitive_count == 1);
		HAPPYRAY_CHECK(boxes.material_ids == a.material_ids && boxes.object_ids == a.object_ids);
		HAPPYRAY_CHECK(NearlyEqual(boxes.boxes[0].min_bound, Vec3f(1, -0.6f, 0)));
		HAPPYRAY_CHECK(NearlyEqual(boxes.boxes[0].max_bound, Vec3f(2.2f, 1, 0.6f)));

		FlatScene other_topology = b;
		other_topology.indices	 = {0, 2, 1};
		bool threw				 = false;
		try
		{
			MakeSweptTriangleBoxes(a, other_topology, 0.0f, 1.0f);
		}
		catch (const std::invalid_argument &)
		{
			threw = true;
		}
		HAPPYRAY_CHECK(threw);
	}

	// `o`/`g` statements become per-triangle object ids that survive subdivision and flattening.
	void TestObjLoaderObjectIds()
	{
		using namespace happyray;
		const std::string path = (std::filesystem::temp_directory_path() / "happyray_core_tests_objects.obj").string();
		{
			std::ofstream obj(path);
			obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
				   "f 1 2 3\n"	// Default object: 0.
				   "o A\no B\n" // A has no faces yet, so it gets no id.
				   "f 1 2 4\n"	// (B): 1.
				   "g left arm \n"
				   "f 1 3 4\n" // (B, "left arm"): 2.
				   "g right\n"
				   "f 2 3 4\n" // (B, right): 3.
				   "g left arm\n"
				   "f 1 2 3 4\n" // Repeated group continues object 2 (quad -> 2 triangles).
				   "o A\n"
				   "f 1 2 4\n"; // (A): 4.
		}
		Scene scene;
		HAPPYRAY_CHECK(LoadObj(path, &scene));
		std::filesystem::remove(path);
		HAPPYRAY_CHECK(scene.meshes.size() == 1);
		if (scene.meshes.size() != 1)
			return;
		const Mesh &mesh = scene.meshes[0];
		// Subdivision rewrites triangles in place and appends children, so the first entries keep
		// the parsed order.
		HAPPYRAY_CHECK(mesh.TriangleCount() > 7);
		HAPPYRAY_CHECK(mesh.object_ids.size() == mesh.TriangleCount());
		const std::vector<uint32_t> parsed(mesh.object_ids.begin(), mesh.object_ids.begin() + 7);
		HAPPYRAY_CHECK(parsed == std::vector<uint32_t>({0, 1, 2, 3, 2, 2, 4}));
		for (uint32_t object_id : mesh.object_ids)
			HAPPYRAY_CHECK(object_id < 5);

		// A second mesh without object ids is one more object after the first mesh's five.
		Mesh second = mesh;
		second.object_ids.clear();
		scene.meshes.push_back(second);
		const FlatScene flat = FlattenScene(scene);
		HAPPYRAY_CHECK(flat.object_ids.size() == flat.indices.size() / 3);
		HAPPYRAY_CHECK(flat.View().object_ids == flat.object_ids.data());
		for (uint32_t tri = 0; tri < mesh.TriangleCount(); ++tri)
		{
			HAPPYRAY_CHECK(flat.object_ids[tri] == mesh.object_ids[tri]);
			HAPPYRAY_CHECK(flat.object_ids[mesh.TriangleCount() + tri] == 5);
		}
	}

	void TestObjLoaderCubeWithMaterialsAndQuads(const std::string &assets_dir)
	{
		using namespace happyray;
		Scene scene;
		HAPPYRAY_CHECK(LoadObj(assets_dir + "/cube.obj", &scene));
		HAPPYRAY_CHECK(scene.meshes.size() == 1);
		HAPPYRAY_CHECK(scene.materials.size() == 2);
		if (scene.meshes.empty())
		{
			return;
		}
		const Mesh &mesh = scene.meshes[0];
		// The 12 original triangles and their children retain their face materials.
		HAPPYRAY_CHECK(mesh.TriangleCount() > 12);
		HAPPYRAY_CHECK(mesh.material_ids.size() == mesh.TriangleCount());
		CheckSubdivisionLimit(scene);
		const uint32_t red_id = (scene.materials[0].name == "red") ? 0 : 1;
		HAPPYRAY_CHECK(mesh.material_ids[0] == red_id);
		HAPPYRAY_CHECK(mesh.material_ids[3] == red_id);
		HAPPYRAY_CHECK(mesh.material_ids[4] != red_id);
		double red_area = 0, blue_area = 0;
		for (size_t tri = 0; tri < mesh.indices.size(); tri += 3)
		{
			const Vec3f	 a	  = mesh.positions[mesh.indices[tri]];
			const Vec3f	 b	  = mesh.positions[mesh.indices[tri + 1]];
			const Vec3f	 c	  = mesh.positions[mesh.indices[tri + 2]];
			const double area = 0.5 * Length(Cross(b - a, c - a));
			if (mesh.material_ids[tri / 3] == red_id)
				red_area += area;
			else
				blue_area += area;
		}
		HAPPYRAY_CHECK(std::fabs(red_area - 2) < 1e-6);
		HAPPYRAY_CHECK(std::fabs(blue_area - 4) < 1e-6);

		// cube.mtl's Kd lines should have been parsed into the matching materials via mtllib.
		const uint32_t blue_id = red_id == 0 ? 1 : 0;
		HAPPYRAY_CHECK(NearlyEqual(scene.materials[red_id].diffuse_color, Vec3f(0.8f, 0.1f, 0.1f)));
		HAPPYRAY_CHECK(NearlyEqual(scene.materials[blue_id].diffuse_color, Vec3f(0.1f, 0.1f, 0.8f)));

		const BBox bounds = scene.WorldBounds();
		HAPPYRAY_CHECK(NearlyEqual(bounds.Extent(), Vec3f(1.0f, 1.0f, 1.0f)));
	}

	void TestKeyframeAnimation()
	{
		using namespace happyray;
		FlatScene first;
		first.positions = {Vec3f(0, 0, 0), Vec3f(2, 0, 0)};
		first.normals	= {Vec3f(1, 0, 0), Vec3f(1, 0, 0)};
		first.indices	= {0, 0, 0};
		FlatScene second;
		second.positions = {Vec3f(10, 0, 0), Vec3f(12, 0, 0)};
		second.normals	 = {Vec3f(0, 1, 0), Vec3f(0, 1, 0)};
		second.indices	 = {1, 0, 1};

		KeyframeAnimation animation({first, second}, 0.5f);
		FlatScene		  frame = animation.InterpolatedScene();
		HAPPYRAY_CHECK(frame.indices == second.indices);
		HAPPYRAY_CHECK(NearlyEqual(frame.positions[0], first.positions[0]));

		animation.Advance();
		frame = animation.InterpolatedScene();
		HAPPYRAY_CHECK(NearlyEqual(frame.positions[0], Vec3f(5, 0, 0)));
		HAPPYRAY_CHECK(NearlyEqual(frame.normals[0], Normalize(Vec3f(1, 1, 0))));
		animation.TogglePause();
		animation.Advance();
		HAPPYRAY_CHECK(NearlyEqual(animation.InterpolatedScene().positions[0], Vec3f(5, 0, 0)));

		animation.TogglePause();
		animation.Advance();
		HAPPYRAY_CHECK(NearlyEqual(animation.InterpolatedScene().positions[0], Vec3f(10, 0, 0)));
		animation.Advance();
		animation.Advance();
		HAPPYRAY_CHECK(NearlyEqual(animation.InterpolatedScene().positions[0], Vec3f(0, 0, 0)));
	}

} // namespace

int main(int argc, char **argv)
{
	const std::string assets_dir = (argc > 1) ? argv[1] : "assets";

	TestVecMath();
	TestMat4();
	TestBBox();
	TestBBoxExtractor();
	TestAdditionalPrimitiveBounds();
	TestPrimitiveIntersections();
	TestClosestPointPrimitives();
	TestClosestPointMatchesSampling();
	TestBezierCurveMatchesStackSubdivision();
	TestTriangleWoopCenteredGaps();
	TestCenterScene();
	TestBezierCurveHitsFarFromOrigin();
	TestQuaternion();
	TestHalton();
	TestTriangleSubdivision();
	TestDefaultMaxDistance();
	TestFlattenScene();
	TestSweptTriangleBoxes();
	TestObjLoaderTriangleNoNormals(assets_dir);
	TestObjLoaderAttributeSeam(assets_dir);
	TestObjLoaderCubeWithMaterialsAndQuads(assets_dir);
	TestObjLoaderObjectIds();
	TestKeyframeAnimation();

	if (g_failures == 0)
	{
		std::printf("All core tests passed.\n");
		return EXIT_SUCCESS;
	}
	std::fprintf(stderr, "%d core test check(s) failed.\n", g_failures);
	return EXIT_FAILURE;
}
