// Lightweight, dependency-free unit tests for src/core (no third-party test framework, matching
// the project's "no non-essential dependencies" preference). Each check prints a message and
// the test binary returns non-zero on first failure; registered with CTest.
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
		// The camera looks down -Z in its own space, so the world origin (5 units in front of the
		// eye along +Z) should map to camera-space (0, 0, -5).
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
		HAPPYRAY_CHECK(IntersectTriangles(flat, crossing));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, separated));
		HAPPYRAY_CHECK(IntersectTriangles(flat, coplanar_overlap));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, coplanar_separated));
		HAPPYRAY_CHECK(IntersectTriangles(flat, degenerate_inside));
		HAPPYRAY_CHECK(!IntersectTriangles(flat, degenerate_outside));
	}

	void TestQuaternion()
	{
		using namespace happyray;
		constexpr float kPi = 3.14159265358979323846f;

		// Identity leaves every vector untouched.
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
		// No vn lines were present, so LoadObj must have derived a flat normal of +Z for every
		// vertex of this single, counter-clockwise-when-viewed-from+Z triangle.
		for (const VertexAttributes &attributes : mesh.attributes)
		{
			HAPPYRAY_CHECK(NearlyEqual(attributes.normal, Vec3f(0.0f, 0.0f, 1.0f)));
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.x, 0.0f));
			HAPPYRAY_CHECK(NearlyEqual(attributes.uv.y, 0.0f));
		}

		// No usemtl/mtllib at all: Scene::materials stays empty, but FlattenScene must still supply
		// a valid (white, default) diffuse color at index 0 for the unassigned material id every
		// triangle implicitly uses.
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
	TestQuaternion();
	TestHalton();
	TestTriangleSubdivision();
	TestDefaultMaxDistance();
	TestFlattenScene();
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
