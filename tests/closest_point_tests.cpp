// Closest-point queries through every grid and cell overlap mode must return brute force's answer.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../src/accel/closest_point.h"
#include "../src/accel/irregular_grid.h"
#include "../src/accel/two_level_grid.h"
#include "../src/accel/uniform_grid.h"
#include "../src/app/attractor_scene.h"
#include "../src/app/obj_loader.h"
#include "../src/core/random.h"

namespace
{
	using namespace happyray;

	int g_failures = 0;

	void Check(bool condition, const std::string &message)
	{
		if (!condition)
		{
			std::fprintf(stderr, "FAILED: %s\n", message.c_str());
			++g_failures;
		}
	}

	struct DeviceScene
	{
		thrust::device_vector<Vec3f>	   positions, normals;
		thrust::device_vector<uint32_t>	   indices, material_ids;
		thrust::device_vector<Point>	   points;
		thrust::device_vector<Sphere>	   spheres;
		thrust::device_vector<BezierCurve> curves;
		thrust::device_vector<BBox>		   boxes;
		SceneView						   view;

		explicit DeviceScene(const FlatScene &scene)
			: positions(scene.positions.begin(), scene.positions.end()),
			  normals(scene.normals.begin(), scene.normals.end()), indices(scene.indices.begin(), scene.indices.end()),
			  material_ids(scene.material_ids.begin(), scene.material_ids.end()),
			  points(scene.points.begin(), scene.points.end()), spheres(scene.spheres.begin(), scene.spheres.end()),
			  curves(scene.curves.begin(), scene.curves.end()), boxes(scene.boxes.begin(), scene.boxes.end()),
			  view(scene.View())
		{
			view.positions				 = thrust::raw_pointer_cast(positions.data());
			view.normals				 = thrust::raw_pointer_cast(normals.data());
			view.indices				 = thrust::raw_pointer_cast(indices.data());
			view.material_ids			 = thrust::raw_pointer_cast(material_ids.data());
			view.object_ids				 = nullptr;
			view.material_diffuse_colors = nullptr;
			view.points					 = thrust::raw_pointer_cast(points.data());
			view.spheres				 = thrust::raw_pointer_cast(spheres.data());
			view.curves					 = thrust::raw_pointer_cast(curves.data());
			view.boxes					 = thrust::raw_pointer_cast(boxes.data());
		}
	};

	std::vector<BBox> PrimitiveBounds(const FlatScene &flat)
	{
		const SceneView	  view = flat.View();
		std::vector<BBox> bounds(view.primitive_count);
		DispatchPrimitive(view.primitive_type,
						  [&](auto tag)
						  {
							  using P = typename decltype(tag)::type;
							  for (uint32_t i = 0; i < view.primitive_count; ++i)
								  bounds[i] = ComputePrimitiveBBox<P>(view, i);
						  });
		return bounds;
	}

	// A third each near primitives (often inside solids), anywhere in the scene, and far outside it.
	std::vector<Vec3f> MakeQueries(const FlatScene &flat, uint32_t count)
	{
		const std::vector<BBox> prims = PrimitiveBounds(flat);
		BBox					scene;
		for (const BBox &box : prims)
			scene.Grow(box);
		Rng				   rng(3, 9);
		auto			   unit = [&]() { return Vec3f(rng.NextFloat(), rng.NextFloat(), rng.NextFloat()); };
		std::vector<Vec3f> queries;
		for (uint32_t i = 0; i < count; ++i)
		{
			if (i % 3 == 0)
			{
				const BBox &box = prims[rng.NextUint32() % prims.size()];
				queries.push_back(box.Center() + (unit() - Vec3f(0.5f)) * (Length(box.Extent()) * 2.0f));
			}
			else if (i % 3 == 1)
				queries.push_back(scene.min_bound + unit() * scene.Extent());
			else
				queries.push_back(scene.Center() + (unit() - Vec3f(0.5f)) * (scene.Extent() * 6.0f));
		}
		return queries;
	}

	std::vector<ClosestPointRecord> Run(const SceneView					   &scene,
										const AccelStructures			   &accel,
										const thrust::device_vector<Vec3f> &queries,
										float								max_distance)
	{
		thrust::device_vector<ClosestPointRecord> out(queries.size());
		FindClosestPoints(scene,
						  accel,
						  thrust::raw_pointer_cast(queries.data()),
						  static_cast<uint32_t>(queries.size()),
						  max_distance,
						  thrust::raw_pointer_cast(out.data()));
		thrust::host_vector<ClosestPointRecord> host = out;
		return std::vector<ClosestPointRecord>(host.begin(), host.end());
	}

	// Kernels may contract floating-point operations differently, so equal answers agree to rounding
	// and near-ties may resolve to different primitives.
	bool SameRecord(const ClosestPointRecord &a, const ClosestPointRecord &b)
	{
		if (a.prim == 0xffffffffu || b.prim == 0xffffffffu)
			return a.prim == b.prim;
		const float tolerance = 1e-5f * std::fmax(1.0f, b.distance);
		return std::fabs(a.distance - b.distance) <= tolerance &&
			   (a.prim != b.prim || Length(a.position - b.position) <= 10.0f * tolerance);
	}

	void CheckScene(const char *name, const FlatScene &flat, uint32_t query_count)
	{
		const DeviceScene					  scene(flat);
		const std::vector<Vec3f>			  host_queries = MakeQueries(flat, query_count);
		const thrust::device_vector<Vec3f>	  queries(host_queries.begin(), host_queries.end());
		AccelStructures						  brute;
		const std::vector<ClosestPointRecord> reference = Run(scene.view, brute, queries, INFINITY);

		// A bound that excludes some, but not all, primitives' closest points.
		std::vector<float> distances;
		for (const ClosestPointRecord &record : reference)
		{
			Check(record.prim != 0xffffffffu, std::string(name) + ": unbounded brute force finds a primitive");
			distances.push_back(record.distance);
		}
		std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
		const float							  max_distance		= distances[distances.size() / 2];
		const std::vector<ClosestPointRecord> bounded_reference = Run(scene.view, brute, queries, max_distance);

		const CellOverlapMode modes[] = {CellOverlapMode::kAabb, CellOverlapMode::kExact, CellOverlapMode::kHybrid};
		for (CellOverlapMode mode : modes)
		{
			const UniformGrid	uniform(scene.view, UniformGrid::kDefaultDensity, mode);
			const TwoLevelGrid	two_level(scene.view,
										  TwoLevelGrid::kDefaultTopLevelDensity,
										  TwoLevelGrid::kDefaultSecondLevelDensity,
										  16,
										  mode);
			const IrregularGrid irregular(scene.view,
										  IrregularGrid::kDefaultTopLevelDensity,
										  IrregularGrid::kDefaultSecondLevelDensity,
										  16,
										  0.7f,
										  2,
										  mode);
			AccelStructures		accel;
			accel.uniform_grid		  = uniform.View();
			accel.two_level_grid	  = two_level.ClosestPointView();
			accel.irregular_grid	  = irregular.View();
			const AccelType	  types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
			const char *const type_names[] = {"uniform", "two-level", "irregular"};
			for (int t = 0; t < 3; ++t)
			{
				accel.type										 = types[t];
				const std::vector<ClosestPointRecord> got		 = Run(scene.view, accel, queries, INFINITY);
				const std::vector<ClosestPointRecord> bounded	 = Run(scene.view, accel, queries, max_distance);
				uint32_t							  mismatches = 0, bounded_mismatches = 0;
				for (size_t i = 0; i < got.size(); ++i)
				{
					mismatches += !SameRecord(got[i], reference[i]);
					bounded_mismatches += !SameRecord(bounded[i], bounded_reference[i]);
				}
				Check(mismatches == 0 && bounded_mismatches == 0,
					  std::string(name) + " " + type_names[t] + " " + CellOverlapModeName(mode) + ": " +
						  std::to_string(mismatches) + " unbounded, " + std::to_string(bounded_mismatches) +
						  " bounded mismatches of " + std::to_string(got.size()));
			}
		}
		std::printf("%s: checked %u queries\n", name, query_count);
	}

	FlatScene MakeBoxScene(uint32_t count)
	{
		FlatScene flat		= FlattenScene(MakeAttractorScene(PrimitiveType::kSphere, count));
		flat.primitive_type = PrimitiveType::kBox;
		Rng rng(5, 1);
		for (const Sphere &sphere : flat.spheres)
		{
			const Vec3f half(sphere.radius * (0.2f + 3.0f * rng.NextFloat()),
							 sphere.radius * (0.2f + 3.0f * rng.NextFloat()),
							 sphere.radius * (0.2f + 3.0f * rng.NextFloat()));
			flat.boxes.push_back(BBox(sphere.vtx[0] - half, sphere.vtx[0] + half));
		}
		flat.spheres.clear();
		return flat;
	}

} // namespace

int main(int argc, char **argv)
{
	const std::string assets_dir = argc > 1 ? argv[1] : "assets";
	Scene			  mesh;
	if (LoadObj(assets_dir + "/ogre.obj", &mesh))
		CheckScene("triangles", FlattenScene(mesh), 1500);
	else
		Check(false, "loads ogre.obj");
	CheckScene("points", FlattenScene(MakeAttractorScene(PrimitiveType::kPoint, 4000)), 3000);
	CheckScene("spheres", FlattenScene(MakeAttractorScene(PrimitiveType::kSphere, 4000)), 3000);
	CheckScene("curves", FlattenScene(MakeAttractorScene(PrimitiveType::kBezierCurve, 4000)), 1500);
	CheckScene("boxes", MakeBoxScene(4000), 3000);

	std::printf("Closest-point tests: %d failure(s)\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
