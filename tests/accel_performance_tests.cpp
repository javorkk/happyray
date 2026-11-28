// Builds and traces every grid once after a warm-up, then runs closest-point query sets through
// each; timings are informational, only correctness and allocation errors fail. Pure Thrust, so it
// builds for both backends.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "../src/accel/closest_point.h"
#include "../src/accel/intersect_common.h"
#include "../src/accel/knn.h"
#include "../src/accel/sah_cost.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/core/random.h"

namespace
{

	using namespace happyray;

	constexpr uint32_t kRayCount		 = 1024 * 1024;
	constexpr uint32_t kQueryCount		 = 256 * 1024;
	constexpr int	   kClosestPointRuns = 5;
	// Sparser than the ray-tuned defaults: closest-point queries spend their time walking cells, so
	// fewer, larger cells are faster with both backends. The uniform grid keeps its default.
	constexpr float kClosestPointTopLevelDensity = 0.06f;
	constexpr float kClosestPointLeafDensity	 = 0.6f;
	// Prints per-query work counters before the throughput summary when working on closest point.
	constexpr bool kPrintClosestPointDetails = false;
	// Fewer k-nearest queries keep the k records per query within memory.
	constexpr uint32_t kKNearestQueryCount = 64 * 1024;
	constexpr uint32_t kKNearestKs[]	   = {8, 32};

	template <typename T> T *Raw(thrust::device_vector<T> &values)
	{
		return thrust::raw_pointer_cast(values.data());
	}

	struct DeviceScene
	{
		thrust::device_vector<Vec3f>	positions, normals, colors;
		thrust::device_vector<uint32_t> indices, materials;

		explicit DeviceScene(const FlatScene &scene)
			: positions(scene.positions), normals(scene.normals), colors(scene.material_diffuse_colors),
			  indices(scene.indices), materials(scene.material_ids)
		{
		}

		SceneView View()
		{
			return {Raw(positions),
					Raw(normals),
					Raw(indices),
					Raw(materials),
					Raw(colors),
					static_cast<uint32_t>(materials.size())};
		}
	};

	template <typename View> struct Trace
	{
		SceneView  scene;
		View	   grid;
		const Ray *rays;

		HAPPYRAY_HD float operator()(uint32_t i) const
		{
			HitRecord hit;
			return IntersectSceneGrid<Triangle>(scene, grid, rays[i], &hit) ? hit.t : -1.0f;
		}
	};

	// Reading one element back forces the CUDA backend to finish pending work.
	template <typename T> void Synchronize(const T *device_data)
	{
		static_cast<void>(static_cast<T>(*thrust::device_pointer_cast(device_data)));
	}

	void SyncGrid(const UniformGridView &view)
	{
		Synchronize(view.cell_ranges);
	}
	void SyncGrid(const TwoLevelGridView &view)
	{
		Synchronize(view.top_level_cells);
	}
	void SyncGrid(const IrregularGridView &view)
	{
		Synchronize(view.top_level.cell_ranges);
	}

	// Analytic footprint (count * sizeof(element)) of each grid's persistent buffers; excludes
	// transient build scratch.
	size_t GridMemoryBytes(const UniformGrid &grid)
	{
		const auto num_cells = static_cast<size_t>(grid.LeafCellCount());
		return num_cells * sizeof(CellRange) + static_cast<size_t>(grid.PrimIndexCount()) * sizeof(uint32_t);
	}

	size_t GridMemoryBytes(const TwoLevelGrid &grid)
	{
		const TwoLevelGridView view = grid.View();
		const auto num_top_cells	= static_cast<size_t>(view.top_level.resolution.x) * view.top_level.resolution.y *
								   view.top_level.resolution.z;
		size_t bytes = num_top_cells * sizeof(TopLevelCell);
		bytes += num_top_cells * sizeof(uint32_t);
		bytes += static_cast<size_t>(grid.LeafCellCount()) * sizeof(CellRange);
		bytes += static_cast<size_t>(grid.SecondLevelPrimIndexCount()) * sizeof(uint32_t);
		// Closest-point empty radius per top-level and nested cell.
		bytes += (num_top_cells + static_cast<size_t>(grid.LeafCellCount())) * sizeof(uint8_t);
		return bytes;
	}

	size_t GridMemoryBytes(const IrregularGrid &grid)
	{
		const IrregularGridView view = grid.View();
		const auto num_top_cells	 = static_cast<size_t>(view.top_level.resolution.x) * view.top_level.resolution.y *
								   view.top_level.resolution.z;
		size_t bytes = num_top_cells * sizeof(CellRange);
		bytes += static_cast<size_t>(grid.TopLevelPrimIndexCount()) * sizeof(uint32_t);
		bytes += num_top_cells * sizeof(IrregularCellInfo);
		bytes += num_top_cells * sizeof(IrregularTopCell);
		bytes += static_cast<size_t>(grid.LocalEntryCount()) * sizeof(uint32_t);
		bytes += static_cast<size_t>(grid.LeafCellCount()) * sizeof(IrregularCell);
		bytes += static_cast<size_t>(grid.PrimIndexCount()) * sizeof(uint32_t);
		return bytes;
	}

	struct Measurement
	{
		std::string		   name;
		double			   build_ms;
		float			   trace_ms;
		size_t			   memory_bytes;
		float			   sah_cost;
		std::vector<float> hits;

		double Mrays() const
		{
			return kRayCount / (1000.0 * trace_ms);
		}
	};

	template <typename MakeGrid>
	Measurement Measure(const std::string		   &name,
						MakeGrid					make_grid,
						const SceneView			   &scene,
						const BBox				   &scene_bounds,
						const SahCostConstants	   &costs,
						thrust::device_vector<Ray> &rays)
	{
		{
			auto warmup_grid = make_grid();
			SyncGrid(warmup_grid->View());
		}

		const auto start = std::chrono::steady_clock::now();
		auto	   grid	 = make_grid();
		SyncGrid(grid->View());
		const auto	 end		  = std::chrono::steady_clock::now();
		const size_t memory_bytes = GridMemoryBytes(*grid);
		const double build_ms	  = std::chrono::duration<double, std::milli>(end - start).count();
		const float	 sah_cost	  = grid->ComputeSahCost(scene_bounds, costs);

		thrust::device_vector<float> hits(kRayCount);
		const auto					 trace_begin = std::chrono::steady_clock::now();
		thrust::transform(thrust::counting_iterator<uint32_t>(0),
						  thrust::counting_iterator<uint32_t>(kRayCount),
						  hits.begin(),
						  Trace<decltype(grid->View())>{scene, grid->View(), Raw(rays)});
		Synchronize(Raw(hits));
		const auto	trace_end = std::chrono::steady_clock::now();
		const float trace_ms  = std::chrono::duration<float, std::milli>(trace_end - trace_begin).count();
		if (!std::isfinite(build_ms) || build_ms <= 0 || !std::isfinite(trace_ms) || trace_ms <= 0)
			throw std::runtime_error("Invalid measured duration");
		if (!std::isfinite(sah_cost) || sah_cost <= 0)
			throw std::runtime_error("Invalid SAH cost estimate");
		if (memory_bytes == 0)
			throw std::runtime_error("Invalid memory accounting");
		const thrust::host_vector<float> host = hits;
		return {
			name,
			build_ms,
			trace_ms,
			memory_bytes,
			sah_cost,
			{host.begin(), host.end()}
		  };
	}

	bool SameHit(float a, float b)
	{
		return (a == -1 && b == -1) || (a >= 0 && b >= 0 && std::isfinite(a) && std::isfinite(b) &&
										std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(b)));
	}

	struct Distribution
	{
		double	 mean = 0;
		uint32_t p50 = 0, p99 = 0, max = 0;
	};

	Distribution Summarize(std::vector<uint32_t> values)
	{
		Distribution result;
		if (values.empty())
			return result;
		std::sort(values.begin(), values.end());
		double sum = 0;
		for (uint32_t value : values)
			sum += value;
		result.mean = sum / values.size();
		result.p50	= values[values.size() / 2];
		result.p99	= values[values.size() * 99 / 100];
		result.max	= values.back();
		return result;
	}

	// Near: on a random triangle, offset along its normal by up to 1% of the scene diagonal.
	// Inside: uniform in the scene bounds. Far: outside the bounds, within three times their extent.
	std::vector<Vec3f> MakeClosestPointQueries(const FlatScene &flat, const BBox &bounds, int set)
	{
		Rng				   rng(20261124, 3 + set);
		const float		   diagonal = Length(bounds.Extent());
		const SceneView	   view		= flat.View();
		std::vector<Vec3f> queries;
		queries.reserve(kQueryCount);
		while (queries.size() < kQueryCount)
		{
			const Vec3f unit(rng.NextFloat(), rng.NextFloat(), rng.NextFloat());
			if (set == 0)
			{
				const Triangle triangle = FetchTriangle(view, rng.NextUint32() % view.primitive_count);
				float		   u = rng.NextFloat(), v = rng.NextFloat();
				if (u + v > 1)
				{
					u = 1 - u;
					v = 1 - v;
				}
				const Vec3f normal =
					Normalize(Cross(triangle.vtx[1] - triangle.vtx[0], triangle.vtx[2] - triangle.vtx[0]));
				queries.push_back(triangle.vtx[0] * (1 - u - v) + triangle.vtx[1] * u + triangle.vtx[2] * v +
								  normal * (0.02f * diagonal * (rng.NextFloat() - 0.5f)));
			}
			else if (set == 1)
				queries.push_back(bounds.min_bound + bounds.Extent() * unit);
			else
			{
				const Vec3f query = bounds.Center() + bounds.Extent() * (unit - Vec3f(0.5f)) * 3.0f;
				if (DistanceSquaredToBBox(query, bounds) > 0)
					queries.push_back(query);
			}
		}
		return queries;
	}

	bool SameClosestPoint(const ClosestPointRecord &a, const ClosestPointRecord &b)
	{
		return a.prim != 0xffffffffu && b.prim != 0xffffffffu &&
			   std::fabs(a.distance - b.distance) <= 1e-5f * std::max(1.0f, b.distance);
	}

	// Throughput of k-nearest queries over the closest-point query sets; every grid must agree with
	// the uniform grid rank by rank, and the uniform grid with brute force on a sample.
	void RunKNearest(const FlatScene &flat, const SceneView &view, const BBox &bounds, AccelStructures accel)
	{
		const AccelType	  types[]	   = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		const char *const grid_names[] = {"uniform", "two-level", "irregular"};
		const char *const set_names[]  = {"near", "inside", "far"};
		size_t			  mismatches   = 0;
		for (uint32_t k : kKNearestKs)
		{
			double throughput[3][3] = {};
			for (int set = 0; set < 3; ++set)
			{
				std::vector<Vec3f> host_queries = MakeClosestPointQueries(flat, bounds, set);
				host_queries.resize(kKNearestQueryCount);
				thrust::device_vector<Vec3f>			  queries(host_queries);
				thrust::device_vector<ClosestPointRecord> records(size_t(kKNearestQueryCount) * k);
				std::vector<ClosestPointRecord>			  reference;
				for (int t = 0; t < 3; ++t)
				{
					accel.type = types[t];
					FindKNearestBatch(view, accel, Raw(queries), kKNearestQueryCount, k, INFINITY, Raw(records));
					Synchronize(Raw(records));
					float runs[kClosestPointRuns];
					for (float &run : runs)
					{
						const auto begin = std::chrono::steady_clock::now();
						FindKNearestBatch(view, accel, Raw(queries), kKNearestQueryCount, k, INFINITY, Raw(records));
						Synchronize(Raw(records));
						run =
							std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
						if (!std::isfinite(run) || run <= 0)
							throw std::runtime_error("Invalid measured duration");
					}
					std::sort(runs, runs + kClosestPointRuns);
					throughput[set][t] = kKNearestQueryCount / (1000.0 * runs[kClosestPointRuns / 2]);

					const thrust::host_vector<ClosestPointRecord> host_records = records;
					if (t == 0)
					{
						reference.assign(host_records.begin(), host_records.end());
						std::vector<ClosestPointRecord> expected(k);
						for (uint32_t sample = 0; sample < 16; ++sample)
						{
							const uint32_t i = sample * (kKNearestQueryCount / 16);
							FindKNearestBruteForce<Triangle>(flat.View(),
															 host_queries[i],
															 k,
															 INFINITY,
															 expected.data());
							// Host and device round differently; near sliver triangles the error grows with
							// the coordinates rather than the distance.
							const float tolerance = 1e-5f * std::max(1.0f, Length(host_queries[i]));
							for (uint32_t j = 0; j < k; ++j)
							{
								const ClosestPointRecord &got = reference[size_t(i) * k + j];
								mismatches +=
									got.prim == 0xffffffffu || std::fabs(got.distance - expected[j].distance) >
																   tolerance + 1e-5f * expected[j].distance;
							}
						}
					}
					else
						for (size_t i = 0; i < host_records.size(); ++i)
							mismatches += !SameClosestPoint(host_records[i], reference[i]);
				}
			}
			std::printf("\nk-nearest (k=%u, %u queries) throughput, Mq/s (speedup over uniform)\n%-7s",
						k,
						kKNearestQueryCount,
						"Set");
			for (const char *name : grid_names)
				std::printf(" %20s", name);
			std::printf("\n");
			for (int set = 0; set < 3; ++set)
			{
				std::printf("%-7s", set_names[set]);
				for (int t = 0; t < 3; ++t)
				{
					char cell[32];
					std::snprintf(cell,
								  sizeof(cell),
								  "%.2f (%.2fx)",
								  throughput[set][t],
								  throughput[set][t] / throughput[set][0]);
					std::printf(" %20s", cell);
				}
				std::printf("\n");
			}
		}
		if (mismatches)
		{
			std::printf("k-nearest mismatches: %zu (grids vs uniform + 16 oracle queries/set)\n", mismatches);
			throw std::runtime_error("k-nearest correctness check failed");
		}
	}

	void RunClosestPoint(const FlatScene &flat, const SceneView &view, const BBox &bounds)
	{
		const UniformGrid	uniform(view);
		const TwoLevelGrid	two_level(view, kClosestPointTopLevelDensity, kClosestPointLeafDensity);
		const IrregularGrid irregular(view, kClosestPointTopLevelDensity, kClosestPointLeafDensity);
		AccelStructures		accel;
		accel.uniform_grid			   = uniform.View();
		accel.two_level_grid		   = two_level.ClosestPointView();
		accel.irregular_grid		   = irregular.View();
		const AccelType	  types[]	   = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		const char *const grid_names[] = {"uniform", "two-level", "irregular"};
		const char *const set_names[]  = {"near", "inside", "far"};

		if constexpr (kPrintClosestPointDetails)
		{
			std::printf("\nClosest point: %u queries per set, median of %d timed runs; two-level and irregular "
						"grids at top=%g, leaf=%g (uniform at its default); counts per query (mean/p50/p99/max)\n",
						kQueryCount,
						kClosestPointRuns,
						kClosestPointTopLevelDensity,
						kClosestPointLeafDensity);
			std::printf("%-10s %-7s %10s %9s %30s %30s %10s\n",
						"Grid",
						"Set",
						"ms",
						"Mq/s",
						"cells examined",
						"primitives tested",
						"leaves");
		}
		size_t					 mismatches		  = 0;
		double					 throughput[3][3] = {};
		std::vector<std::string> breakdown_rows;
		for (int set = 0; set < 3; ++set)
		{
			const std::vector<Vec3f>					host_queries = MakeClosestPointQueries(flat, bounds, set);
			thrust::device_vector<Vec3f>				queries(host_queries);
			thrust::device_vector<ClosestPointRecord>	records(kQueryCount);
			thrust::device_vector<ClosestPointCounters> counters(kQueryCount);
			std::vector<ClosestPointRecord>				reference;
			for (int t = 0; t < 3; ++t)
			{
				accel.type = types[t];
				// The counting pass doubles as the warm-up.
				FindClosestPoints(view, accel, Raw(queries), kQueryCount, INFINITY, Raw(records), Raw(counters));
				Synchronize(Raw(counters));
				float runs[kClosestPointRuns];
				for (float &run : runs)
				{
					const auto begin = std::chrono::steady_clock::now();
					FindClosestPoints(view, accel, Raw(queries), kQueryCount, INFINITY, Raw(records));
					Synchronize(Raw(records));
					run = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
					if (!std::isfinite(run) || run <= 0)
						throw std::runtime_error("Invalid measured duration");
				}
				std::sort(runs, runs + kClosestPointRuns);
				const float ms	   = runs[kClosestPointRuns / 2];
				throughput[set][t] = kQueryCount / (1000.0 * ms);

				if constexpr (kPrintClosestPointDetails)
				{
					const thrust::host_vector<ClosestPointCounters> host_counters = counters;
					std::vector<uint32_t>							cells, prims, leaves;
					for (const ClosestPointCounters &c : host_counters)
					{
						cells.push_back(c.top_cells_examined + c.rows_examined + c.cells_examined);
						prims.push_back(c.prims_tested);
						leaves.push_back(c.leaves_visited);
					}
					const Distribution cell_stats = Summarize(cells), prim_stats = Summarize(prims);
					double			   sums[10] = {};
					for (const ClosestPointCounters &c : host_counters)
					{
						const uint32_t fields[10] = {c.top_cells_examined,
													 c.top_cells_empty,
													 c.rows_examined,
													 c.cells_examined,
													 c.leaves_loaded,
													 c.leaves_empty,
													 c.leaves_skipped,
													 c.nearest_lookups,
													 c.leaves_visited,
													 c.prims_tested};
						for (int f = 0; f < 10; ++f)
							sums[f] += fields[f];
					}
					char breakdown[256];
					int	 length =
						std::snprintf(breakdown, sizeof(breakdown), "%-10s %-7s", grid_names[t], set_names[set]);
					for (int f = 0; f < 10; ++f)
					{
						length += std::snprintf(breakdown + length,
												sizeof(breakdown) - length,
												" %9.1f",
												sums[f] / kQueryCount);
						if (f == 5)
							length += std::snprintf(breakdown + length,
													sizeof(breakdown) - length,
													" %6.1f%%",
													sums[4] > 0 ? 100.0 * sums[5] / sums[4] : 0.0);
					}
					breakdown_rows.push_back(breakdown);
					char cell_text[64], prim_text[64];
					std::snprintf(cell_text,
								  sizeof(cell_text),
								  "%.1f/%u/%u/%u",
								  cell_stats.mean,
								  cell_stats.p50,
								  cell_stats.p99,
								  cell_stats.max);
					std::snprintf(prim_text,
								  sizeof(prim_text),
								  "%.1f/%u/%u/%u",
								  prim_stats.mean,
								  prim_stats.p50,
								  prim_stats.p99,
								  prim_stats.max);
					std::printf("%-10s %-7s %10.3f %9.2f %30s %30s %10.1f\n",
								grid_names[t],
								set_names[set],
								ms,
								kQueryCount / (1000.0 * ms),
								cell_text,
								prim_text,
								Summarize(leaves).mean);
				}

				const thrust::host_vector<ClosestPointRecord> host_records = records;
				if (t == 0)
				{
					reference.assign(host_records.begin(), host_records.end());
					for (uint32_t sample = 0; sample < 64; ++sample)
					{
						const uint32_t	   i = sample * (kQueryCount / 64);
						ClosestPointRecord expected;
						FindClosestPointBruteForce<Triangle>(flat.View(), host_queries[i], INFINITY, &expected);
						mismatches += !SameClosestPoint(reference[i], expected);
					}
				}
				else
					for (uint32_t i = 0; i < kQueryCount; ++i)
						mismatches += !SameClosestPoint(host_records[i], reference[i]);
			}
		}
		if constexpr (kPrintClosestPointDetails)
		{
			std::printf(
				"\nClosest-point work per query (means). Distance tests: top-level cells, irregular voxel rows, "
				"leaf cells (irregular: voxel steps). Loaded leaf cells: empty ones, non-empty ones left to "
				"another run (irregular), extra voxel lookups for that choice, and ones tested\n");
			std::printf("%-10s %-7s %9s %9s %9s %9s %9s %9s %7s %9s %9s %9s %9s\n",
						"Grid",
						"Set",
						"top",
						"top empty",
						"rows",
						"cells",
						"loaded",
						"empty",
						"empty%",
						"skipped",
						"nearest",
						"visited",
						"prims");
			for (const std::string &row : breakdown_rows)
				std::printf("%s\n", row.c_str());
		}

		std::printf("\nClosest-point throughput, Mq/s (speedup over uniform)\n%-7s", "Set");
		for (const char *name : grid_names)
			std::printf(" %20s", name);
		std::printf("\n");
		for (int set = 0; set < 3; ++set)
		{
			std::printf("%-7s", set_names[set]);
			for (int t = 0; t < 3; ++t)
			{
				char cell[32];
				std::snprintf(cell,
							  sizeof(cell),
							  "%.2f (%.2fx)",
							  throughput[set][t],
							  throughput[set][t] / throughput[set][0]);
				std::printf(" %20s", cell);
			}
			std::printf("\n");
		}
		if (mismatches)
		{
			std::printf("Closest-point mismatches: %zu (grids vs uniform + 64 oracle queries/set)\n", mismatches);
			throw std::runtime_error("Closest-point correctness check failed");
		}
		RunKNearest(flat, view, bounds, accel);
	}

	void Run(const char *path)
	{
		Scene scene;
		if (!LoadObj(path, &scene))
			throw std::runtime_error("Cannot load scene");
		const BBox	 bounds			  = scene.WorldBounds();
		const double limit_squared	  = 0.01 * SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
		double		 max_edge_squared = 0;
		for (const Mesh &mesh : scene.meshes)
		{
			for (size_t tri = 0; tri < mesh.indices.size(); tri += 3)
			{
				for (size_t edge = 0; edge < 3; ++edge)
				{
					const double length = SceneEdgeLengthSquared(mesh.positions[mesh.indices[tri + edge]],
																 mesh.positions[mesh.indices[tri + (edge + 1) % 3]]);
					if (!std::isfinite(length) || length > limit_squared)
						throw std::runtime_error("Scene subdivision edge limit violated");
					max_edge_squared = std::max(max_edge_squared, length);
				}
			}
		}
		const FlatScene flat = FlattenScene(scene);
		if (flat.material_ids.empty())
			throw std::runtime_error("Scene contains no triangles");
		DeviceScene		 device(flat);
		const SceneView	 view = device.View();
		std::vector<Ray> rays;
		rays.reserve(kRayCount);
		Rng rng(20261021, 17);
		for (uint32_t i = 0; i < kRayCount; ++i)
		{
			const float x = rng.NextFloat(), y = rng.NextFloat(), z = rng.NextFloat();
			const Vec3f origin = bounds.min_bound + bounds.Extent() * Vec3f(x, y, z);
			const float dz	   = 2 * rng.NextFloat() - 1;
			const float phi	   = 6.283185307179586f * rng.NextFloat();
			const float r	   = std::sqrt(std::max(0.0f, 1 - dz * dz));
			rays.emplace_back(origin, Vec3f(r * std::cos(phi), r * std::sin(phi), dz));
		}
		thrust::device_vector<Ray> device_rays(rays);
		const SahCalibration	   calibration	   = MeasureSahCalibration();
		const SahCostConstants	  &uniform_costs   = calibration.uniform;
		const SahCostConstants	  &two_level_costs = calibration.two_level;
		const SahCostConstants	  &irregular_costs = calibration.irregular;
		std::vector<Measurement>   results;
		auto					   label = [](const char *name, float top, float leaf)
		{
			char text[96];
			if (top > 0)
				std::snprintf(text, sizeof(text), "%s (top=%g, leaf=%g)", name, top, leaf);
			else
				std::snprintf(text, sizeof(text), "%s (density=%g)", name, leaf);
			return std::string(text);
		};
		results.push_back(Measure(
			label("uniform", 0, UniformGrid::kDefaultDensity),
			[&] { return std::make_unique<UniformGrid>(view); },
			view,
			bounds,
			uniform_costs,
			device_rays));
		results.push_back(Measure(
			label("two-level", TwoLevelGrid::kDefaultTopLevelDensity, TwoLevelGrid::kDefaultSecondLevelDensity),
			[&] { return std::make_unique<TwoLevelGrid>(view); },
			view,
			bounds,
			two_level_costs,
			device_rays));
		results.push_back(Measure(
			label("irregular", IrregularGrid::kDefaultTopLevelDensity, IrregularGrid::kDefaultSecondLevelDensity),
			[&] { return std::make_unique<IrregularGrid>(view); },
			view,
			bounds,
			irregular_costs,
			device_rays));

		std::printf("\nScene: %s (%u triangles), %u random closest-hit rays\n", path, view.primitive_count, kRayCount);
		std::printf("Subdivision: edge limit %.12g, maximum edge %.12g (0.1 x scene diagonal)\n",
					std::sqrt(limit_squared),
					std::sqrt(max_edge_squared));
		PrintSahCalibration(calibration);
		std::printf("%-38s %12s %12s %12s %14s %12s\n",
					"Grid (default densities)",
					"Build ms",
					"Random ms",
					"Mrays/s",
					"Memory MiB",
					"SAH cost");
		for (const auto &result : results)
			std::printf("%-38s %12.3f %12.3f %12.2f %14.3f %12.3f\n",
						result.name.c_str(),
						result.build_ms,
						result.trace_ms,
						result.Mrays(),
						result.memory_bytes / 1048576.0,
						result.sah_cost);
		const auto &u	  = results[0];
		const auto &t	  = results[1];
		const auto &g	  = results[2];
		auto		order = [](const char *metric, bool matches)
		{
			if (!matches)
				std::printf("Ordering uniform < two-level < irregular (%s): NOT OBSERVED\n", metric);
		};
		order("build time", u.build_ms < t.build_ms && t.build_ms < g.build_ms);
		order("random throughput", u.Mrays() < t.Mrays() && t.Mrays() < g.Mrays());
		order("memory", u.memory_bytes < t.memory_bytes && t.memory_bytes < g.memory_bytes);
		order("SAH cost", u.sah_cost > t.sah_cost && t.sah_cost > g.sah_cost);

		size_t mismatches = 0;
		for (const auto &result : results)
		{
			for (uint32_t i = 0; i < kRayCount; ++i)
				if (!SameHit(result.hits[i], u.hits[i]))
					++mismatches;
			for (uint32_t sample = 0; sample < 256; ++sample)
			{
				const uint32_t i = sample * (kRayCount / 256);
				HitRecord	   hit;
				const float	   expected = IntersectSceneBruteForce(flat.View(), rays[i], &hit) ? hit.t : -1;
				if (!SameHit(result.hits[i], expected))
					++mismatches;
			}
		}
		if (mismatches)
		{
			std::printf("Closest-hit mismatches: %zu (full-batch comparison + 256 oracle rays/grid)\n", mismatches);
			throw std::runtime_error("Closest-hit correctness check failed");
		}
		RunClosestPoint(flat, view, bounds);
	}

} // namespace

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
			throw std::runtime_error("Usage: happyray_accel_performance_tests scene.obj");
		Run(argv[1]);
		return EXIT_SUCCESS;
	}
	catch (const std::exception &error)
	{
		std::fprintf(stderr, "Acceleration performance test failed: %s\n", error.what());
		return EXIT_FAILURE;
	}
}
