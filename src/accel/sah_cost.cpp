// MeasureSahCalibration(): times the real traversal and intersection routines on synthetic
// scenes. Pure Thrust, so it builds for both the CUDA and TBB backends.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "sah_cost.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "../core/bbox.h"
#include "../core/random.h"
#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace happyray
{
	namespace
	{

		// 4^3 top-level cells x 16^3 local cells, so nested traversal crosses top-level cells; the
		// uniform mock uses the product as its flat resolution.
		constexpr int kMockTopLevelResolution	 = 4;  // Top-level cells per axis (two-level/irregular).
		constexpr int kMockSecondLevelResolution = 16; // Second-level/local cells per axis, per top cell.
		constexpr int kMockResolution			 = kMockTopLevelResolution * kMockSecondLevelResolution;
		// The irregular mock models leaves that each replace 2x2x2 cells of a denser initial grid;
		// a ray crosses about 4 of those, so each visited leaf counts as 4 steps.
		constexpr uint32_t kMockIrregularCellsPerLeaf = 4;
		// Rays are generated once and read from memory, so timings exclude ray construction.
		constexpr uint32_t kCalibrationRayCount = 1u << 17;
		constexpr int kIntersectionTriangleCount = 100;

		// Rays from a random point on one face of [0,1]^3 to a random point on the opposite face.
		std::vector<Ray> GenerateUnitCubeRays(uint64_t seed, uint32_t count)
		{
			std::vector<Ray> rays(count);
			for (uint32_t i = 0; i < count; ++i)
			{
				Rng		  rng(seed, i);
				const int axis		= static_cast<int>(rng.NextUint32() % 3u);
				float	  near_t[3] = {rng.NextFloat(), rng.NextFloat(), rng.NextFloat()};
				float	  far_t[3]	= {rng.NextFloat(), rng.NextFloat(), rng.NextFloat()};
				near_t[axis]		= 0.0f;
				far_t[axis]			= 1.0f;
				const Vec3f near_point(near_t[0], near_t[1], near_t[2]);
				const Vec3f far_point(far_t[0], far_t[1], far_t[2]);
				Ray			ray(near_point, far_point - near_point);
				ray.t_min = -1.0e-3f; // Origin lies exactly on the entry face.
				ray.t_max = 1.0f + 1.0e-3f;
				rays[i]	  = ray;
			}
			return rays;
		}

		// Built by hand because UniformGrid collapses an empty scene to 1x1x1.
		struct MockUniformGrid
		{
			thrust::device_vector<CellRange> cell_ranges;
			UniformGridView					 view;

			MockUniformGrid(const BBox &bounds, int resolution)
			{
				const auto num_cells = static_cast<uint32_t>(resolution) * resolution * resolution;
				view.bounds_min		 = bounds.min_bound;
				view.bounds_max		 = bounds.max_bound;
				const Vec3f extent	 = bounds.Extent();
				view.cell_size		 = Vec3f(extent.x / resolution, extent.y / resolution, extent.z / resolution);
				view.inv_cell_size	 = Vec3f(1.0f / view.cell_size.x, 1.0f / view.cell_size.y, 1.0f / view.cell_size.z);
				view.resolution		 = Vec3i(resolution, resolution, resolution);
				cell_ranges.assign(num_cells, CellRange{0u, 0u});
				view.cell_ranges  = thrust::raw_pointer_cast(cell_ranges.data());
				view.prim_indices = nullptr;
			}
		};

		struct MakeMockSecondLevelInfoFunctor
		{
			int second_resolution;

			HAPPYRAY_HD TopLevelCell operator()(uint32_t t) const
			{
				TopLevelCell info;
				info.resolution = second_resolution;
				info.cell_range_offset =
					t * static_cast<uint32_t>(second_resolution) * second_resolution * second_resolution;
				return info;
			}
		};

		struct MockTwoLevelGrid
		{
			thrust::device_vector<TopLevelCell> top_level_cells;
			thrust::device_vector<CellRange>	cell_ranges;
			TwoLevelGridView					view;

			MockTwoLevelGrid(const BBox &bounds, int top_resolution, int second_resolution)
			{
				const auto num_top_cells = static_cast<uint32_t>(top_resolution) * top_resolution * top_resolution;
				const auto num_local = static_cast<uint32_t>(second_resolution) * second_resolution * second_resolution;
				const Vec3f extent	 = bounds.Extent();
				const Vec3f top_cell_size(extent.x / top_resolution,
										  extent.y / top_resolution,
										  extent.z / top_resolution);

				view.top_level.bounds_min = bounds.min_bound;
				view.top_level.bounds_max = bounds.max_bound;
				view.top_level.cell_size  = top_cell_size;
				view.top_level.inv_cell_size =
					Vec3f(1.0f / top_cell_size.x, 1.0f / top_cell_size.y, 1.0f / top_cell_size.z);
				view.top_level.resolution = Vec3i(top_resolution, top_resolution, top_resolution);
				// Mark all cells occupied so traversal descends into each nested grid.

				top_level_cells.resize(num_top_cells);
				thrust::transform(thrust::counting_iterator<uint32_t>(0),
								  thrust::counting_iterator<uint32_t>(num_top_cells),
								  top_level_cells.begin(),
								  MakeMockSecondLevelInfoFunctor{second_resolution});

				cell_ranges.assign(num_top_cells * num_local, CellRange{0u, 0u});

				view.top_level_cells		   = thrust::raw_pointer_cast(top_level_cells.data());
				view.second_level_cell_ranges  = thrust::raw_pointer_cast(cell_ranges.data());
				view.second_level_prim_indices = nullptr;
			}
		};

		// Pre-merge cell for local voxel `v` of top-level cell `t`, in shared-lattice coordinates.
		struct MakeMockIrregularCellFunctor
		{
			int top_resolution_x, top_resolution_y;
			int local_resolution;

			HAPPYRAY_HD IrregularCell operator()(uint32_t v) const
			{
				const auto num_local   = static_cast<uint32_t>(local_resolution) * local_resolution * local_resolution;
				const uint32_t t	   = v / num_local;
				const uint32_t local_v = v % num_local;

				const int local_res_xy = local_resolution * local_resolution;
				const int lz		   = static_cast<int>(local_v) / local_res_xy;
				const int lrem		   = static_cast<int>(local_v) % local_res_xy;
				const int ly		   = lrem / local_resolution;
				const int lx		   = lrem % local_resolution;

				const int top_res_xy = top_resolution_x * top_resolution_y;
				const int tz		 = static_cast<int>(t) / top_res_xy;
				const int trem		 = static_cast<int>(t) % top_res_xy;
				const int ty		 = trem / top_resolution_x;
				const int tx		 = trem % top_resolution_x;

				IrregularCell cell;
				cell.min = Vec3i(tx * local_resolution + lx, ty * local_resolution + ly, tz * local_resolution + lz);
				cell.max = cell.min + Vec3i(1, 1, 1);
				cell.references = {0u, 0u};
				return cell;
			}
		};

		// All top-level cells share one local resolution, so shift == 0.
		struct MakeMockIrregularCellInfoFunctor
		{
			Vec3f top_bounds_min;
			Vec3f top_cell_size;
			int	  top_resolution_x, top_resolution_y;
			int	  local_resolution;

			HAPPYRAY_HD IrregularCellInfo operator()(uint32_t t) const
			{
				const int	res_xy = top_resolution_x * top_resolution_y;
				const int	tz	   = static_cast<int>(t) / res_xy;
				const int	rem	   = static_cast<int>(t) % res_xy;
				const int	ty	   = rem / top_resolution_x;
				const int	tx	   = rem % top_resolution_x;
				const Vec3f cell_min(top_bounds_min.x + tx * top_cell_size.x,
									 top_bounds_min.y + ty * top_cell_size.y,
									 top_bounds_min.z + tz * top_cell_size.z);

				IrregularCellInfo info;
				info.bounds_min		 = cell_min;
				info.cell_size		 = Vec3f(top_cell_size.x / local_resolution,
										 top_cell_size.y / local_resolution,
										 top_cell_size.z / local_resolution);
				info.inv_cell_size	 = Vec3f(1.0f / info.cell_size.x, 1.0f / info.cell_size.y, 1.0f / info.cell_size.z);
				info.resolution		 = Vec3i(local_resolution, local_resolution, local_resolution);
				const auto num_local = static_cast<uint32_t>(local_resolution) * local_resolution * local_resolution;
				info.entries_offset	 = t * num_local;
				info.shift			 = Vec3i(0, 0, 0);
				info.global_offset	 = Vec3i(tx * local_resolution, ty * local_resolution, tz * local_resolution);
				return info;
			}
		};

		struct MakeMockTopCellFunctor
		{
			HAPPYRAY_HD IrregularTopCell operator()(const IrregularCellInfo &info) const
			{
				return MakeIrregularTopCell(info);
			}
		};

		struct MockIrregularGrid
		{
			thrust::device_vector<IrregularCellInfo> infos;
			thrust::device_vector<IrregularTopCell>	 top_cells;
			thrust::device_vector<uint32_t>			 entries;
			thrust::device_vector<IrregularCell>	 cells;
			IrregularGridView						 view;

			MockIrregularGrid(const BBox &bounds, int top_resolution, int local_resolution)
			{
				const auto num_top_cells = static_cast<uint32_t>(top_resolution) * top_resolution * top_resolution;
				const auto num_local_per_top =
					static_cast<uint32_t>(local_resolution) * local_resolution * local_resolution;
				const uint32_t num_entries = num_top_cells * num_local_per_top;
				const Vec3f	   extent	   = bounds.Extent();
				const Vec3f	   top_cell_size(extent.x / top_resolution,
											 extent.y / top_resolution,
											 extent.z / top_resolution);

				view.top_level.bounds_min = bounds.min_bound;
				view.top_level.bounds_max = bounds.max_bound;
				view.top_level.cell_size  = top_cell_size;
				view.top_level.inv_cell_size =
					Vec3f(1.0f / top_cell_size.x, 1.0f / top_cell_size.y, 1.0f / top_cell_size.z);
				view.top_level.resolution	= Vec3i(top_resolution, top_resolution, top_resolution);
				view.top_level.cell_ranges	= nullptr;
				view.top_level.prim_indices = nullptr;

				infos.resize(num_top_cells);
				thrust::transform(thrust::counting_iterator<uint32_t>(0),
								  thrust::counting_iterator<uint32_t>(num_top_cells),
								  infos.begin(),
								  MakeMockIrregularCellInfoFunctor{bounds.min_bound,
																   top_cell_size,
																   top_resolution,
																   top_resolution,
																   local_resolution});

				top_cells.resize(num_top_cells);
				thrust::transform(infos.begin(), infos.end(), top_cells.begin(), MakeMockTopCellFunctor());

				entries.resize(num_entries);
				thrust::sequence(entries.begin(), entries.end());
				cells.resize(num_entries);
				thrust::transform(thrust::counting_iterator<uint32_t>(0),
								  thrust::counting_iterator<uint32_t>(num_entries),
								  cells.begin(),
								  MakeMockIrregularCellFunctor{top_resolution, top_resolution, local_resolution});

				view.cell_infos			  = thrust::raw_pointer_cast(infos.data());
				view.top_cells			  = thrust::raw_pointer_cast(top_cells.data());
				view.local_entries		  = thrust::raw_pointer_cast(entries.data());
				view.local_cells		  = thrust::raw_pointer_cast(cells.data());
				view.prim_indices		  = nullptr;
				view.global_cell_size	  = Vec3f(top_cell_size.x / local_resolution,
											  top_cell_size.y / local_resolution,
											  top_cell_size.z / local_resolution);
				view.global_resolution	  = Vec3i(local_resolution, local_resolution, local_resolution);
				view.global_inv_cell_size = Vec3f(1.0f / view.global_cell_size.x,
												  1.0f / view.global_cell_size.y,
												  1.0f / view.global_cell_size.z);
				view.global_shift		  = Vec3i(IrregularLog2(local_resolution),
										  IrregularLog2(local_resolution),
										  IrregularLog2(local_resolution));
			}
		};

		// Run each structure's real traversal with a step-counting visitor that never stops early.
		struct UniformStepFunctor
		{
			UniformGridView grid;
			const Ray	   *rays;

			HAPPYRAY_HD float operator()(uint32_t i) const
			{
				uint32_t steps = 0;
				detail::TraverseGrid(grid,
									 rays[i],
									 [&](CellRange, float *) -> bool
									 {
										 ++steps;
										 return false;
									 });
				return static_cast<float>(steps);
			}
		};

		struct TwoLevelStepFunctor
		{
			TwoLevelGridView grid;
			const Ray		*rays;

			HAPPYRAY_HD float operator()(uint32_t i) const
			{
				uint32_t steps = 0;
				detail::TraverseGrid(
					grid,
					rays[i],
					[&](CellRange, float *) -> bool
					{
						++steps;
						return false;
					},
					[&](uint32_t) { ++steps; });
				return static_cast<float>(steps);
			}
		};

		struct IrregularStepFunctor
		{
			IrregularGridView grid;
			const Ray		 *rays;

			HAPPYRAY_HD float operator()(uint32_t i) const
			{
				uint32_t steps = 0;
				detail::TraverseGrid(grid,
									 rays[i],
									 [&](CellRange, float *) -> bool
									 {
										 steps += kMockIrregularCellsPerLeaf;
										 return false;
									 });
				return static_cast<float>(steps);
			}
		};

		// Returns ns per unit (min over a few trials after a warm-up). Divides by the reduction's sum
		// (total steps) unless `known_unit_count` is given; the reduction also keeps the work from
		// being optimized away.
		template <typename Functor> float TimePerUnitNs(Functor functor, uint32_t count, float known_unit_count = 0.0f)
		{
			constexpr int kWarmupTrials = 1;
			constexpr int kTimedTrials	= 2;
			float		  best_ms		= 1e30f;
			float		  total_units	= 0.0f;
			for (int trial = 0; trial < kWarmupTrials + kTimedTrials; ++trial)
			{
				const auto	begin	   = std::chrono::steady_clock::now();
				const float units	   = thrust::transform_reduce(thrust::counting_iterator<uint32_t>(0),
															  thrust::counting_iterator<uint32_t>(count),
															  functor,
															  0.0f,
															  thrust::plus<float>());
				const auto	finish	   = std::chrono::steady_clock::now();
				const float elapsed_ms = std::chrono::duration<float, std::milli>(finish - begin).count();
				if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0 || !std::isfinite(units) || units <= 0)
					throw std::runtime_error("Invalid measured calibration duration");
				total_units = known_unit_count > 0 ? known_unit_count : units;
				if (trial >= kWarmupTrials)
					best_ms = std::min(best_ms, elapsed_ms);
			}
			return best_ms * 1.0e6f / total_units;
		}

		struct HostTriangle
		{
			Vec3f a, b, c;
		};

		// Six diagonal planes through the box (x=y, x+y=1, y=z, ...), so most rays hit something.
		void AppendDiagonalQuads(const BBox &bounds, std::vector<HostTriangle> *out)
		{
			const Vec3f &lo		= bounds.min_bound;
			const Vec3f &hi		= bounds.max_bound;
			auto		 corner = [&](int bx, int by, int bz)
			{ return Vec3f(bx ? hi.x : lo.x, by ? hi.y : lo.y, bz ? hi.z : lo.z); };
			auto add_quad = [&](Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3)
			{
				out->push_back({p0, p1, p2});
				out->push_back({p0, p2, p3});
			};
			add_quad(corner(0, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(0, 0, 1));
			add_quad(corner(1, 0, 0), corner(0, 1, 0), corner(0, 1, 1), corner(1, 0, 1));
			add_quad(corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 1), corner(0, 1, 1));
			add_quad(corner(0, 1, 0), corner(1, 1, 0), corner(1, 0, 1), corner(0, 0, 1));
			add_quad(corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 1), corner(1, 0, 1));
			add_quad(corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 1), corner(0, 0, 1));
		}

		// Repeatedly bisects the longest edge of the largest triangle until `target_count` is reached.
		void SubdivideTriangles(std::vector<HostTriangle> *triangles, int target_count)
		{
			while (static_cast<int>(triangles->size()) < target_count)
			{
				size_t largest				= 0;
				float  largest_area_squared = -1.0f;
				for (size_t i = 0; i < triangles->size(); ++i)
				{
					const HostTriangle &t			 = (*triangles)[i];
					const float			area_squared = LengthSquared(Cross(t.b - t.a, t.c - t.a));
					if (area_squared > largest_area_squared)
					{
						largest_area_squared = area_squared;
						largest				 = i;
					}
				}
				const HostTriangle t  = (*triangles)[largest];
				const float		   ab = LengthSquared(t.a - t.b);
				const float		   bc = LengthSquared(t.b - t.c);
				const float		   ca = LengthSquared(t.c - t.a);
				HostTriangle	   t1, t2;
				if (ab >= bc && ab >= ca)
				{
					const Vec3f m = (t.a + t.b) * 0.5f;
					t1			  = {t.a, m, t.c};
					t2			  = {m, t.b, t.c};
				}
				else if (bc >= ca)
				{
					const Vec3f m = (t.b + t.c) * 0.5f;
					t1			  = {t.a, t.b, m};
					t2			  = {t.a, m, t.c};
				}
				else
				{
					const Vec3f m = (t.c + t.a) * 0.5f;
					t1			  = {t.a, t.b, m};
					t2			  = {m, t.b, t.c};
				}
				(*triangles)[largest] = t1;
				triangles->push_back(t2);
			}
		}

		struct HostMesh
		{
			std::vector<Vec3f>	  positions;
			std::vector<uint32_t> indices;
		};

		HostMesh BuildIntersectionMesh(const BBox &bounds)
		{
			std::vector<HostTriangle> triangles;
			AppendDiagonalQuads(bounds, &triangles);
			SubdivideTriangles(&triangles, kIntersectionTriangleCount);

			HostMesh mesh;
			mesh.positions.reserve(triangles.size() * 3);
			mesh.indices.reserve(triangles.size() * 3);
			for (const HostTriangle &t : triangles)
			{
				const auto base = static_cast<uint32_t>(mesh.positions.size());
				mesh.positions.push_back(t.a);
				mesh.positions.push_back(t.b);
				mesh.positions.push_back(t.c);
				mesh.indices.push_back(base + 0);
				mesh.indices.push_back(base + 1);
				mesh.indices.push_back(base + 2);
			}
			return mesh;
		}

		// Tests every triangle against the ray's full [t_min, t_max] so each test costs the same.
		// Returns the closest t so the work isn't optimized away.
		struct IntersectionTestFunctor
		{
			const Ray	   *rays;
			const Vec3f	   *positions;
			const uint32_t *indices;

			HAPPYRAY_HD float operator()(uint32_t i) const
			{
				const Ray &ray		 = rays[i];
				float	   closest_t = ray.t_max;
				for (int tri = 0; tri < kIntersectionTriangleCount; ++tri)
				{
					const Vec3f &v0 = positions[indices[tri * 3 + 0]];
					const Vec3f &v1 = positions[indices[tri * 3 + 1]];
					const Vec3f &v2 = positions[indices[tri * 3 + 2]];
					float		 t, u, v;
					if (IntersectRayTriangle(ray, v0, v1, v2, &t, &u, &v) && t < closest_t)
						closest_t = t;
				}
				return closest_t;
			}
		};

		// Measured directly; deriving it from the difference of two traversals proved too noisy.
		float MeasureIntersectionCostNs(const thrust::device_vector<Ray> &rays)
		{
			const BBox							  bounds(Vec3f(0, 0, 0), Vec3f(1, 1, 1));
			const HostMesh						  mesh = BuildIntersectionMesh(bounds);
			const thrust::device_vector<Vec3f>	  positions(mesh.positions);
			const thrust::device_vector<uint32_t> indices(mesh.indices);
			const auto							  ray_count = static_cast<uint32_t>(rays.size());
			const float							  intersection_ns =
				TimePerUnitNs(IntersectionTestFunctor{thrust::raw_pointer_cast(rays.data()),
													  thrust::raw_pointer_cast(positions.data()),
													  thrust::raw_pointer_cast(indices.data())},
							  ray_count,
							  static_cast<float>(ray_count) * kIntersectionTriangleCount);
			if (!std::isfinite(intersection_ns) || intersection_ns <= 0)
				throw std::runtime_error("Invalid measured intersection cost");
			return intersection_ns;
		}

		// Each structure gets its own mock grid so its extra indirection shows up in the cost.
		float MeasureUniformTraversalNs(const thrust::device_vector<Ray> &rays)
		{
			const BBox			  bounds(Vec3f(0, 0, 0), Vec3f(1, 1, 1));
			const MockUniformGrid grid(bounds, kMockResolution);
			return TimePerUnitNs(UniformStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
								 static_cast<uint32_t>(rays.size()));
		}

		float MeasureTwoLevelTraversalNs(const thrust::device_vector<Ray> &rays)
		{
			const BBox			   bounds(Vec3f(0, 0, 0), Vec3f(1, 1, 1));
			const MockTwoLevelGrid grid(bounds, kMockTopLevelResolution, kMockSecondLevelResolution);
			return TimePerUnitNs(TwoLevelStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
								 static_cast<uint32_t>(rays.size()));
		}

		float MeasureIrregularTraversalNs(const thrust::device_vector<Ray> &rays)
		{
			const BBox				bounds(Vec3f(0, 0, 0), Vec3f(1, 1, 1));
			const MockIrregularGrid grid(bounds, kMockTopLevelResolution, kMockSecondLevelResolution);
			return TimePerUnitNs(IrregularStepFunctor{grid.view, thrust::raw_pointer_cast(rays.data())},
								 static_cast<uint32_t>(rays.size()));
		}

	} // namespace

	SahCalibration MeasureSahCalibration()
	{
		const thrust::device_vector<Ray> rays(GenerateUnitCubeRays(0x1234ULL, kCalibrationRayCount));
		SahCalibration					 calibration;
		const float						 intersection_ns = MeasureIntersectionCostNs(rays);
		calibration.uniform								 = {MeasureUniformTraversalNs(rays), intersection_ns};
		calibration.two_level							 = {MeasureTwoLevelTraversalNs(rays), intersection_ns};
		calibration.irregular							 = {MeasureIrregularTraversalNs(rays), intersection_ns};
		return calibration;
	}

	void PrintSahCalibration(const SahCalibration &calibration)
	{
		auto print = [](const char *name, const SahCostConstants &costs)
		{
			std::printf("  %-12s traversal %8.3f ns/step, intersection %8.3f ns/test "
						"(ratio %.3f)\n",
						name,
						costs.traversal,
						costs.intersection,
						costs.intersection / costs.traversal);
		};
		std::printf("Measured SAH cost constants (happyray::MeasureSahCalibration(), see"
					" src/accel/sah_cost.h): traversal ns/step, intersection ns/test:\n");
		print("uniform", calibration.uniform);
		print("two-level", calibration.two_level);
		print("irregular", calibration.irregular);
	}

} // namespace happyray
