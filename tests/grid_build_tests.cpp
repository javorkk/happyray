#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "../src/accel/irregular_grid.h"
#include "../src/accel/two_level_grid.h"
#include "../src/accel/uniform_grid.h"
#include "../src/app/attractor_scene.h"

namespace
{

	int g_failures = 0;

	void Check(bool condition, const char *message)
	{
		if (!condition)
		{
			std::fprintf(stderr, "FAILED: %s\n", message);
			++g_failures;
		}
	}

	struct CheckTraversalFunctor
	{
		happyray::UniformGridView  uniform;
		happyray::TwoLevelGridView two_level;
		const happyray::Ray		  *rays;
		uint32_t				  *results;

		HAPPYRAY_HD void operator()(uint32_t i) const
		{
			uint32_t *out	= results + i * 20;
			uint32_t  count = 0;
			happyray::detail::TraverseGrid(uniform,
										   rays[i],
										   [&](happyray::CellRange range, float *)
										   {
											   out[1 + count++] = range.start;
											   return false;
										   });
			out[0] = count;
			count  = 0;
			happyray::detail::TraverseGrid(two_level,
										   rays[i],
										   [&](happyray::CellRange range, float *)
										   {
											   out[11 + count++] = range.start;
											   return false;
										   });
			out[10] = count;
		}
	};

	void CheckTraversalBoundaries()
	{
		using namespace happyray;
		const std::vector<Ray>					 rays	  = {Ray(Vec3f(-1, 0.5f, 0.5f), Vec3f(1, 0, 0)),
															 Ray(Vec3f(3, 0.5f, 0.5f), Vec3f(-1, -0.0f, 0)),
															 Ray(Vec3f(-1), Vec3f(1)),
															 Ray(Vec3f(3), Vec3f(-1)),
															 Ray(Vec3f(0.5f), Vec3f(0, 1, 0)),
															 Ray(Vec3f(0.5f), Vec3f(0, 0, 1)),
															 Ray(Vec3f(-1, 3, 0.5f), Vec3f(1, 0, 0)),
															 Ray(Vec3f(0.5f), Vec3f(1, 0, 0), 0.0f, 0.49f),
															 Ray(Vec3f(0.5f), Vec3f(1, 0, 0), 0.0f, 0.5f),
															 Ray(Vec3f(-1, 0.5f, 0.5f), Vec3f(1, 0, 0), 2.1f, 3.0f)};
		const std::vector<std::vector<uint32_t>> expected = {
			{0, 1},
			{1, 0},
			{0, 4, 6, 7},
			{7, 3, 1, 0},
			{0, 2},
			{0, 4},
			{},
			{0},
			{0, 1},
			{1}
		   };
		std::vector<CellRange>	  ranges(8);
		std::vector<TopLevelCell> top_cells(8);
		for (uint32_t i = 0; i < 8; ++i)
		{
			ranges[i]					   = {i, i + 1};
			top_cells[i].cell_range_offset = i;
		}
		thrust::device_vector<CellRange>	device_ranges(ranges);
		thrust::device_vector<TopLevelCell> device_top(top_cells);
		thrust::device_vector<Ray>			device_rays(rays);
		thrust::device_vector<uint32_t>		results(rays.size() * 20, 0u);
		UniformGridView						uniform;
		uniform.bounds_min = Vec3f(0);
		uniform.bounds_max = Vec3f(2);
		uniform.cell_size = uniform.inv_cell_size = Vec3f(1);
		uniform.resolution						  = Vec3i(2);
		uniform.cell_ranges						  = thrust::raw_pointer_cast(device_ranges.data());
		TwoLevelGridView two;
		two.top_level				 = {uniform.bounds_min,
										uniform.bounds_max,
										uniform.cell_size,
										uniform.inv_cell_size,
										uniform.resolution};
		two.top_level_cells			 = thrust::raw_pointer_cast(device_top.data());
		two.second_level_cell_ranges = uniform.cell_ranges;
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(rays.size())),
						 CheckTraversalFunctor{uniform,
											   two,
											   thrust::raw_pointer_cast(device_rays.data()),
											   thrust::raw_pointer_cast(results.data())});
		const thrust::host_vector<uint32_t> host = results;
		for (size_t i = 0; i < rays.size(); ++i)
		{
			for (size_t level : {0u, 10u})
			{
				Check(host[i * 20 + level] == expected[i].size(), "DDA boundary visit count");
				for (size_t j = 0; j < expected[i].size(); ++j)
					Check(host[i * 20 + level + 1 + j] == expected[i][j], "DDA boundary visit order");
			}
		}

		// Exercise indexed stepping within refined leaves, not just 1x1x1 leaves.
		std::vector<CellRange> refined_ranges(64);
		for (uint32_t i = 0; i < 64; ++i)
			refined_ranges[i] = {i, i + 1};
		for (uint32_t i = 0; i < 8; ++i)
		{
			top_cells[i].resolution		   = 2;
			top_cells[i].cell_range_offset = i * 8;
		}
		device_ranges				 = refined_ranges;
		device_top					 = top_cells;
		uniform.resolution			 = Vec3i(4);
		uniform.cell_size			 = Vec3f(0.5f);
		uniform.inv_cell_size		 = Vec3f(2);
		uniform.cell_ranges			 = thrust::raw_pointer_cast(device_ranges.data());
		two.top_level_cells			 = thrust::raw_pointer_cast(device_top.data());
		two.second_level_cell_ranges = uniform.cell_ranges;
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(2),
						 CheckTraversalFunctor{uniform,
											   two,
											   thrust::raw_pointer_cast(device_rays.data()),
											   thrust::raw_pointer_cast(results.data())});
		const thrust::host_vector<uint32_t> refined				   = results;
		const uint32_t						uniform_expected[2][4] = {
			 {20, 21, 22, 23},
			 {23, 22, 21, 20}
		 };
		const uint32_t two_expected[2][4] = {
			{6,	7,  14, 15},
			{15, 14, 7,	6 }
		  };
		for (size_t i = 0; i < 2; ++i)
		{
			Check(refined[i * 20] == 4 && refined[i * 20 + 10] == 4, "Refined DDA visit count");
			for (size_t j = 0; j < 4; ++j)
			{
				Check(refined[i * 20 + 1 + j] == uniform_expected[i][j], "Refined uniform DDA visit order");
				Check(refined[i * 20 + 11 + j] == two_expected[i][j], "Refined two-level DDA visit order");
			}
		}
	}

	struct OverlapCase
	{
		happyray::Vec3f a, b, c;
		bool			expected;
	};

	struct CheckOverlapFunctor
	{
		happyray::SceneView scene;
		uint32_t		   *results;

		HAPPYRAY_HD void operator()(uint32_t i) const
		{
			results[i] = happyray::detail::CellIntersection<happyray::Triangle>(scene, i)(happyray::Vec3f(0, 0, 0),
																						  happyray::Vec3f(2, 2, 2));
		}
	};

	void CheckTriangleCellIntersection()
	{
		using happyray::Vec3f;
		const std::vector<OverlapCase> cases = {
			{Vec3f(0,	  0, 0), Vec3f(0.5f,	 0,	0), Vec3f(0,		 0.5f, 0), true },
			{Vec3f(2,	  0, 0), Vec3f(2,	  1,	 0), Vec3f(2,	  0,	 1), false}, // Box axis.
			{Vec3f(4,	  0, 0), Vec3f(0,	  4,	 0), Vec3f(0,	  0,	 4), false}, // Plane.
			{Vec3f(2.5f,	 0, 0), Vec3f(0,		 2.5f, 0), Vec3f(2.5f,   0,	  1), false}, // Edge axis.
			{Vec3f(-4,	   0, 0), Vec3f(4,	   0,	  0), Vec3f(0,	   4,	  0), true }, // All vertices outside.
			{Vec3f(1,	  0, 0), Vec3f(1,	  1,	 0), Vec3f(1,	  0,	 1), true }, // Face contact.
			{Vec3f(1,	  1, 1), Vec3f(2,	  1,	 1), Vec3f(1,	  2,	 1), true }, // Corner contact.
			{Vec3f(1.005f, 0, 0), Vec3f(1.005f, 1,	   0), Vec3f(1.005f, 0,	1), true },
			{Vec3f(0,	  0, 0), Vec3f(0,	  0,	 0), Vec3f(0,	  0,	 0), true }, // Point.
			{Vec3f(2,	  2, 2), Vec3f(2,	  2,	 2), Vec3f(2,	  2,	 2), false},
			{Vec3f(-2,	   0, 0), Vec3f(0,	   0,	  0), Vec3f(2,	   0,	  0), true }, // Line.
			{Vec3f(2,	  0, 0), Vec3f(3,	  0,	 0), Vec3f(4,	  0,	 0), false},
		};
		std::vector<Vec3f>	  positions;
		std::vector<uint32_t> indices;
		for (const auto &test : cases)
		{
			for (const Vec3f &vertex : {test.a, test.b, test.c})
			{
				indices.push_back(static_cast<uint32_t>(positions.size()));
				positions.push_back(vertex);
			}
		}
		happyray::SceneView host_scene;
		host_scene.positions	   = positions.data();
		host_scene.indices		   = indices.data();
		host_scene.primitive_count = static_cast<uint32_t>(cases.size());
		const thrust::device_vector<Vec3f>	  device_positions(positions.begin(), positions.end());
		const thrust::device_vector<uint32_t> device_indices(indices.begin(), indices.end());
		happyray::SceneView					  scene = host_scene;
		scene.positions								= thrust::raw_pointer_cast(device_positions.data());
		scene.indices								= thrust::raw_pointer_cast(device_indices.data());
		thrust::device_vector<uint32_t> results(cases.size());
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(scene.primitive_count),
						 CheckOverlapFunctor{scene, thrust::raw_pointer_cast(results.data())});
		const thrust::host_vector<uint32_t> actual = results;
		for (uint32_t i = 0; i < cases.size(); ++i)
		{
			Check(
				happyray::detail::CellIntersection<happyray::Triangle>(host_scene, i)(Vec3f(0, 0, 0), Vec3f(2, 2, 2)) ==
					cases[i].expected,
				"host triangle/cell overlap");
			Check(actual[i] == cases[i].expected, "backend triangle/cell overlap");
		}
	}

	template <typename T> thrust::host_vector<T> Download(const T *data, uint32_t count)
	{
		const auto begin = thrust::device_pointer_cast(data);
		return thrust::host_vector<T>(begin, begin + count);
	}

	void CheckFilteredFragments(const happyray::SceneView &scene, const happyray::UniformGridView &grid)
	{
		const uint32_t num_cells	   = grid.resolution.x * grid.resolution.y * grid.resolution.z;
		const auto	   ranges		   = Download(grid.cell_ranges, num_cells);
		uint32_t	   reference_count = 0;
		for (const auto &range : ranges)
		{
			reference_count = std::max(reference_count, range.end);
		}
		const auto			 refs = Download(grid.prim_indices, reference_count);
		const happyray::BBox box  = happyray::ComputePrimitiveBBox<happyray::Triangle>(scene, 0);
		happyray::Vec2i		 x_range, y_range, z_range;
		happyray::ComputeCellRange(box,
								   grid.bounds_min,
								   grid.inv_cell_size,
								   grid.resolution,
								   &x_range,
								   &y_range,
								   &z_range);
		const happyray::detail::CellIntersection<happyray::Triangle> intersects(scene, 0);
		uint32_t													 expected_refs = 0;
		for (int z = 0; z < grid.resolution.z; ++z)
		{
			for (int y = 0; y < grid.resolution.y; ++y)
			{
				for (int x = 0; x < grid.resolution.x; ++x)
				{
					const uint32_t cell = happyray::FlattenCellIndex(x, y, z, grid.resolution.x, grid.resolution.y);
					const bool	   expected =
						x >= x_range.x && x <= x_range.y && y >= y_range.x && y <= y_range.y && z >= z_range.x &&
						z <= z_range.y &&
						intersects(grid.bounds_min + happyray::Vec3f(x + 0.5f, y + 0.5f, z + 0.5f) * grid.cell_size,
								   grid.cell_size);
					Check(ranges[cell].end - ranges[cell].start == static_cast<uint32_t>(expected),
						  "writer/extractor must retain exactly the intersecting candidates");
					expected_refs += expected;
				}
			}
		}
		Check(reference_count == expected_refs, "uniform ranges cover SAT references");
		Check(expected_refs > 0 && expected_refs < num_cells, "BBox false positives removed");
		for (uint32_t tri : refs)
			Check(tri == 0, "filtered triangle ID");
	}

	void
	CheckSatCellRanges(const happyray::SceneView &scene, const happyray::GridRangeView &grid, uint32_t reference_count)
	{
		const uint32_t		 num_cells = grid.resolution.x * grid.resolution.y * grid.resolution.z;
		const auto			 ranges	   = Download(grid.cell_ranges, num_cells);
		const auto			 refs	   = Download(grid.prim_indices, reference_count);
		const happyray::BBox box	   = happyray::ComputePrimitiveBBox<happyray::Triangle>(scene, 0);
		happyray::Vec2i		 x_range, y_range, z_range;
		happyray::ComputeCellRange(box,
								   grid.bounds_min,
								   grid.inv_cell_size,
								   grid.resolution,
								   &x_range,
								   &y_range,
								   &z_range);
		const happyray::detail::CellIntersection<happyray::Triangle> intersects(scene, 0);
		uint32_t													 expected_refs	 = 0;
		uint32_t													 aabb_candidates = 0;
		for (int z = 0; z < grid.resolution.z; ++z)
		{
			for (int y = 0; y < grid.resolution.y; ++y)
			{
				for (int x = 0; x < grid.resolution.x; ++x)
				{
					const uint32_t cell = happyray::FlattenCellIndex(x, y, z, grid.resolution.x, grid.resolution.y);
					const bool in_aabb_range = x >= x_range.x && x <= x_range.y && y >= y_range.x && y <= y_range.y &&
											   z >= z_range.x && z <= z_range.y;
					aabb_candidates += in_aabb_range;
					const bool expected =
						in_aabb_range &&
						intersects(grid.bounds_min + happyray::Vec3f(x + 0.5f, y + 0.5f, z + 0.5f) * grid.cell_size,
								   grid.cell_size);
					Check(ranges[cell].end - ranges[cell].start == static_cast<uint32_t>(expected),
						  "two-level ranges must retain exactly SAT-intersecting cells");
					expected_refs += expected;
				}
			}
		}
		Check(expected_refs == reference_count, "two-level range endpoints cover SAT references");
		Check(expected_refs < aabb_candidates, "SAT rejects AABB-only cell candidates");
		for (uint32_t tri : refs)
			Check(tri == 0, "two-level range triangle ID");
	}

	void CheckFragmentWriters()
	{
		using happyray::Vec3f;
		const std::vector<Vec3f>	positions = {Vec3f(0, 0, 0), Vec3f(1, 1, 0), Vec3f(0, 1, 1)};
		const std::vector<uint32_t> indices	  = {0, 1, 2};
		happyray::SceneView			host_scene;
		host_scene.positions	   = positions.data();
		host_scene.indices		   = indices.data();
		host_scene.primitive_count = 1;
		const thrust::device_vector<Vec3f>	  device_positions(positions.begin(), positions.end());
		const thrust::device_vector<uint32_t> device_indices(indices.begin(), indices.end());
		happyray::SceneView					  scene = host_scene;
		scene.positions								= thrust::raw_pointer_cast(device_positions.data());
		scene.indices								= thrust::raw_pointer_cast(device_indices.data());

		const happyray::UniformGrid uniform(scene, 128);
		CheckFilteredFragments(host_scene, uniform.View());
		Check(uniform.LeafCellCount() == static_cast<uint32_t>(uniform.View().resolution.x) *
											 uniform.View().resolution.y * uniform.View().resolution.z,
			  "uniform leaf count includes every grid cell");
		const auto uniform_ranges		   = Download(uniform.View().cell_ranges, uniform.LeafCellCount());
		uint32_t   uniform_reference_count = 0;
		for (const auto &range : uniform_ranges)
		{
			uniform_reference_count = std::max(uniform_reference_count, range.end);
		}
		Check(uniform.PrimIndexCount() >= uniform_reference_count,
			  "uniform ranges fit in pair storage including padding");

		const happyray::TwoLevelGrid two_level(scene, 128, 128, 0);
		auto						 two_view = two_level.View();
		const uint32_t				 num_top_cells =
			two_view.top_level.resolution.x * two_view.top_level.resolution.y * two_view.top_level.resolution.z;
		const auto top_level_cells	 = Download(two_view.top_level_cells, num_top_cells);
		two_view.top_level_cells	 = top_level_cells.data();
		uint32_t empty_top_cells	 = 0;
		uint32_t occupied_top_cell	 = num_top_cells;
		uint32_t expected_leaf_count = 0;
		for (uint32_t cell = 0; cell < num_top_cells; ++cell)
		{
			const uint32_t resolution = static_cast<uint32_t>(top_level_cells[cell].Resolution());
			expected_leaf_count += resolution * resolution * resolution;
			if (top_level_cells[cell].IsEmpty())
			{
				++empty_top_cells;
				Check(top_level_cells[cell].resolution < 0 && top_level_cells[cell].Resolution() > 0,
					  "empty top-level resolution uses its sign bit");
			}
			else if (occupied_top_cell == num_top_cells)
			{
				const auto	   candidate = happyray::SecondLevelView(two_view, cell);
				const uint32_t candidate_cell_count =
					candidate.resolution.x * candidate.resolution.y * candidate.resolution.z;
				const auto candidate_ranges = Download(candidate.cell_ranges, candidate_cell_count);
				for (const auto &range : candidate_ranges)
				{
					if (range.end > 0)
					{
						occupied_top_cell = cell;
						break;
					}
				}
			}
		}
		Check(two_level.LeafCellCount() == expected_leaf_count,
			  "two-level leaf count includes all second-level cells, including empty cells");
		Check(empty_top_cells > 0 && occupied_top_cell < num_top_cells,
			  "top-level cell metadata distinguishes empty cells");
		const auto second_level_view = happyray::SecondLevelView(two_view, occupied_top_cell);
		Check(second_level_view.resolution.x == second_level_view.resolution.y &&
				  second_level_view.resolution.y == second_level_view.resolution.z,
			  "two-level second grid has equal axis resolutions");
		const uint32_t second_level_cell_count =
			second_level_view.resolution.x * second_level_view.resolution.y * second_level_view.resolution.z;
		const auto second_level_ranges	= Download(second_level_view.cell_ranges, second_level_cell_count);
		uint32_t   second_level_ref_end = 0;
		for (const auto &range : second_level_ranges)
		{
			second_level_ref_end = std::max(second_level_ref_end, range.end);
		}
		CheckSatCellRanges(host_scene, second_level_view, second_level_ref_end);

		const happyray::IrregularGrid irregular(scene, 0.12f, 128, 0, 0, 0);
		const auto					  irregular_view = irregular.View();
		const auto					  info			 = Download(irregular_view.cell_infos, 1).front();
		const uint32_t				  num_cells		 = info.resolution.x * info.resolution.y * info.resolution.z;
		Check(irregular.LeafCellCount() == num_cells, "unmerged irregular leaf count matches finest grid");
		const auto						 cells = Download(irregular_view.local_cells, num_cells);
		std::vector<happyray::CellRange> ranges;
		for (const auto &cell : cells)
		{
			ranges.push_back(cell.references);
		}
		const thrust::device_vector<happyray::CellRange> device_ranges(ranges.begin(), ranges.end());
		happyray::UniformGridView						 finest;
		finest.bounds_min	 = info.bounds_min;
		finest.cell_size	 = info.cell_size;
		finest.inv_cell_size = info.inv_cell_size;
		finest.resolution	 = info.resolution;
		finest.cell_ranges	 = thrust::raw_pointer_cast(device_ranges.data());
		finest.prim_indices	 = irregular_view.prim_indices;
		CheckFilteredFragments(host_scene, finest);

		const happyray::UniformGrid single_cell(scene, 1);
		Check(single_cell.LeafCellCount() == 1, "single-cell uniform leaf count");
		Check(single_cell.PrimIndexCount() == 1, "no rejected sentinel required for final cell");
		const auto single_ranges = Download(single_cell.View().cell_ranges, 1);
		Check(single_ranges[0].start == 0 && single_ranges[0].end == 1, "single occupied-cell range endpoints");
	}

	void CheckCellStarts(const std::vector<uint32_t> &keys, uint32_t num_cells)
	{
		const thrust::device_vector<uint32_t> device_keys(keys.begin(), keys.end());
		thrust::device_vector<uint32_t>		  starts(num_cells + 1, static_cast<uint32_t>(keys.size()));
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(keys.size())),
						 happyray::detail::CellExtractor{thrust::raw_pointer_cast(device_keys.data()),
														 thrust::raw_pointer_cast(starts.data())});
		thrust::inclusive_scan(starts.rbegin(), starts.rend(), starts.rbegin(), thrust::minimum<uint32_t>());
		const thrust::host_vector<uint32_t> actual = starts;
		if (actual.size() != num_cells + 1)
		{
			std::fprintf(stderr, "FAILED: incorrect CSR size\n");
			++g_failures;
			return;
		}
		for (uint32_t cell = 0; cell <= num_cells; ++cell)
		{
			const uint32_t expected =
				static_cast<uint32_t>(std::lower_bound(keys.begin(), keys.end(), cell) - keys.begin());
			if (actual[cell] != expected)
			{
				std::fprintf(stderr,
							 "FAILED: cell %u of %u (%zu pairs): offset %u, expected %u\n",
							 cell,
							 num_cells,
							 keys.size(),
							 actual[cell],
							 expected);
				++g_failures;
			}
		}
	}

	struct DeviceFlatScene
	{
		thrust::device_vector<happyray::Vec3f>		 positions;
		thrust::device_vector<uint32_t>				 indices, material_ids;
		thrust::device_vector<happyray::Point>		 points;
		thrust::device_vector<happyray::Sphere>		 spheres;
		thrust::device_vector<happyray::BezierCurve> curves;
		happyray::SceneView							 view;

		explicit DeviceFlatScene(const happyray::FlatScene &scene)
			: positions(scene.positions.begin(), scene.positions.end()),
			  indices(scene.indices.begin(), scene.indices.end()),
			  material_ids(scene.material_ids.begin(), scene.material_ids.end()),
			  points(scene.points.begin(), scene.points.end()), spheres(scene.spheres.begin(), scene.spheres.end()),
			  curves(scene.curves.begin(), scene.curves.end()), view(scene.View())
		{
			view.positions				 = thrust::raw_pointer_cast(positions.data());
			view.indices				 = thrust::raw_pointer_cast(indices.data());
			view.material_ids			 = thrust::raw_pointer_cast(material_ids.data());
			view.normals				 = nullptr;
			view.object_ids				 = nullptr;
			view.material_diffuse_colors = nullptr;
			view.points					 = thrust::raw_pointer_cast(points.data());
			view.spheres				 = thrust::raw_pointer_cast(spheres.data());
			view.curves					 = thrust::raw_pointer_cast(curves.data());
		}
	};

	// Exact and hybrid cell assignment store identical references; AABB stores a superset.
	void CheckOverlapModeReferenceCounts()
	{
		using happyray::CellOverlapMode;
		const happyray::PrimitiveType types[] = {happyray::PrimitiveType::kPoint,
												 happyray::PrimitiveType::kSphere,
												 happyray::PrimitiveType::kBezierCurve};
		for (happyray::PrimitiveType type : types)
		{
			const DeviceFlatScene scene(happyray::FlattenScene(happyray::MakeAttractorScene(type, 2000)));
			const CellOverlapMode modes[] = {CellOverlapMode::kAabb, CellOverlapMode::kExact, CellOverlapMode::kHybrid};
			uint32_t			  counts[3][3];
			for (int m = 0; m < 3; ++m)
			{
				counts[m][0] = happyray::UniformGrid(scene.view, 3.0f, modes[m]).PrimIndexCount();
				counts[m][1] =
					happyray::TwoLevelGrid(scene.view, 0.12f, 1.2f, 16, modes[m]).SecondLevelPrimIndexCount();
				counts[m][2] = happyray::IrregularGrid(scene.view, 0.12f, 2.4f, 16, 0.7f, 2, modes[m]).PrimIndexCount();
			}
			for (int grid = 0; grid < 3; ++grid)
			{
				Check(counts[1][grid] == counts[2][grid], "exact and hybrid store identical reference counts");
				Check(counts[0][grid] >= counts[1][grid], "AABB references cover exact references");
				Check(counts[1][grid] >= 2000, "every primitive is referenced");
			}
			if (type == happyray::PrimitiveType::kBezierCurve)
				Check(counts[0][0] > counts[1][0], "exact curve assignment rejects AABB-only cells");
		}
	}

} // namespace

int main()
{
	CheckOverlapModeReferenceCounts();
	CheckTraversalBoundaries();
	CheckTriangleCellIntersection();
	CheckFragmentWriters();
	CheckCellStarts({}, 0);
	CheckCellStarts({}, 8);
	CheckCellStarts({0}, 1);
	CheckCellStarts({0, 0, 0}, 1);
	CheckCellStarts({0, 1, 2, 3}, 4);
	CheckCellStarts({3, 3, 6, 6, 6}, 10); // Leading, interior, and trailing empty cells.
	CheckCellStarts({7}, 8);
	CheckCellStarts({0}, 8);
	CheckCellStarts({0, 8191}, 8192);
	CheckCellStarts(std::vector<uint32_t>(4097, 3), 8);

	// Exhaust every combination of empty, singleton, and repeated runs in a small grid.
	for (uint32_t pattern = 0; pattern < 243; ++pattern)
	{
		std::vector<uint32_t> keys;
		uint32_t			  counts = pattern;
		for (uint32_t cell = 0; cell < 5; ++cell)
		{
			keys.insert(keys.end(), counts % 3, cell);
			counts /= 3;
		}
		CheckCellStarts(keys, 5);
	}

	// Multiple runs crossing typical parallel execution block boundaries.
	std::vector<uint32_t> keys;
	for (uint32_t cell = 2; cell < 1024; cell += 7)
	{
		keys.insert(keys.end(), (cell * 17) % 513 + 1, cell);
	}
	CheckCellStarts(keys, 1024);

	std::printf("Grid construction tests: %d failure(s)\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
