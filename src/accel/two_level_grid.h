// `View().top_level.resolution`'s product, already public; these fill
// The two-level grid stores sorted primitive ranges per top-level and nested
// cell. Sparse top-level cells use a 1x1x1 nested grid.
//
// TwoLevelGrid is exposed through a plain-C++ interface so it can be referenced
// from ordinary .cpp translation units without that caller needing to know
// which Thrust backend is active, the same as UniformGrid (uniform_grid.h) and
// Renderer (src/render/renderer.h).
#pragma once

#include <cstdint>
#include <memory>

#include "ray.h"
#include "sah_cost.h"
#include "scene.h"
#include "triangle.h"
#include "uniform_grid.h"
#include "vec.h"

namespace happyray
{

	struct GridBoundsView
	{
		Vec3f bounds_min;
		Vec3f bounds_max;
		Vec3f cell_size;
		Vec3f inv_cell_size;
		Vec3i resolution = Vec3i(1, 1, 1);
	};

	struct GridRangeView : GridBoundsView
	{
		const CellRange *cell_ranges	  = nullptr;
		const uint32_t	*triangle_indices = nullptr;
	};

	// Per-top-level-cell second-level grid metadata. `cell_range_offset` locates
	// this grid's ranges in the shared second-level arrays.
	struct alignas(8) TopLevelCell
	{
		int		 resolution		   = 1;
		uint32_t cell_range_offset = 0;

		HAPPYRAY_HD bool IsEmpty() const
		{
			return resolution < 0;
		}
		HAPPYRAY_HD int Resolution() const
		{
			return IsEmpty() ? -resolution : resolution;
		}
	};
	static_assert(sizeof(TopLevelCell) == 2 * sizeof(uint32_t));
	static_assert(alignof(TopLevelCell) == 8);

	struct TwoLevelGridView
	{
		GridBoundsView		top_level;
		const TopLevelCell *top_level_cells				  = nullptr; // One entry per top-level cell.
		const CellRange	   *second_level_cell_ranges	  = nullptr;
		const uint32_t	   *second_level_triangle_indices = nullptr; // Sorted by second-level cell.
	};

	// Builds the range view for top-level cell `cell_index`'s second-level grid.
	HAPPYRAY_HD inline GridRangeView SecondLevelView(const TwoLevelGridView &grid, uint32_t cell_index)
	{
		const GridBoundsView &top		 = grid.top_level;
		const TopLevelCell	 &info		 = grid.top_level_cells[cell_index];
		const int			  resolution = info.Resolution();
		const int			  res_xy	 = top.resolution.x * top.resolution.y;
		const int			  cz		 = static_cast<int>(cell_index) / res_xy;
		const int			  rem		 = static_cast<int>(cell_index) % res_xy;
		const int			  cy		 = rem / top.resolution.x;
		const int			  cx		 = rem % top.resolution.x;
		const Vec3f			  bounds_min(top.bounds_min.x + cx * top.cell_size.x,
								 top.bounds_min.y + cy * top.cell_size.y,
								 top.bounds_min.z + cz * top.cell_size.z);
		const Vec3f cell_size(top.cell_size.x / resolution, top.cell_size.y / resolution, top.cell_size.z / resolution);
		GridRangeView view;
		view.bounds_min		  = bounds_min;
		view.bounds_max		  = bounds_min + top.cell_size;
		view.cell_size		  = cell_size;
		view.inv_cell_size	  = Vec3f(1.0f / cell_size.x, 1.0f / cell_size.y, 1.0f / cell_size.z);
		view.resolution		  = Vec3i(resolution, resolution, resolution);
		view.cell_ranges	  = grid.second_level_cell_ranges + info.cell_range_offset;
		view.triangle_indices = grid.second_level_triangle_indices;
		return view;
	}

	namespace detail
	{

		// Two-level traversal, ported from happyray's TLGridTraverser: the outer DDA
		// directly initializes an inner DDA for each non-empty top-level cell, with no
		// nested WalkGridCells calls or second-level box test. Top-level cells with no
		// triangles are skipped using their resolution sign bit.
		// `visit_top(top_index)` is called for every top-level cell visited (a no-op
		// for queries; the SAH calibration counts steps with it);
		// `visit_leaf(leaf_index, &ray_t_max)` for every second-level cell, where
		// `leaf_index` indexes the shared second-level range arrays.
		template <typename VisitTopFn, typename VisitLeafFn>
		HAPPYRAY_HD inline bool TraverseTwoLevelGrid(const TwoLevelGridView &grid,
													 const Ray				&ray,
													 VisitTopFn			   &&visit_top,
													 VisitLeafFn		   &&visit_leaf)
		{
			const GridBoundsView &top	  = grid.top_level;
			const Vec3f			  inv_dir = SafeRcp(ray.direction);
			float				  t_near, t_far;
			if (!IntersectRayBoxRcp(ray.origin,
									ray.direction,
									inv_dir,
									top.bounds_min,
									top.bounds_max,
									ray.t_min,
									ray.t_max,
									&t_near,
									&t_far))
			{
				return false;
			}
			float		ray_t_max = ray.t_max;
			const Vec3f top_entry = (ray.origin + ray.direction * t_near - top.bounds_min) * top.inv_cell_size;
			Vec3i		cell(ClampInt(static_cast<int>(top_entry.x), 0, top.resolution.x - 1),
						 ClampInt(static_cast<int>(top_entry.y), 0, top.resolution.y - 1),
						 ClampInt(static_cast<int>(top_entry.z), 0, top.resolution.z - 1));
			const Vec3i step(ray.direction.x > 0.0f ? 1 : 0,
							 ray.direction.y > 0.0f ? 1 : 0,
							 ray.direction.z > 0.0f ? 1 : 0);
			Vec3f next(ray.direction.x != 0.0f
						   ? ((cell.x + step.x) * top.cell_size.x + top.bounds_min.x - ray.origin.x) * inv_dir.x
						   : INFINITY,
					   ray.direction.y != 0.0f
						   ? ((cell.y + step.y) * top.cell_size.y + top.bounds_min.y - ray.origin.y) * inv_dir.y
						   : INFINITY,
					   ray.direction.z != 0.0f
						   ? ((cell.z + step.z) * top.cell_size.z + top.bounds_min.z - ray.origin.z) * inv_dir.z
						   : INFINITY);
			const Vec3f delta(top.cell_size.x * std::fabs(inv_dir.x),
							  top.cell_size.y * std::fabs(inv_dir.y),
							  top.cell_size.z * std::fabs(inv_dir.z));
			const Vec3i increment(ray.direction.x > 0.0f ? 1 : -1,
								  ray.direction.y > 0.0f ? 1 : -1,
								  ray.direction.z > 0.0f ? 1 : -1);
			float t_cell = t_near;

			for (;;)
			{
				const uint32_t top_index =
					static_cast<uint32_t>(cell.x) +
					static_cast<uint32_t>(top.resolution.x) *
						(static_cast<uint32_t>(cell.y) +
						 static_cast<uint32_t>(top.resolution.y) * static_cast<uint32_t>(cell.z));
				visit_top(top_index);

				const TopLevelCell &info = grid.top_level_cells[top_index];
				if (!info.IsEmpty())
				{
					const int	resolution = info.Resolution();
					const Vec3i leaf_resolution(resolution);
					const Vec3f cell_min(top.bounds_min.x + cell.x * top.cell_size.x,
										 top.bounds_min.y + cell.y * top.cell_size.y,
										 top.bounds_min.z + cell.z * top.cell_size.z);
					const Vec3f sub_inv(top.inv_cell_size.x * leaf_resolution.x,
										top.inv_cell_size.y * leaf_resolution.y,
										top.inv_cell_size.z * leaf_resolution.z);
					const Vec3f sub_cell_size(top.cell_size.x / leaf_resolution.x,
											  top.cell_size.y / leaf_resolution.y,
											  top.cell_size.z / leaf_resolution.z);
					const Vec3f leaf_entry = (ray.origin + ray.direction * t_cell - cell_min) * sub_inv;
					Vec3i		leaf_cell(ClampInt(static_cast<int>(leaf_entry.x), 0, leaf_resolution.x - 1),
									  ClampInt(static_cast<int>(leaf_entry.y), 0, leaf_resolution.y - 1),
									  ClampInt(static_cast<int>(leaf_entry.z), 0, leaf_resolution.z - 1));
					const Vec3i leaf_step(ray.direction.x > 0.0f ? 1 : 0,
										  ray.direction.y > 0.0f ? 1 : 0,
										  ray.direction.z > 0.0f ? 1 : 0);
					Vec3f leaf_next(
						ray.direction.x != 0.0f
							? ((leaf_cell.x + leaf_step.x) * sub_cell_size.x + cell_min.x - ray.origin.x) * inv_dir.x
							: INFINITY,
						ray.direction.y != 0.0f
							? ((leaf_cell.y + leaf_step.y) * sub_cell_size.y + cell_min.y - ray.origin.y) * inv_dir.y
							: INFINITY,
						ray.direction.z != 0.0f
							? ((leaf_cell.z + leaf_step.z) * sub_cell_size.z + cell_min.z - ray.origin.z) * inv_dir.z
							: INFINITY);
					const Vec3f leaf_delta(sub_cell_size.x * std::fabs(inv_dir.x),
										   sub_cell_size.y * std::fabs(inv_dir.y),
										   sub_cell_size.z * std::fabs(inv_dir.z));
					const Vec3i leaf_increment(ray.direction.x > 0.0f ? 1 : -1,
											   ray.direction.y > 0.0f ? 1 : -1,
											   ray.direction.z > 0.0f ? 1 : -1);

					for (;;)
					{
						const uint32_t leaf = FlattenCellIndex(leaf_cell.x,
															   leaf_cell.y,
															   leaf_cell.z,
															   leaf_resolution.x,
															   leaf_resolution.y);
						if (visit_leaf(info.cell_range_offset + leaf, &ray_t_max))
						{
							return true;
						}

						if (leaf_next.x < leaf_next.y && leaf_next.x < leaf_next.z)
						{
							if (leaf_next.x > ray_t_max)
								break;
							leaf_cell.x += leaf_increment.x;
							if (static_cast<unsigned>(leaf_cell.x) >= static_cast<unsigned>(leaf_resolution.x))
								break;
							leaf_next.x += leaf_delta.x;
						}
						else if (leaf_next.y < leaf_next.z)
						{
							if (leaf_next.y > ray_t_max)
								break;
							leaf_cell.y += leaf_increment.y;
							if (static_cast<unsigned>(leaf_cell.y) >= static_cast<unsigned>(leaf_resolution.y))
								break;
							leaf_next.y += leaf_delta.y;
						}
						else
						{
							if (leaf_next.z > ray_t_max)
								break;
							leaf_cell.z += leaf_increment.z;
							if (static_cast<unsigned>(leaf_cell.z) >= static_cast<unsigned>(leaf_resolution.z))
								break;
							leaf_next.z += leaf_delta.z;
						}
					}
				}

				if (next.x < next.y && next.x < next.z)
				{
					if (next.x > ray_t_max)
						return false;
					cell.x += increment.x;
					if (static_cast<unsigned>(cell.x) >= static_cast<unsigned>(top.resolution.x))
						return false;
					t_cell = next.x;
					next.x += delta.x;
				}
				else if (next.y < next.z)
				{
					if (next.y > ray_t_max)
						return false;
					cell.y += increment.y;
					if (static_cast<unsigned>(cell.y) >= static_cast<unsigned>(top.resolution.y))
						return false;
					t_cell = next.y;
					next.y += delta.y;
				}
				else
				{
					if (next.z > ray_t_max)
						return false;
					cell.z += increment.z;
					if (static_cast<unsigned>(cell.z) >= static_cast<unsigned>(top.resolution.z))
						return false;
					t_cell = next.z;
					next.z += delta.z;
				}
			}
		}

	} // namespace detail

	// Finds the closest hit by walking the top-level grid and, for every non-empty
	// cell the ray passes through, its second-level grid (see
	// detail::TraverseTwoLevelGrid above). Returns false if the ray hits nothing.
	HAPPYRAY_HD inline bool
	IntersectSceneTwoLevelGrid(const SceneView &scene, const TwoLevelGridView &grid, const Ray &ray, HitRecord *out_hit)
	{
		ClosestTriangleHit hit;
		detail::TraverseTwoLevelGrid(
			grid,
			ray,
			[](uint32_t) {},
			[&](uint32_t leaf, float *ray_t_max) -> bool
			{
				IntersectTrianglesClosest(scene,
										  grid.second_level_triangle_indices,
										  grid.second_level_cell_ranges[leaf].start,
										  grid.second_level_cell_ranges[leaf].end,
										  ray,
										  ray_t_max,
										  &hit);
				return false;
			});
		return ResolveClosestHit(scene, ray, hit, out_hit);
	}

	// Returns true as soon as any triangle in any second-level cell the ray visits
	// occludes `ray`.
	HAPPYRAY_HD inline bool IsOccludedTwoLevelGrid(const SceneView &scene, const TwoLevelGridView &grid, const Ray &ray)
	{
		return detail::TraverseTwoLevelGrid(
			grid,
			ray,
			[](uint32_t) {},
			[&](uint32_t leaf, float *) -> bool
			{
				return TestTrianglesInRangeOccluded(scene,
													grid.second_level_triangle_indices,
													grid.second_level_cell_ranges[leaf].start,
													grid.second_level_cell_ranges[leaf].end,
													ray);
			});
	}

	class TwoLevelGrid
	{
	  public:
		static constexpr float kDefaultTopLevelDensity	  = 0.12f;
		static constexpr float kDefaultSecondLevelDensity = 1.2f;
		// `scene` must point at memory reachable by the selected Thrust backend.
		// Cells at or below `dense_cell_threshold` use a 1x1x1 second-level grid.
		explicit TwoLevelGrid(const SceneView &scene,
							  float			   top_level_density	= kDefaultTopLevelDensity,
							  float			   second_level_density = kDefaultSecondLevelDensity,
							  uint32_t		   dense_cell_threshold = 16);
		~TwoLevelGrid();
		TwoLevelGrid(const TwoLevelGrid &)			  = delete;
		TwoLevelGrid &operator=(const TwoLevelGrid &) = delete;

		TwoLevelGridView View() const;

		// Per-array element counts of every device buffer this grid actually holds,
		// used by
		// tests/accel_performance_tests.cpp to compute an analytic memory-footprint
		// estimate (count * sizeof(element)) without needing any CUDA-specific
		// allocation introspection. `top_level`'s own cell count is
		// `View().top_level.resolution`'s product, already public; these fill
		// in the remaining counts that aren't otherwise derivable from the view.
		uint32_t SecondLevelCellCount() const;
		uint32_t SecondLevelTriangleIndexCount() const;

		// Structural SAH cost estimate for this grid as currently built (see
		// sah_cost.h), normalized by `scene_bounds`'s own half surface area -- not
		// from casting rays, but the same surface-area-weighted "expected cost"
		// heuristic the grid density search (see src/app/grid_density_search.h) uses
		// to prefer one candidate density over another. Each nested grid's reference
		// count is aggregated while the sorted ranges are built.
		float ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const;

	  private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

} // namespace happyray
