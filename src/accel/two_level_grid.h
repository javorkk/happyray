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
		const uint32_t	*prim_indices = nullptr;
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
		const uint32_t	   *second_level_prim_indices = nullptr; // Sorted by second-level cell.
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
		view.prim_indices = grid.second_level_prim_indices;
		return view;
	}

	namespace detail
	{

		// Two-level traversal, ported from happyray's TLGridTraverser: the outer DDA
		// directly initializes an inner DDA for each non-empty top-level cell, with no
		// recursive traversal calls or second-level box test. Top-level cells with no
		// primitives are skipped using their resolution sign bit.
		// `visit_top(top_index)` is called for every top-level cell visited (a no-op
		// for queries; the SAH calibration counts steps with it);
		// `visit_leaf(range, &ray_t_max)` for every second-level cell.
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
			const Vec3f &origin			   = ray.origin;
			const Vec3f &dir			   = ray.direction;
			const Vec3f	 entry			   = (origin + dir * t_near - top.bounds_min) * top.inv_cell_size;
			const int	 top_resolution[3] = {top.resolution.x, top.resolution.y, top.resolution.z};
			int			 cell[3]		   = {ClampInt(static_cast<int>(entry.x), 0, top_resolution[0] - 1),
											  ClampInt(static_cast<int>(entry.y), 0, top_resolution[1] - 1),
											  ClampInt(static_cast<int>(entry.z), 0, top_resolution[2] - 1)};
			const int	 increment[3]	   = {dir.x > 0.0f ? 1 : -1, dir.y > 0.0f ? 1 : -1, dir.z > 0.0f ? 1 : -1};
			float		 next[3]		   = {
				 dir.x != 0.0f ? ((cell[0] + (dir.x > 0.0f)) * top.cell_size.x + top.bounds_min.x - origin.x) * inv_dir.x
											   : INFINITY,
				dir.y != 0.0f ? ((cell[1] + (dir.y > 0.0f)) * top.cell_size.y + top.bounds_min.y - origin.y) * inv_dir.y
							  : INFINITY,
				dir.z != 0.0f ? ((cell[2] + (dir.z > 0.0f)) * top.cell_size.z + top.bounds_min.z - origin.z) * inv_dir.z
							  : INFINITY};
			const float delta[3]  = {top.cell_size.x * std::fabs(inv_dir.x),
									 top.cell_size.y * std::fabs(inv_dir.y),
									 top.cell_size.z * std::fabs(inv_dir.z)};
			float		ray_t_max = ray.t_max;
			float		t_cell	  = t_near;
			for (;;)
			{
				const uint32_t top_index =
					FlattenCellIndex(cell[0], cell[1], cell[2], top_resolution[0], top_resolution[1]);
				visit_top(top_index);

				const TopLevelCell &info = grid.top_level_cells[top_index];
				if (!info.IsEmpty())
				{
					const int	resolution = info.Resolution();
					const Vec3f cell_min(top.bounds_min.x + cell[0] * top.cell_size.x,
										 top.bounds_min.y + cell[1] * top.cell_size.y,
										 top.bounds_min.z + cell[2] * top.cell_size.z);
					const Vec3f sub_inv		  = top.inv_cell_size * resolution;
					const Vec3f sub_cell_size = top.cell_size / resolution;
					const Vec3f leaf_entry	  = (origin + dir * t_cell - cell_min) * sub_inv;
					int			leaf_cell[3]  = {ClampInt(static_cast<int>(leaf_entry.x), 0, resolution - 1),
												 ClampInt(static_cast<int>(leaf_entry.y), 0, resolution - 1),
												 ClampInt(static_cast<int>(leaf_entry.z), 0, resolution - 1)};
					float		leaf_next[3]  = {
						   dir.x != 0.0f
								   ? ((leaf_cell[0] + (dir.x > 0.0f)) * sub_cell_size.x + cell_min.x - origin.x) * inv_dir.x
								   : INFINITY,
						dir.y != 0.0f
							? ((leaf_cell[1] + (dir.y > 0.0f)) * sub_cell_size.y + cell_min.y - origin.y) * inv_dir.y
							: INFINITY,
						dir.z != 0.0f
							? ((leaf_cell[2] + (dir.z > 0.0f)) * sub_cell_size.z + cell_min.z - origin.z) * inv_dir.z
							: INFINITY};
					const float leaf_delta[3] = {sub_cell_size.x * std::fabs(inv_dir.x),
												 sub_cell_size.y * std::fabs(inv_dir.y),
												 sub_cell_size.z * std::fabs(inv_dir.z)};
					for (;;)
					{
						const uint32_t leaf =
							info.cell_range_offset +
							FlattenCellIndex(leaf_cell[0], leaf_cell[1], leaf_cell[2], resolution, resolution);
						if (visit_leaf(grid.second_level_cell_ranges[leaf], &ray_t_max))
						{
							return true;
						}

						const int axis	  = leaf_next[0] < leaf_next[1] ? (leaf_next[0] < leaf_next[2] ? 0 : 2)
																		: (leaf_next[1] < leaf_next[2] ? 1 : 2);
						bool	  advance = false;
						switch (axis)
						{
						case 0:
							if (leaf_next[0] > ray_t_max)
								break;
							leaf_cell[0] += increment[0];
							if (static_cast<unsigned>(leaf_cell[0]) >= static_cast<unsigned>(resolution))
								break;
							leaf_next[0] += leaf_delta[0];
							advance = true;
							break;
						case 1:
							if (leaf_next[1] > ray_t_max)
								break;
							leaf_cell[1] += increment[1];
							if (static_cast<unsigned>(leaf_cell[1]) >= static_cast<unsigned>(resolution))
								break;
							leaf_next[1] += leaf_delta[1];
							advance = true;
							break;
						default:
							if (leaf_next[2] > ray_t_max)
								break;
							leaf_cell[2] += increment[2];
							if (static_cast<unsigned>(leaf_cell[2]) >= static_cast<unsigned>(resolution))
								break;
							leaf_next[2] += leaf_delta[2];
							advance = true;
							break;
						}
						if (!advance)
							break;
					}
				}

				const int axis = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2);
				switch (axis)
				{
				case 0:
					if (next[0] > ray_t_max)
						return false;
					t_cell = next[0];
					cell[0] += increment[0];
					if (static_cast<unsigned>(cell[0]) >= static_cast<unsigned>(top_resolution[0]))
						return false;
					next[0] += delta[0];
					break;
				case 1:
					if (next[1] > ray_t_max)
						return false;
					t_cell = next[1];
					cell[1] += increment[1];
					if (static_cast<unsigned>(cell[1]) >= static_cast<unsigned>(top_resolution[1]))
						return false;
					next[1] += delta[1];
					break;
				default:
					if (next[2] > ray_t_max)
						return false;
					t_cell = next[2];
					cell[2] += increment[2];
					if (static_cast<unsigned>(cell[2]) >= static_cast<unsigned>(top_resolution[2]))
						return false;
					next[2] += delta[2];
					break;
				}
			}
		}

	} // namespace detail

	// Finds the closest hit by walking the top-level grid and, for every non-empty
	// cell the ray passes through, its second-level grid (see
	// detail::TraverseTwoLevelGrid above). Returns false if the ray hits nothing.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool
	IntersectSceneTwoLevelGrid(const SceneView &scene, const TwoLevelGridView &grid, const Ray &ray, HitRecord *out_hit)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IntersectSceneTwoLevelGrid<Q>(scene, grid, ray, out_hit);
									 });
		}
		else
		{
			ClosestHit hit;
			detail::TraverseTwoLevelGrid(
				grid,
				ray,
				[](uint32_t) {},
				detail::ClosestHitVisitor<P>{scene, grid.second_level_prim_indices, ray, hit});
			return ResolveClosestHit<P>(scene, ray, hit, out_hit);
		}
	}

	// Returns true as soon as any primitive in any second-level cell the ray visits
	// occludes `ray`.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool IsOccludedTwoLevelGrid(const SceneView &scene, const TwoLevelGridView &grid, const Ray &ray)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IsOccludedTwoLevelGrid<Q>(scene, grid, ray);
									 });
		}
		else
		{
			return detail::TraverseTwoLevelGrid(
				grid,
				ray,
				[](uint32_t) {},
				detail::OcclusionVisitor<P>{scene, grid.second_level_prim_indices, ray});
		}
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
							  uint32_t		   dense_cell_threshold = 16,
							  CellOverlapMode  mode					= DefaultCellOverlapMode());
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
		uint32_t SecondLevelPrimIndexCount() const;

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
