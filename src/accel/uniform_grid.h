// The uniform grid: a flat, non-owning view with one explicit start/end range
// for traversal (UniformGridView, FlattenCellIndex,
// IntersectSceneGrid/IsOccludedGrid -- all HAPPYRAY_HD, so this exact traversal
// code runs on whichever hardware the active Renderer, src/render/renderer.h,
// does), plus the owning builder (UniformGrid) that constructs one and exposes
// it as a UniformGridView.
//
// UniformGrid is exposed through a plain-C++ interface so it can be referenced
// from ordinary .cpp translation units without that caller needing to know
// which Thrust backend is active. There is exactly one implementation
// (uniform_grid.cpp), parallelized via Thrust functors/algorithms
// (thrust::transform/for_each/exclusive_scan/sort_by_key/inclusive_scan) -- the
// sort-based (happyray/HAGrid-style) construction algorithm doesn't change
// between backends, only which hardware it runs on, which is a build-time
// choice (see CMakeLists.txt's HAPPYRAY_THRUST_BACKEND option), same as Renderer.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

#include "bbox.h"
#include "primitive.h"
#include "ray.h"
#include "sah_cost.h"
#include "scene.h"
#include "triangle.h"
#include "vec.h"

namespace happyray
{

	// Shared host/device build math for all three grid builders.
	HAPPYRAY_HD inline BBox ComputeTriangleBBox(const SceneView &scene, uint32_t tri)
	{
		return BBoxExtractor<Triangle>::Get(FetchTriangle(scene, tri));
	}

	struct alignas(8) CellRange
	{
		uint32_t start = 0;
		uint32_t end   = 0;
	};
	static_assert(sizeof(CellRange) == 2 * sizeof(uint32_t));

	// Sizes a grid from the expanded geometry bounds and target triangle density.
	HAPPYRAY_HD inline Vec3i
	ComputeGridResolution(const BBox &bounds, uint32_t triangle_count, float target_triangles_per_cell)
	{
		if (triangle_count == 0)
		{
			return Vec3i(1, 1, 1);
		}

		const Vec3f raw_extent = bounds.Extent();
		// Floor degenerate axes so the volume and resolution formula remain
		// well-defined.
		const float diagonal   = Length(raw_extent);
		const float min_extent = diagonal > 0.0f ? diagonal * 1e-3f : 1.0f;
		const Vec3f extent(raw_extent.x > min_extent ? raw_extent.x : min_extent,
						   raw_extent.y > min_extent ? raw_extent.y : min_extent,
						   raw_extent.z > min_extent ? raw_extent.z : min_extent);

		const float volume = extent.x * extent.y * extent.z;
		const float cells_per_unit_volume =
			std::cbrt(target_triangles_per_cell * static_cast<float>(triangle_count) / volume);

		return Vec3i(
			extent.x * cells_per_unit_volume + 0.5f > 1.0f ? static_cast<int>(extent.x * cells_per_unit_volume + 0.5f)
														   : 1,
			extent.y * cells_per_unit_volume + 0.5f > 1.0f ? static_cast<int>(extent.y * cells_per_unit_volume + 0.5f)
														   : 1,
			extent.z * cells_per_unit_volume + 0.5f > 1.0f ? static_cast<int>(extent.z * cells_per_unit_volume + 0.5f)
														   : 1);
	}

	// Conservative BBox binning: inclusive cell coordinates, clamped to the grid.
	HAPPYRAY_HD inline void ComputeCellRange(const BBox	 &box,
											 const Vec3f &bounds_min,
											 const Vec3f &inv_cell_size,
											 const Vec3i &resolution,
											 Vec2i		 *x_range,
											 Vec2i		 *y_range,
											 Vec2i		 *z_range)
	{
		x_range->x =
			ClampInt(static_cast<int>((box.min_bound.x - bounds_min.x) * inv_cell_size.x), 0, resolution.x - 1);
		x_range->y =
			ClampInt(static_cast<int>((box.max_bound.x - bounds_min.x) * inv_cell_size.x), 0, resolution.x - 1);
		y_range->x =
			ClampInt(static_cast<int>((box.min_bound.y - bounds_min.y) * inv_cell_size.y), 0, resolution.y - 1);
		y_range->y =
			ClampInt(static_cast<int>((box.max_bound.y - bounds_min.y) * inv_cell_size.y), 0, resolution.y - 1);
		z_range->x =
			ClampInt(static_cast<int>((box.min_bound.z - bounds_min.z) * inv_cell_size.z), 0, resolution.z - 1);
		z_range->y =
			ClampInt(static_cast<int>((box.max_bound.z - bounds_min.z) * inv_cell_size.z), 0, resolution.z - 1);
	}

	struct UniformGridView
	{
		Vec3f bounds_min;
		Vec3f bounds_max;
		Vec3f cell_size;
		Vec3f inv_cell_size;
		Vec3i resolution = Vec3i(1, 1, 1);
		// One begin/end pair per cell; empty cells have equal indices.
		const CellRange *cell_ranges	  = nullptr;
		const uint32_t	*triangle_indices = nullptr; // Sorted by cell; indexes into SceneView.
	};

	HAPPYRAY_HD inline uint32_t FlattenCellIndex(int cx, int cy, int cz, int resolution_x, int resolution_y)
	{
		return static_cast<uint32_t>(cz) * resolution_y * resolution_x + static_cast<uint32_t>(cy) * resolution_x +
			   static_cast<uint32_t>(cx);
	}

	HAPPYRAY_HD inline uint32_t FlattenCellIndex(const Vec3i &cell, const Vec3i &resolution)
	{
		return FlattenCellIndex(cell.x, cell.y, cell.z, resolution.x, resolution.y);
	}

	namespace detail
	{

		// Number of grid cells overlapped by the triangle AABB's clamped cell range.
		HAPPYRAY_HD inline uint32_t CountCandidateCells(const BBox	&box,
														const Vec3f &bounds_min,
														const Vec3f &inv_cell_size,
														const Vec3i &resolution)
		{
			Vec2i x_range, y_range, z_range;
			ComputeCellRange(box, bounds_min, inv_cell_size, resolution, &x_range, &y_range, &z_range);
			return static_cast<uint32_t>(x_range.y - x_range.x + 1) * static_cast<uint32_t>(y_range.y - y_range.x + 1) *
				   static_cast<uint32_t>(z_range.y - z_range.x + 1);
		}

		// Preserve the grid's 1% cell enlargement while delegating the SAT to core geometry.
		struct TriangleCellIntersection
		{
			TriangleBoxIntersection intersects;

			HAPPYRAY_HD TriangleCellIntersection(const SceneView &scene, uint32_t tri)
				: TriangleCellIntersection(FetchTriangle(scene, tri))
			{
			}

			HAPPYRAY_HD explicit TriangleCellIntersection(const Triangle &triangle) : intersects(triangle)
			{
			}

			HAPPYRAY_HD bool operator()(const Vec3f &center, const Vec3f &cell_size) const
			{
				const Vec3f half = cell_size * 0.505f;
				return intersects(BBox(center - half, center + half));
			}
		};

		struct CellExtractor
		{
			const uint32_t *keys;
			uint32_t	   *cell_start;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				if (i == 0 || keys[i - 1] != keys[i])
					cell_start[keys[i]] = i;
			}
		};

		struct CellRangeExtractor
		{
			const uint32_t *keys;
			CellRange	   *ranges;
			uint32_t		pair_count;
			uint32_t		cell_count;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				const uint32_t cell = keys[i];
				if (cell >= cell_count)
					return;
				if (i == 0 || keys[i - 1] != cell)
					ranges[cell].start = i;
				if (i + 1 == pair_count || keys[i + 1] != cell)
					ranges[cell].end = i + 1;
			}
		};

		// Shared by every FragmentCounter/FragmentWriter pair
		// (uniform/two-level/irregular): walks the triangle's AABB-derived cell range
		// [cx0,cx1]x[cy0,cy1]x[cz0,cz1] (from ComputeCellRange) and invokes `emit(cell)`
		// with a Vec3i only for cells the triangle's *actual* shape intersects (the full
		// SAT test, TriangleCellIntersection above), not just its bounding box. This
		// conservative AABB range can vastly overcount true intersections for long,
		// thin, diagonally-oriented triangles (e.g. hair-strand geometry) -- a single
		// such triangle's bbox can span a cube-shaped range spanning tens of thousands
		// of cells even though the triangle only actually threads through a sliver of
		// them. Builders may reserve the full AABB range and pad unused output slots;
		// this helper emits only the exact SAT intersections.
		template <typename EmitFn>
		HAPPYRAY_HD inline void ForEachIntersectedCell(const Triangle &triangle,
													   const BBox	  &box,
													   const Vec3f	  &bounds_min,
													   const Vec3f	  &inv_cell_size,
													   const Vec3f	  &cell_size,
													   const Vec3i	  &resolution,
													   EmitFn		 &&emit)
		{
			Vec2i x_range, y_range, z_range;
			ComputeCellRange(box, bounds_min, inv_cell_size, resolution, &x_range, &y_range, &z_range);
			const TriangleCellIntersection intersects(triangle);
			for (int cz = z_range.x; cz <= z_range.y; ++cz)
			{
				for (int cy = y_range.x; cy <= y_range.y; ++cy)
				{
					for (int cx = x_range.x; cx <= x_range.y; ++cx)
					{
						const Vec3f center = bounds_min + Vec3f(cx + 0.5f, cy + 0.5f, cz + 0.5f) * cell_size;
						if (intersects(center, cell_size))
						{
							emit(Vec3i(cx, cy, cz));
						}
					}
				}
			}
		}

		HAPPYRAY_HD inline uint32_t CountExactIntersectedCells(const Triangle &triangle,
															   const BBox	  &box,
															   const Vec3f	  &bounds_min,
															   const Vec3f	  &inv_cell_size,
															   const Vec3f	  &cell_size,
															   const Vec3i	  &resolution)
		{
			uint32_t count = 0;
			ForEachIntersectedCell(triangle,
								   box,
								   bounds_min,
								   inv_cell_size,
								   cell_size,
								   resolution,
								   [&](const Vec3i &) { ++count; });
			return count;
		}

		HAPPYRAY_HD inline uint32_t CountTriangleCellIntersections(const Triangle &triangle,
																   const BBox	  &box,
																   const Vec3f	  &bounds_min,
																   const Vec3f	  &inv_cell_size,
																   const Vec3f	  &cell_size,
																   const Vec3i	  &resolution)
		{
#if HAPPYRAY_FAST_CELL_COUNT
			return CountCandidateCells(box, bounds_min, inv_cell_size, resolution);
#else
			return CountExactIntersectedCells(triangle, box, bounds_min, inv_cell_size, cell_size, resolution);
#endif
		}

		// Reciprocal of a ray direction component, +/-infinity for zero (HAGrid's
		// safe_rcp) -- lets every traversal routine below replace per-step divisions
		// with multiplications.
		HAPPYRAY_HD inline float SafeRcp(float x)
		{
			return x != 0.0f ? 1.0f / x : std::copysign(INFINITY, x);
		}

		HAPPYRAY_HD inline Vec3f SafeRcp(const Vec3f &d)
		{
			return Vec3f(SafeRcp(d.x), SafeRcp(d.y), SafeRcp(d.z));
		}

		// Slab test against [box_min, box_max] using a precomputed reciprocal
		// direction, clipped to [t_min, t_max]. Zero direction components are handled
		// explicitly (no 0 * inf NaNs).
		HAPPYRAY_HD inline bool IntersectRayBoxRcp(const Vec3f &origin,
												   const Vec3f &dir,
												   const Vec3f &inv_dir,
												   const Vec3f &box_min,
												   const Vec3f &box_max,
												   float		t_min,
												   float		t_max,
												   float	   *out_t_near,
												   float	   *out_t_far)
		{
			float t_near = t_min, t_far = t_max;
#define HAPPYRAY_SLAB(a)                                                                                               \
	if (dir.a != 0.0f)                                                                                                 \
	{                                                                                                                  \
		const float t0 = (box_min.a - origin.a) * inv_dir.a;                                                           \
		const float t1 = (box_max.a - origin.a) * inv_dir.a;                                                           \
		t_near		   = std::fmax(t_near, std::fmin(t0, t1));                                                         \
		t_far		   = std::fmin(t_far, std::fmax(t0, t1));                                                          \
	}                                                                                                                  \
	else if (origin.a < box_min.a || origin.a > box_max.a)                                                             \
	{                                                                                                                  \
		return false;                                                                                                  \
	}
			HAPPYRAY_SLAB(x)
			HAPPYRAY_SLAB(y)
			HAPPYRAY_SLAB(z)
#undef HAPPYRAY_SLAB
			*out_t_near = t_near;
			*out_t_far	= t_far;
			return t_near <= t_far;
		}

		// Amanatides-Woo 3D-DDA over a res_x*res_y*res_z grid with origin `grid_min`,
		// starting at ray parameter `t_entry` (which must lie on or inside the grid's
		// box), ported from happyray's UGridTraverser/TLGridTraverser: reciprocal
		// direction instead of divisions, tDelta = cellSize * |1/dir|, and scalar
		// per-axis state (no dynamically indexed arrays, which the CUDA backend would
		// otherwise spill to local memory). Invokes `visit(cell_index, cx, cy, cz,
		// t_cell_entry, &ray_t_max)` for each cell in near-to-far order; `visit` may
		// shrink `ray_t_max` (closest hit), and the walk stops once the next cell
		// boundary lies beyond it -- i.e. as soon as a hit inside the current cell is
		// found -- or when `visit` returns true (occlusion), or the ray leaves the
		// grid. Returns true only if `visit` did.
		template <typename VisitFn>
		HAPPYRAY_HD inline bool WalkGridCells(const Vec3f &origin,
											  const Vec3f &dir,
											  const Vec3f &inv_dir,
											  const Vec3f &grid_min,
											  const Vec3f &cell_size,
											  const Vec3f &inv_cell_size,
											  const Vec3i &resolution,
											  float		   t_entry,
											  float		  *ray_t_max,
											  VisitFn	 &&visit)
		{
			const Vec3f entry = (origin + dir * t_entry - grid_min) * inv_cell_size;
			Vec3i		cell(ClampInt(static_cast<int>(entry.x), 0, resolution.x - 1),
						 ClampInt(static_cast<int>(entry.y), 0, resolution.y - 1),
						 ClampInt(static_cast<int>(entry.z), 0, resolution.z - 1));

			// Next boundary crossing per axis; zero-direction axes never step.
			const Vec3i step(dir.x > 0.0f ? 1 : 0, dir.y > 0.0f ? 1 : 0, dir.z > 0.0f ? 1 : 0);
			Vec3f next(dir.x != 0.0f ? ((cell.x + step.x) * cell_size.x + grid_min.x - origin.x) * inv_dir.x : INFINITY,
					   dir.y != 0.0f ? ((cell.y + step.y) * cell_size.y + grid_min.y - origin.y) * inv_dir.y : INFINITY,
					   dir.z != 0.0f ? ((cell.z + step.z) * cell_size.z + grid_min.z - origin.z) * inv_dir.z
									 : INFINITY);
			const Vec3f delta(cell_size.x * std::fabs(inv_dir.x),
							  cell_size.y * std::fabs(inv_dir.y),
							  cell_size.z * std::fabs(inv_dir.z));
			const Vec3i increment(dir.x > 0.0f ? 1 : -1, dir.y > 0.0f ? 1 : -1, dir.z > 0.0f ? 1 : -1);

			float t_cell = t_entry;
			for (;;)
			{
				const uint32_t cell_index = static_cast<uint32_t>(cell.x) +
											static_cast<uint32_t>(resolution.x) *
												(static_cast<uint32_t>(cell.y) +
												 static_cast<uint32_t>(resolution.y) * static_cast<uint32_t>(cell.z));
				if (visit(cell_index, cell.x, cell.y, cell.z, t_cell, ray_t_max))
				{
					return true;
				}
				if (next.x < next.y && next.x < next.z)
				{
					if (next.x > *ray_t_max)
						return false;
					cell.x += increment.x;
					if (static_cast<unsigned>(cell.x) >= static_cast<unsigned>(resolution.x))
						return false;
					t_cell = next.x;
					next.x += delta.x;
				}
				else if (next.y < next.z)
				{
					if (next.y > *ray_t_max)
						return false;
					cell.y += increment.y;
					if (static_cast<unsigned>(cell.y) >= static_cast<unsigned>(resolution.y))
						return false;
					t_cell = next.y;
					next.y += delta.y;
				}
				else
				{
					if (next.z > *ray_t_max)
						return false;
					cell.z += increment.z;
					if (static_cast<unsigned>(cell.z) >= static_cast<unsigned>(resolution.z))
						return false;
					t_cell = next.z;
					next.z += delta.z;
				}
			}
		}

		// Uniform-grid traversal (happyray's UGridTraverser): clips `ray` to the grid's
		// box, then walks it with WalkGridCells above, invoking `visit_cell(cell_index,
		// &ray_t_max)` per cell.
		template <typename VisitCellFn>
		HAPPYRAY_HD inline bool TraverseGrid(const UniformGridView &grid, const Ray &ray, VisitCellFn &&visit_cell)
		{
			const Vec3f inv_dir = SafeRcp(ray.direction);
			float		t_near, t_far;
			if (!IntersectRayBoxRcp(ray.origin,
									ray.direction,
									inv_dir,
									grid.bounds_min,
									grid.bounds_max,
									ray.t_min,
									ray.t_max,
									&t_near,
									&t_far))
			{
				return false;
			}
			float ray_t_max = ray.t_max;
			return WalkGridCells(ray.origin,
								 ray.direction,
								 inv_dir,
								 grid.bounds_min,
								 grid.cell_size,
								 grid.inv_cell_size,
								 grid.resolution,
								 t_near,
								 &ray_t_max,
								 [&](uint32_t cell_index, int, int, int, float, float *t_max)
								 { return visit_cell(cell_index, t_max); });
		}

	} // namespace detail

	// Closest-hit state carried through traversal: like happyray's `bestHit` and
	// HAGrid's `Hit`, only the triangle id and its barycentrics are tracked while
	// walking cells -- normals and materials are fetched once, by
	// ResolveClosestHit, after traversal ends.
	struct ClosestTriangleHit
	{
		uint32_t triangle = 0xffffffffu;
		float	 t = 0.0f, u = 0.0f, v = 0.0f;
	};

	// Tests every triangle in the contiguous range [begin, end) of `triangle_indices`
	// against `ray`, clipped to `*ray_t_max`, which shrinks (and `*hit` updates) on
	// every closer hit. Returns true if any triangle in the range was hit. Shared
	// by all three grids' closest-hit queries.
	HAPPYRAY_HD inline bool IntersectTrianglesClosest(const SceneView	 &scene,
													  const uint32_t	 *triangle_indices,
													  uint32_t			  begin,
													  uint32_t			  end,
													  const Ray			 &ray,
													  float				 *ray_t_max,
													  ClosestTriangleHit *hit)
	{
		bool hit_anything = false;
		Ray	 probe		  = ray;
		probe.t_max		  = *ray_t_max;

		for (uint32_t i = begin; i < end; ++i)
		{
			const uint32_t tri		= triangle_indices[i];
			const Triangle triangle = FetchTriangle(scene, tri);
			float		   t, u, v;
			if (IntersectRayTriangle(probe, triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], &t, &u, &v))
			{
				probe.t_max	  = t;
				hit->triangle = tri;
				hit->t		  = t;
				hit->u		  = u;
				hit->v		  = v;
				hit_anything  = true;
			}
		}

		*ray_t_max = probe.t_max;

		return hit_anything;
	}

	// Fills `out_hit` (position, barycentric-interpolated shading normal, material)
	// from the closest hit found during traversal. Returns false if nothing was
	// hit.
	HAPPYRAY_HD inline bool
	ResolveClosestHit(const SceneView &scene, const Ray &ray, const ClosestTriangleHit &hit, HitRecord *out_hit)
	{
		if (hit.triangle == 0xffffffffu)
		{
			return false;
		}
		const uint32_t i0	 = scene.indices[hit.triangle * 3 + 0];
		const uint32_t i1	 = scene.indices[hit.triangle * 3 + 1];
		const uint32_t i2	 = scene.indices[hit.triangle * 3 + 2];
		out_hit->t			 = hit.t;
		out_hit->position	 = ray.At(hit.t);
		out_hit->normal		 = Normalize(scene.normals[i0] * (1.0f - hit.u - hit.v) + scene.normals[i1] * hit.u +
									 scene.normals[i2] * hit.v);
		out_hit->material_id = FetchMaterialId(scene, hit.triangle);
		return true;
	}

	// Returns true as soon as any triangle in the contiguous range [begin, end) occludes
	// `ray`.
	HAPPYRAY_HD inline bool TestTrianglesInRangeOccluded(const SceneView &scene,
														 const uint32_t	 *triangle_indices,
														 uint32_t		  begin,
														 uint32_t		  end,
														 const Ray		 &ray)
	{
		for (uint32_t i = begin; i < end; ++i)
		{
			const uint32_t tri		= triangle_indices[i];
			const Triangle triangle = FetchTriangle(scene, tri);
			float		   t, u, v;
			if (IntersectRayTriangle(ray, triangle.vtx[0], triangle.vtx[1], triangle.vtx[2], &t, &u, &v))
			{
				return true;
			}
		}
		return false;
	}

	// Finds the closest hit across the triangles referenced by `grid`'s cells, in
	// the same sense as IntersectSceneBruteForce. Returns false if the ray hits
	// nothing.
	HAPPYRAY_HD inline bool
	IntersectSceneGrid(const SceneView &scene, const UniformGridView &grid, const Ray &ray, HitRecord *out_hit)
	{
		ClosestTriangleHit hit;
		detail::TraverseGrid(grid,
							 ray,
							 [&](uint32_t cell_index, float *ray_t_max) -> bool
							 {
								 IntersectTrianglesClosest(scene,
														   grid.triangle_indices,
														   grid.cell_ranges[cell_index].start,
														   grid.cell_ranges[cell_index].end,
														   ray,
														   ray_t_max,
														   &hit);
								 return false; // The walk itself stops once a hit lies before the next
											   // cell boundary.
							 });
		return ResolveClosestHit(scene, ray, hit, out_hit);
	}

	// Returns true as soon as any triangle referenced by `grid`'s cells occludes
	// `ray`.
	HAPPYRAY_HD inline bool IsOccludedGrid(const SceneView &scene, const UniformGridView &grid, const Ray &ray)
	{
		return detail::TraverseGrid(grid,
									ray,
									[&](uint32_t cell_index, float *) -> bool
									{
										return TestTrianglesInRangeOccluded(scene,
																			grid.triangle_indices,
																			grid.cell_ranges[cell_index].start,
																			grid.cell_ranges[cell_index].end,
																			ray);
									});
	}

	class UniformGrid
	{
	  public:
		static constexpr float kDefaultDensity = 3.0f;
		// `scene` must point at memory reachable by whichever Thrust backend this
		// binary was built with (device memory for the CUDA backend, ordinary host
		// memory for the TBB backend).
		explicit UniformGrid(const SceneView &scene, float target_triangles_per_cell = kDefaultDensity);
		~UniformGrid();
		UniformGrid(const UniformGrid &)			= delete;
		UniformGrid &operator=(const UniformGrid &) = delete;

		UniformGridView View() const;

		// Total number of (cell, triangle) references stored by this grid.
		uint32_t TriangleIndexCount() const;

		// Structural SAH cost estimate for this grid as currently built (see
		// sah_cost.h), normalized by `scene_bounds`'s own half surface area -- not
		// from casting rays, but the same surface-area-weighted "expected cost"
		// heuristic the grid density search (see src/app/grid_density_search.h) uses
		// to prefer one candidate density over another. Every cell is the same size
		// here, so the whole sum collapses to one closed-form expression: no per-cell
		// reduction needed at all.
		float ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const;

	  private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

} // namespace happyray
