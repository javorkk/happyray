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

#include "../app/scene.h"
#include "../core/bbox.h"
#include "../core/primitive.h"
#include "../core/primitive_traits.h"
#include "../core/ray.h"
#include "../core/triangle.h"
#include "../core/vec.h"
#include "sah_cost.h"

namespace happyray
{

	// How builders assign primitives to cells (counting pass / writing pass).
	enum class CellOverlapMode
	{
		kAabb,	 // AABB cells / AABB cells.
		kExact,	 // Exact cells / exact cells.
		kHybrid, // AABB cells / exact cells, unused slots trimmed.
	};

	// kExact on CUDA, kHybrid on TBB.
	CellOverlapMode DefaultCellOverlapMode();

	const char *CellOverlapModeName(CellOverlapMode mode);

	struct alignas(8) CellRange
	{
		uint32_t start = 0;
		uint32_t end   = 0;
	};
	static_assert(sizeof(CellRange) == 2 * sizeof(uint32_t));

	// Sizes a grid from the expanded geometry bounds and target primitive density.
	HAPPYRAY_HD inline Vec3i
	ComputeGridResolution(const BBox &bounds, uint32_t primitive_count, float target_prims_per_cell)
	{
		if (primitive_count == 0)
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
			std::cbrt(target_prims_per_cell * static_cast<float>(primitive_count) / volume);

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
		const CellRange *cell_ranges  = nullptr;
		const uint32_t	*prim_indices = nullptr; // Sorted by cell; indexes into SceneView.
	};

	HAPPYRAY_HD inline uint32_t FlattenCellIndex(int cx, int cy, int cz, int resolution_x, int resolution_y)
	{
		return static_cast<uint32_t>(cx) +
			   static_cast<uint32_t>(resolution_x) *
				   (static_cast<uint32_t>(cy) + static_cast<uint32_t>(resolution_y) * static_cast<uint32_t>(cz));
	}

	HAPPYRAY_HD inline uint32_t FlattenCellIndex(const Vec3i &cell, const Vec3i &resolution)
	{
		return FlattenCellIndex(cell.x, cell.y, cell.z, resolution.x, resolution.y);
	}

	namespace detail
	{

		// Number of grid cells overlapped by the primitive AABB's clamped cell range.
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

		// Exact primitive/cell test with the grid's 1% cell enlargement.
		template <typename P> struct CellIntersection
		{
			typename PrimitiveTraits<P>::CellOverlap intersects;

			HAPPYRAY_HD CellIntersection(const SceneView &scene, uint32_t prim)
				: CellIntersection(PrimitiveTraits<P>::Fetch(scene, prim))
			{
			}

			HAPPYRAY_HD explicit CellIntersection(const P &primitive) : intersects(primitive)
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

		// Shared by every grid builder: walks the primitive's AABB cell range and calls `emit(cell)` for
		// every cell in it (kAabb) or only for cells the primitive's actual shape overlaps (otherwise).
		// Exact overlap matters for long, thin, diagonal primitives, whose AABB range can be orders of
		// magnitude larger than the cells they touch.
		template <typename P, CellOverlapMode Mode, typename EmitFn>
		HAPPYRAY_HD inline uint32_t ForEachOverlappedCell(const P	  &primitive,
														  const BBox  &box,
														  const Vec3f &bounds_min,
														  const Vec3f &inv_cell_size,
														  const Vec3f &cell_size,
														  const Vec3i &resolution,
														  EmitFn	 &&emit)
		{
			Vec2i x_range, y_range, z_range;
			ComputeCellRange(box, bounds_min, inv_cell_size, resolution, &x_range, &y_range, &z_range);
			uint32_t count = 0;
			if constexpr (Mode == CellOverlapMode::kAabb)
			{
				for (int cz = z_range.x; cz <= z_range.y; ++cz)
					for (int cy = y_range.x; cy <= y_range.y; ++cy)
						for (int cx = x_range.x; cx <= x_range.y; ++cx)
						{
							emit(Vec3i(cx, cy, cz));
							++count;
						}
			}
			else
			{
				const CellIntersection<P> intersects(primitive);
				for (int cz = z_range.x; cz <= z_range.y; ++cz)
					for (int cy = y_range.x; cy <= y_range.y; ++cy)
						for (int cx = x_range.x; cx <= x_range.y; ++cx)
						{
							const Vec3f center = bounds_min + Vec3f(cx + 0.5f, cy + 0.5f, cz + 0.5f) * cell_size;
							if (intersects(center, cell_size))
							{
								emit(Vec3i(cx, cy, cz));
								++count;
							}
						}
			}
			return count;
		}

		// Counting pass: an upper bound of what ForEachOverlappedCell<P, Mode> emits (exact unless kHybrid).
		template <typename P, CellOverlapMode Mode>
		HAPPYRAY_HD inline uint32_t CountOverlappedCells(const P	 &primitive,
														 const BBox	 &box,
														 const Vec3f &bounds_min,
														 const Vec3f &inv_cell_size,
														 const Vec3f &cell_size,
														 const Vec3i &resolution)
		{
			if constexpr (Mode == CellOverlapMode::kExact)
				return ForEachOverlappedCell<P, Mode>(primitive,
													  box,
													  bounds_min,
													  inv_cell_size,
													  cell_size,
													  resolution,
													  [](const Vec3i &) {});
			else
				return CountCandidateCells(box, bounds_min, inv_cell_size, resolution);
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

		// Clip once, then visit ranges front to back. The visitor may shrink t_max
		// for a closest hit or return true to stop an occlusion query immediately.
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
			const Vec3f &origin		   = ray.origin;
			const Vec3f &dir		   = ray.direction;
			const Vec3f	 entry		   = (origin + dir * t_near - grid.bounds_min) * grid.inv_cell_size;
			const int	 resolution[3] = {grid.resolution.x, grid.resolution.y, grid.resolution.z};
			int			 cell[3]	   = {ClampInt(static_cast<int>(entry.x), 0, resolution[0] - 1),
										  ClampInt(static_cast<int>(entry.y), 0, resolution[1] - 1),
										  ClampInt(static_cast<int>(entry.z), 0, resolution[2] - 1)};
			const int	 increment[3]  = {dir.x > 0.0f ? 1 : -1, dir.y > 0.0f ? 1 : -1, dir.z > 0.0f ? 1 : -1};
			// Zero-direction axes never step, including when the origin is on a boundary.
			float next[3] = {
				dir.x != 0.0f
					? ((cell[0] + (dir.x > 0.0f)) * grid.cell_size.x + grid.bounds_min.x - origin.x) * inv_dir.x
					: INFINITY,
				dir.y != 0.0f
					? ((cell[1] + (dir.y > 0.0f)) * grid.cell_size.y + grid.bounds_min.y - origin.y) * inv_dir.y
					: INFINITY,
				dir.z != 0.0f
					? ((cell[2] + (dir.z > 0.0f)) * grid.cell_size.z + grid.bounds_min.z - origin.z) * inv_dir.z
					: INFINITY};
			const float delta[3]  = {grid.cell_size.x * std::fabs(inv_dir.x),
									 grid.cell_size.y * std::fabs(inv_dir.y),
									 grid.cell_size.z * std::fabs(inv_dir.z)};
			float		ray_t_max = ray.t_max;
			for (;;)
			{
				const uint32_t index = FlattenCellIndex(cell[0], cell[1], cell[2], resolution[0], resolution[1]);
				if (visit_cell(grid.cell_ranges[index], &ray_t_max))
					return true;

				// Legacy MIN_DIMENSION ordering: z wins ties, then y, then x.
				const int axis = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2);
				// Constant indices let the compiler keep DDA state in registers.
				switch (axis)
				{
				case 0:
					if (next[0] > ray_t_max)
						return false;
					cell[0] += increment[0];
					if (static_cast<unsigned>(cell[0]) >= static_cast<unsigned>(resolution[0]))
						return false;
					next[0] += delta[0];
					break;
				case 1:
					if (next[1] > ray_t_max)
						return false;
					cell[1] += increment[1];
					if (static_cast<unsigned>(cell[1]) >= static_cast<unsigned>(resolution[1]))
						return false;
					next[1] += delta[1];
					break;
				default:
					if (next[2] > ray_t_max)
						return false;
					cell[2] += increment[2];
					if (static_cast<unsigned>(cell[2]) >= static_cast<unsigned>(resolution[2]))
						return false;
					next[2] += delta[2];
					break;
				}
			}
		}

	} // namespace detail

	// Tests every primitive in the contiguous range [begin, end) of `prim_indices` against `ray`,
	// clipped to `*ray_t_max`, which shrinks (and `*hit` updates) on every closer hit. Returns true if
	// any primitive in the range was hit. Shared by all three grids' closest-hit queries.
	template <typename P>
	HAPPYRAY_HD inline bool IntersectPrimitivesClosest(const SceneView &scene,
													   const uint32_t  *prim_indices,
													   uint32_t			begin,
													   uint32_t			end,
													   const Ray	   &ray,
													   float		   *ray_t_max,
													   ClosestHit	   *hit)
	{
		bool hit_anything = false;
		Ray	 probe		  = ray;
		probe.t_max		  = *ray_t_max;

		for (uint32_t i = begin; i < end; ++i)
		{
			const uint32_t prim = prim_indices[i];
			float		   t, u = 0.0f, v = 0.0f;
			if (PrimitiveTraits<P>::Intersect(probe, PrimitiveTraits<P>::Fetch(scene, prim), &t, &u, &v))
			{
				probe.t_max	 = t;
				hit->prim	 = prim;
				hit->t		 = t;
				hit->u		 = u;
				hit->v		 = v;
				hit_anything = true;
			}
		}

		*ray_t_max = probe.t_max;

		return hit_anything;
	}

	// Fills `out_hit` (position, shading normal, material) from the closest hit found during
	// traversal. Returns false if nothing was hit.
	template <typename P>
	HAPPYRAY_HD inline bool
	ResolveClosestHit(const SceneView &scene, const Ray &ray, const ClosestHit &hit, HitRecord *out_hit)
	{
		if (hit.prim == 0xffffffffu)
		{
			return false;
		}
		out_hit->t			 = hit.t;
		out_hit->position	 = ray.At(hit.t);
		out_hit->normal		 = PrimitiveTraits<P>::Normal(scene, ray, hit);
		out_hit->material_id = FetchMaterialId(scene, hit.prim);
		return true;
	}

	// Returns true as soon as any primitive in the contiguous range [begin, end) occludes `ray`.
	template <typename P>
	HAPPYRAY_HD inline bool TestPrimitivesInRangeOccluded(const SceneView &scene,
														  const uint32_t  *prim_indices,
														  uint32_t		   begin,
														  uint32_t		   end,
														  const Ray		  &ray)
	{
		for (uint32_t i = begin; i < end; ++i)
		{
			float t, u, v;
			if (PrimitiveTraits<P>::Intersect(ray, PrimitiveTraits<P>::Fetch(scene, prim_indices[i]), &t, &u, &v))
			{
				return true;
			}
		}
		return false;
	}

	namespace detail
	{
		template <typename P> struct ClosestHitVisitor
		{
			const SceneView &scene;
			const uint32_t	*prim_indices;
			const Ray		&ray;
			ClosestHit		&hit;

			HAPPYRAY_HD bool operator()(CellRange range, float *t_max) const
			{
				IntersectPrimitivesClosest<P>(scene, prim_indices, range.start, range.end, ray, t_max, &hit);
				return false;
			}
		};

		template <typename P> struct OcclusionVisitor
		{
			const SceneView &scene;
			const uint32_t	*prim_indices;
			const Ray		&ray;

			HAPPYRAY_HD bool operator()(CellRange range, float *) const
			{
				return TestPrimitivesInRangeOccluded<P>(scene, prim_indices, range.start, range.end, ray);
			}
		};
	} // namespace detail

	// Finds the closest hit across the primitives referenced by `grid`'s cells, in the same sense as
	// IntersectSceneBruteForce. Returns false if the ray hits nothing.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool
	IntersectSceneGrid(const SceneView &scene, const UniformGridView &grid, const Ray &ray, HitRecord *out_hit)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IntersectSceneGrid<Q>(scene, grid, ray, out_hit);
									 });
		}
		else
		{
			ClosestHit hit;
			detail::TraverseGrid(grid, ray, detail::ClosestHitVisitor<P>{scene, grid.prim_indices, ray, hit});
			return ResolveClosestHit<P>(scene, ray, hit, out_hit);
		}
	}

	// Returns true as soon as any primitive referenced by `grid`'s cells occludes `ray`.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool IsOccludedGrid(const SceneView &scene, const UniformGridView &grid, const Ray &ray)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IsOccludedGrid<Q>(scene, grid, ray);
									 });
		}
		else
		{
			return detail::TraverseGrid(grid, ray, detail::OcclusionVisitor<P>{scene, grid.prim_indices, ray});
		}
	}

	class UniformGrid
	{
	  public:
		static constexpr float kDefaultDensity = 3.0f;
		// `scene` must point at memory reachable by whichever Thrust backend this
		// binary was built with (device memory for the CUDA backend, ordinary host
		// memory for the TBB backend).
		explicit UniformGrid(const SceneView &scene,
							 float			  target_prims_per_cell = kDefaultDensity,
							 CellOverlapMode  mode					= DefaultCellOverlapMode());
		~UniformGrid();
		UniformGrid(const UniformGrid &)			= delete;
		UniformGrid &operator=(const UniformGrid &) = delete;

		UniformGridView View() const;

		// Total number of (cell, primitive) references stored by this grid.
		uint32_t PrimIndexCount() const;

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
