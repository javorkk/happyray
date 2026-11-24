// Host/device closest-point queries: unsigned distance from a query position to the nearest
// primitive, through any acceleration structure or by brute force (the correctness oracle).
#pragma once

#include <cmath>
#include <type_traits>

#include "../app/scene.h"
#include "../core/primitive_traits.h"
#include "../core/vec.h"
#include "intersect_common.h"

namespace happyray
{

	struct ClosestPointRecord
	{
		float	 distance = INFINITY;
		Vec3f	 position;
		Vec3f	 normal;
		uint32_t prim = 0xffffffffu;
		float	 u = 0.0f, v = 0.0f; // Primitive-specific, see PrimitiveTraits<P>::ClosestPoint.
	};

	// Replaces `*best` if `prim` is closer. Equal distances keep the smaller id, so structures
	// resolve exact ties like brute force.
	template <typename P>
	HAPPYRAY_HD inline bool
	ClosestPointPrimitive(const SceneView &scene, uint32_t prim, const Vec3f &query, ClosestPointHit *best)
	{
		Vec3f		point;
		float		u = 0.0f, v = 0.0f;
		const float distance_squared = PrimitiveTraits<P>::ClosestPoint(PrimitiveTraits<P>::Fetch(scene, prim),
																		query,
																		best->distance_squared,
																		&point,
																		&u,
																		&v);
		if (!(distance_squared < INFINITY) || distance_squared > best->distance_squared ||
			(distance_squared == best->distance_squared && prim >= best->prim))
			return false;
		best->prim			   = prim;
		best->distance_squared = distance_squared;
		best->point			   = point;
		best->u				   = u;
		best->v				   = v;
		return true;
	}

	template <typename P>
	HAPPYRAY_HD inline bool
	ResolveClosestPoint(const SceneView &scene, const Vec3f &query, const ClosestPointHit &hit, ClosestPointRecord *out)
	{
		if (hit.prim == 0xffffffffu)
			return false;
		out->distance = std::sqrt(hit.distance_squared);
		out->position = hit.point;
		out->normal	  = PrimitiveTraits<P>::ClosestPointNormal(scene, query, hit);
		out->prim	  = hit.prim;
		out->u		  = hit.u;
		out->v		  = hit.v;
		return true;
	}

	// The search starts with `max_distance` as its bound: farther primitives are never reported.
	HAPPYRAY_HD inline ClosestPointHit InitialClosestPointHit(float max_distance)
	{
		ClosestPointHit hit;
		hit.distance_squared = max_distance * max_distance;
		return hit;
	}

	namespace detail
	{
		// Cell `cell` of a grid, grown like the builders' exact cell test so rounding never prunes a
		// cell holding the closest point.
		HAPPYRAY_HD inline float
		CellDistanceSquared(const Vec3f &query, const Vec3f &bounds_min, const Vec3f &cell_size, const Vec3i &cell)
		{
			const Vec3f center = bounds_min + Vec3f(cell.x + 0.5f, cell.y + 0.5f, cell.z + 0.5f) * cell_size;
			const Vec3f half   = cell_size * 0.505f;
			return DistanceSquaredToBBox(query, BBox(center - half, center + half));
		}

		// Lower bound on the distance to every cell at Chebyshev distance >= `ring` from `center`:
		// they lie beyond one of the six faces of the box of nearer cells.
		HAPPYRAY_HD inline float RingDistanceSquared(const Vec3f &query,
													 const Vec3f &bounds_min,
													 const Vec3f &cell_size,
													 const Vec3i &resolution,
													 const Vec3i &center,
													 int		  ring)
		{
			const Vec3f bounds_max = bounds_min + Vec3f(resolution.x, resolution.y, resolution.z) * cell_size;
			float		lower	   = INFINITY;
			for (int axis = 0; axis < 3; ++axis)
			{
				const int	c = (&center.x)[axis], r = (&resolution.x)[axis];
				const float lo = (&bounds_min.x)[axis], size = (&cell_size.x)[axis];
				if (c - ring >= 0)
				{
					BBox below(bounds_min, bounds_max);
					(&below.max_bound.x)[axis] = lo + (c - ring + 1) * size;
					lower					   = fminf(lower, DistanceSquaredToBBox(query, below));
				}
				if (c + ring < r)
				{
					BBox above(bounds_min, bounds_max);
					(&above.min_bound.x)[axis] = lo + (c + ring) * size;
					lower					   = fminf(lower, DistanceSquaredToBBox(query, above));
				}
			}
			return lower;
		}

		// Visits a grid's cells in rings of growing Chebyshev distance around the query's cell,
		// skipping cells farther than `bound()` and stopping once a whole ring is. `visit(index, cell)`
		// returns true to stop the query. Bounds are inclusive so equal-distance ties are seen.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool VisitCellsByDistance(const Vec3f &bounds_min,
													 const Vec3f &cell_size,
													 const Vec3f &inv_cell_size,
													 const Vec3i &resolution,
													 const Vec3f &query,
													 BoundFn	&&bound,
													 VisitFn	&&visit)
		{
			const Vec3f entry = (query - bounds_min) * inv_cell_size;
			const Vec3i center(ClampInt(static_cast<int>(std::floor(entry.x)), 0, resolution.x - 1),
							   ClampInt(static_cast<int>(std::floor(entry.y)), 0, resolution.y - 1),
							   ClampInt(static_cast<int>(std::floor(entry.z)), 0, resolution.z - 1));
			const Vec3i reach = Max(center, resolution - Vec3i(1, 1, 1) - center);
			const int	last_ring =
				  reach.x > reach.y ? (reach.x > reach.z ? reach.x : reach.z) : (reach.y > reach.z ? reach.y : reach.z);
			for (int ring = 0; ring <= last_ring; ++ring)
			{
				if (ring > 0 && RingDistanceSquared(query, bounds_min, cell_size, resolution, center, ring) > bound())
					return false;
				const int z_begin = center.z - ring > 0 ? center.z - ring : 0;
				const int z_end	  = center.z + ring < resolution.z - 1 ? center.z + ring : resolution.z - 1;
				const int y_begin = center.y - ring > 0 ? center.y - ring : 0;
				const int y_end	  = center.y + ring < resolution.y - 1 ? center.y + ring : resolution.y - 1;
				for (int z = z_begin; z <= z_end; ++z)
					for (int y = y_begin; y <= y_end; ++y)
					{
						// Rows on the ring's z or y faces are whole; others only touch its x faces.
						const bool face = z == center.z - ring || z == center.z + ring || y == center.y - ring ||
										  y == center.y + ring;
						const int step = face || ring == 0 ? 1 : 2 * ring;
						for (int x = center.x - ring; x <= center.x + ring; x += step)
						{
							if (x < 0 || x >= resolution.x)
								continue;
							const Vec3i cell(x, y, z);
							if (CellDistanceSquared(query, bounds_min, cell_size, cell) > bound())
								continue;
							if (visit(FlattenCellIndex(cell, resolution), cell))
								return true;
						}
					}
			}
			return false;
		}

		template <typename View, typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool
		VisitCellsByDistance(const View &grid, const Vec3f &query, BoundFn &&bound, VisitFn &&visit)
		{
			return VisitCellsByDistance(grid.bounds_min,
										grid.cell_size,
										grid.inv_cell_size,
										grid.resolution,
										query,
										bound,
										visit);
		}

		// Calls `visit(range)` for leaf cells in near-to-far rings; `visit` returns true to stop.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool
		VisitLeavesByDistance(const UniformGridView &grid, const Vec3f &query, BoundFn &&bound, VisitFn &&visit)
		{
			return VisitCellsByDistance(grid,
										query,
										bound,
										[&](uint32_t index, const Vec3i &)
										{ return visit(LoadRO(grid.cell_ranges + index)); });
		}

		// Rings over the top level; each non-empty top-level cell runs its own rings over its
		// nested grid.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool
		VisitLeavesByDistance(const TwoLevelGridView &grid, const Vec3f &query, BoundFn &&bound, VisitFn &&visit)
		{
			return VisitCellsByDistance(grid.top_level,
										query,
										bound,
										[&](uint32_t top_index, const Vec3i &)
										{
											if (LoadRO(grid.top_level_cells + top_index).IsEmpty())
												return false;
											const GridRangeView nested = SecondLevelView(grid, top_index);
											return VisitCellsByDistance(
												nested,
												query,
												bound,
												[&](uint32_t index, const Vec3i &)
												{ return visit(LoadRO(nested.cell_ranges + index)); });
										});
		}

		// Rings over the top level, then over each top-level cell's finest local voxels. A merged cell
		// spans many voxels, so it is visited only from the voxel holding its lattice point nearest to
		// the query, unless expansion grew it over that voxel (which then maps to a neighbor): then from
		// every voxel mapping to it.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool
		VisitLeavesByDistance(const IrregularGridView &grid, const Vec3f &query, BoundFn &&bound, VisitFn &&visit)
		{
			// Clamped just outside the lattice so far queries don't overflow the conversion.
			const Vec3f lattice =
				Min(Max((query - grid.top_level.bounds_min) * grid.global_inv_cell_size, Vec3f(-1.0f)),
					Vec3f(grid.global_resolution.x * grid.top_level.resolution.x,
						  grid.global_resolution.y * grid.top_level.resolution.y,
						  grid.global_resolution.z * grid.top_level.resolution.z));
			const Vec3i query_voxel(static_cast<int>(std::floor(lattice.x)),
									static_cast<int>(std::floor(lattice.y)),
									static_cast<int>(std::floor(lattice.z)));
			return VisitCellsByDistance(
				grid.top_level,
				query,
				bound,
				[&](uint32_t top_index, const Vec3i &)
				{
					const IrregularCellInfo info	= grid.cell_infos[top_index];
					const uint32_t		   *entries = grid.local_entries + info.entries_offset;
					return VisitCellsByDistance(
						info.bounds_min,
						info.cell_size,
						info.inv_cell_size,
						info.resolution,
						query,
						bound,
						[&](uint32_t index, const Vec3i &local)
						{
							const uint32_t		cell_index = LoadRO(entries + index);
							const IrregularCell cell	   = LoadRO(grid.local_cells + cell_index);
							if (cell.references.start == cell.references.end)
								return false;
							const Vec3i first(info.global_offset.x + (local.x << info.shift.x),
											  info.global_offset.y + (local.y << info.shift.y),
											  info.global_offset.z + (local.z << info.shift.z));
							const Vec3i nearest(ClampInt(query_voxel.x, cell.min.x, cell.max.x - 1),
												ClampInt(query_voxel.y, cell.min.y, cell.max.y - 1),
												ClampInt(query_voxel.z, cell.min.z, cell.max.z - 1));
							const bool	holds_nearest =
								nearest.x >= first.x && nearest.x < first.x + (1 << info.shift.x) &&
								nearest.y >= first.y && nearest.y < first.y + (1 << info.shift.y) &&
								nearest.z >= first.z && nearest.z < first.z + (1 << info.shift.z);
							if (!holds_nearest && LookupIrregularCell(grid, nearest) == cell_index)
								return false;
							return visit(cell.references);
						});
				});
		}

		template <typename GridView> HAPPYRAY_HD inline const uint32_t *LeafPrimIndices(const GridView &grid)
		{
			if constexpr (std::is_same_v<GridView, TwoLevelGridView>)
				return grid.second_level_prim_indices;
			else
				return grid.prim_indices;
		}
	} // namespace detail

	template <typename P = DynamicPrimitive, typename GridView>
	HAPPYRAY_HD inline bool FindClosestPointGrid(const SceneView	&scene,
												 const GridView		&grid,
												 const Vec3f		&query,
												 float				 max_distance,
												 ClosestPointRecord *out)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return FindClosestPointGrid<Q>(scene, grid, query, max_distance, out);
									 });
		}
		else
		{
			const uint32_t *prim_indices = detail::LeafPrimIndices(grid);
			ClosestPointHit hit			 = InitialClosestPointHit(max_distance);
			detail::VisitLeavesByDistance(
				grid,
				query,
				[&]() { return hit.distance_squared; },
				[&](CellRange range)
				{
					for (uint32_t i = range.start; i < range.end; ++i)
						ClosestPointPrimitive<P>(scene, LoadRO(prim_indices + i), query, &hit);
					return false;
				});
			return ResolveClosestPoint<P>(scene, query, hit, out);
		}
	}

	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool
	FindClosestPointBruteForce(const SceneView &scene, const Vec3f &query, float max_distance, ClosestPointRecord *out)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return FindClosestPointBruteForce<Q>(scene, query, max_distance, out);
									 });
		}
		else
		{
			ClosestPointHit hit = InitialClosestPointHit(max_distance);
			for (uint32_t prim = 0; prim < scene.primitive_count; ++prim)
				ClosestPointPrimitive<P>(scene, prim, query, &hit);
			return ResolveClosestPoint<P>(scene, query, hit, out);
		}
	}

	// `P` selects the primitive type and `A` the acceleration structure at compile time, as for
	// IntersectScene; DynamicPrimitive/DynamicAccel dispatch at runtime.
	template <typename P = DynamicPrimitive, typename A = DynamicAccel>
	HAPPYRAY_HD inline bool FindClosestPoint(const SceneView	   &scene,
											 const AccelStructures &accel,
											 const Vec3f		   &query,
											 float					max_distance,
											 ClosestPointRecord	   *out)
	{
		if constexpr (std::is_same_v<A, DynamicAccel>)
		{
			return DispatchAccel(
				accel.type,
				[&](auto tag) { return FindClosestPoint<P, decltype(tag)>(scene, accel, query, max_distance, out); });
		}
		else if constexpr (A::value == AccelType::kUniformGrid)
			return FindClosestPointGrid<P>(scene, accel.uniform_grid, query, max_distance, out);
		else if constexpr (A::value == AccelType::kTwoLevelGrid)
			return FindClosestPointGrid<P>(scene, accel.two_level_grid, query, max_distance, out);
		else if constexpr (A::value == AccelType::kIrregularGrid)
			return FindClosestPointGrid<P>(scene, accel.irregular_grid, query, max_distance, out);
		else
			return FindClosestPointBruteForce<P>(scene, query, max_distance, out);
	}

	// One FindClosestPoint per query, instantiated per primitive type and structure. `queries` and
	// `out` must be reachable by the active Thrust backend; misses keep a default record.
	void FindClosestPoints(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   float				  max_distance,
						   ClosestPointRecord	 *out);

} // namespace happyray
