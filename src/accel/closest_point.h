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

	// Per-query work of a closest-point search, for profiling. The fields don't overlap.
	struct ClosestPointCounters
	{
		uint32_t top_cells_examined = 0; // Top-level distance tests (two-level, irregular).
		uint32_t top_cells_empty	= 0; // Top-level cells within the bound holding no references.
		uint32_t rows_examined		= 0; // Irregular: distance tests of local voxel rows.
		uint32_t cells_examined		= 0; // Leaf-level distance tests (irregular: voxel steps along rows).
		uint32_t leaves_loaded		= 0; // Leaf cells loaded after passing their distance test.
		uint32_t leaves_empty		= 0; // Loaded leaf cells without references.
		uint32_t leaves_skipped		= 0; // Irregular: non-empty cells left to the run nearest the query.
		uint32_t nearest_lookups	= 0; // Irregular: voxel lookups deciding which run visits a cell.
		uint32_t leaves_visited		= 0; // Non-empty leaf cells whose primitives were tested.
		uint32_t prims_tested		= 0; // Primitive tests; a primitive counts once per leaf holding it.
	};

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

		struct FirstRingZero
		{
			HAPPYRAY_HD int operator()(uint32_t) const
			{
				return 0;
			}
		};

		// Visits a grid's cells in rings of growing Chebyshev distance around the query's cell,
		// skipping cells farther than `bound()` and stopping once a whole ring is. `visit(index, cell)`
		// returns true to stop the query. Bounds are inclusive so equal-distance ties are seen. Rings
		// start at `first_ring(center index)`, which may skip rings holding only empty cells.
		template <typename BoundFn, typename VisitFn, typename FirstRingFn = FirstRingZero>
		HAPPYRAY_HD inline bool VisitCellsByDistance(const Vec3f &bounds_min,
													 const Vec3f &cell_size,
													 const Vec3f &inv_cell_size,
													 const Vec3i &resolution,
													 const Vec3f &query,
													 BoundFn	&&bound,
													 VisitFn	&&visit,
													 uint32_t	 *examined,
													 FirstRingFn  first_ring = {})
		{
			const Vec3f entry = (query - bounds_min) * inv_cell_size;
			const Vec3i center(ClampInt(static_cast<int>(std::floor(entry.x)), 0, resolution.x - 1),
							   ClampInt(static_cast<int>(std::floor(entry.y)), 0, resolution.y - 1),
							   ClampInt(static_cast<int>(std::floor(entry.z)), 0, resolution.z - 1));
			const Vec3i reach = Max(center, resolution - Vec3i(1, 1, 1) - center);
			const int	last_ring =
				  reach.x > reach.y ? (reach.x > reach.z ? reach.x : reach.z) : (reach.y > reach.z ? reach.y : reach.z);
			int start = 0;
			if constexpr (!std::is_same_v<FirstRingFn, FirstRingZero>)
				start = first_ring(FlattenCellIndex(center, resolution));
			for (int ring = start; ring <= last_ring; ++ring)
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
							if (examined)
								++*examined;
							if (CellDistanceSquared(query, bounds_min, cell_size, cell) > bound())
								continue;
							if (visit(FlattenCellIndex(cell, resolution), cell))
								return true;
						}
					}
			}
			return false;
		}

		template <typename View, typename BoundFn, typename VisitFn, typename FirstRingFn = FirstRingZero>
		HAPPYRAY_HD inline bool VisitCellsByDistance(const View	 &grid,
													 const Vec3f &query,
													 BoundFn	&&bound,
													 VisitFn	&&visit,
													 uint32_t	 *examined,
													 FirstRingFn  first_ring = {})
		{
			return VisitCellsByDistance(grid.bounds_min,
										grid.cell_size,
										grid.inv_cell_size,
										grid.resolution,
										query,
										bound,
										visit,
										examined,
										first_ring);
		}

		// The first ring holding a non-empty cell, from a grid's per-cell empty radius.
		struct FirstRingFromRadius
		{
			const uint8_t *radius;
			uint32_t	   offset;

			HAPPYRAY_HD int operator()(uint32_t center) const
			{
				return radius ? LoadRO(radius + offset + center) : 0;
			}
		};

		// Loads a leaf cell's range, counting it and whether it is empty.
		HAPPYRAY_HD inline CellRange LoadLeaf(const CellRange *ranges, uint32_t index, ClosestPointCounters *counters)
		{
			const CellRange range = LoadRO(ranges + index);
			if (counters)
			{
				++counters->leaves_loaded;
				counters->leaves_empty += range.start == range.end;
			}
			return range;
		}

		// Address of one counter, or null without counters.
		HAPPYRAY_HD inline uint32_t *Counter(ClosestPointCounters *counters, uint32_t ClosestPointCounters::*field)
		{
			return counters ? &(counters->*field) : nullptr;
		}

		// Calls `visit(range)` for leaf cells in near-to-far rings; `visit` returns true to stop.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool VisitLeavesByDistance(const UniformGridView &grid,
													  const Vec3f			&query,
													  BoundFn			   &&bound,
													  VisitFn			   &&visit,
													  ClosestPointCounters	*counters)
		{
			return VisitCellsByDistance(
				grid,
				query,
				bound,
				[&](uint32_t index, const Vec3i &) { return visit(LoadLeaf(grid.cell_ranges, index, counters)); },
				Counter(counters, &ClosestPointCounters::cells_examined));
		}

		// Rings over the top level; each non-empty top-level cell runs its own rings over its
		// nested grid.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool VisitLeavesByDistance(const TwoLevelGridClosestPointView &grid,
													  const Vec3f						 &query,
													  BoundFn							&&bound,
													  VisitFn							&&visit,
													  ClosestPointCounters				 *counters)
		{
			return VisitCellsByDistance(
				grid.top_level,
				query,
				bound,
				[&](uint32_t top_index, const Vec3i &)
				{
					if (LoadRO(grid.top_level_cells + top_index).IsEmpty())
					{
						if (counters)
							++counters->top_cells_empty;
						return false;
					}
					const GridRangeView nested = SecondLevelView(grid, top_index);
					return VisitCellsByDistance(
						nested,
						query,
						bound,
						[&](uint32_t index, const Vec3i &)
						{ return visit(LoadLeaf(nested.cell_ranges, index, counters)); },
						Counter(counters, &ClosestPointCounters::cells_examined),
						FirstRingFromRadius{grid.second_level_empty_radius,
											static_cast<uint32_t>(nested.cell_ranges - grid.second_level_cell_ranges)});
				},
				Counter(counters, &ClosestPointCounters::top_cells_examined),
				FirstRingFromRadius{grid.top_level_empty_radius, 0});
		}

		// Rings over the top level; in each top-level cell, rings over its rows of local voxels, each
		// walked outwards from the query along x, a merged cell at a time. Voxels a cell's bounds cover
		// map to it or to cells whose references are a subset (HAGrid's expand), so they are skipped.
		// A cell is visited from the run holding its lattice voxel nearest to the query, unless that
		// voxel maps to another cell (expansion grew it there): then from every run mapping to it.
		template <typename BoundFn, typename VisitFn>
		HAPPYRAY_HD inline bool VisitLeavesByDistance(const IrregularGridView &grid,
													  const Vec3f			  &query,
													  BoundFn				 &&bound,
													  VisitFn				 &&visit,
													  ClosestPointCounters	  *counters)
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
					if (counters)
					{
						const CellRange top = grid.top_level.cell_ranges[top_index];
						counters->top_cells_empty += top.start == top.end;
					}
					const IrregularCellInfo info	= grid.cell_infos[top_index];
					const uint32_t		   *entries = grid.local_entries + info.entries_offset;
					const Vec3i			   &offset	= info.global_offset;
					const Vec3i			   &shift	= info.shift;
					const Vec3i			   &res		= info.resolution;
					const int				start_x =
						ClampInt(static_cast<int>(std::floor((query.x - info.bounds_min.x) * info.inv_cell_size.x)),
								 0,
								 res.x - 1);
					// Visits `cell` (at `cell_index`) for the run of local voxels [begin, end) of `row`.
					auto visit_run =
						[&](const IrregularCell &cell, uint32_t cell_index, int begin, int end, const Vec3i &row)
					{
						if (cell.references.start == cell.references.end)
						{
							if (counters)
								++counters->leaves_empty;
							return false;
						}
						const Vec3i nearest(ClampInt(query_voxel.x, cell.min.x, cell.max.x - 1),
											ClampInt(query_voxel.y, cell.min.y, cell.max.y - 1),
											ClampInt(query_voxel.z, cell.min.z, cell.max.z - 1));
						const int	row_y = offset.y + (row.y << shift.y), row_z = offset.z + (row.z << shift.z);
						const bool	holds_nearest = nearest.x >= offset.x + (begin << shift.x) &&
												   nearest.x < offset.x + (end << shift.x) && nearest.y >= row_y &&
												   nearest.y < row_y + (1 << shift.y) && nearest.z >= row_z &&
												   nearest.z < row_z + (1 << shift.z);
						if (!holds_nearest)
						{
							if (counters)
								++counters->nearest_lookups;
							if (LookupIrregularCell(grid, nearest) == cell_index)
							{
								if (counters)
									++counters->leaves_skipped;
								return false;
							}
						}
						return visit(cell.references);
					};
					auto lookup = [&](int x, const Vec3i &row, uint32_t *cell_index)
					{
						if (counters)
							++counters->leaves_loaded;
						*cell_index = LoadRO(entries + FlattenCellIndex(Vec3i(x, row.y, row.z), res));
						return LoadRO(grid.local_cells + *cell_index);
					};
					const Vec3f row_size(info.cell_size.x * res.x, info.cell_size.y, info.cell_size.z);
					const Vec3f row_inv(info.inv_cell_size.x / res.x, info.inv_cell_size.y, info.inv_cell_size.z);
					return VisitCellsByDistance(
						info.bounds_min,
						row_size,
						row_inv,
						Vec3i(1, res.y, res.z),
						query,
						bound,
						[&](uint32_t, const Vec3i &row)
						{
							// Voxels get farther from the query in both directions from `start_x`.
							for (int x = start_x; x < res.x;)
							{
								if (counters)
									++counters->cells_examined;
								if (CellDistanceSquared(query,
														info.bounds_min,
														info.cell_size,
														Vec3i(x, row.y, row.z)) > bound())
									break;
								uint32_t			cell_index;
								const IrregularCell cell = lookup(x, row, &cell_index);
								int					next = (cell.max.x - offset.x) >> shift.x;
								next					 = next > x ? (next < res.x ? next : res.x) : x + 1;
								if (visit_run(cell, cell_index, x, next, row))
									return true;
								x = next;
							}
							for (int x = start_x - 1; x >= 0;)
							{
								if (counters)
									++counters->cells_examined;
								if (CellDistanceSquared(query,
														info.bounds_min,
														info.cell_size,
														Vec3i(x, row.y, row.z)) > bound())
									break;
								uint32_t			cell_index;
								const IrregularCell cell = lookup(x, row, &cell_index);
								int previous = cell.min.x > offset.x ? (cell.min.x - 1 - offset.x) >> shift.x : -1;
								previous	 = previous < x ? previous : x - 1;
								if (visit_run(cell, cell_index, previous + 1, x + 1, row))
									return true;
								x = previous;
							}
							return false;
						},
						Counter(counters, &ClosestPointCounters::rows_examined));
				},
				Counter(counters, &ClosestPointCounters::top_cells_examined));
		}

		template <typename GridView> HAPPYRAY_HD inline const uint32_t *LeafPrimIndices(const GridView &grid)
		{
			if constexpr (std::is_base_of_v<TwoLevelGridView, GridView>)
				return grid.second_level_prim_indices;
			else
				return grid.prim_indices;
		}
	} // namespace detail

	// `counters`, if given, accumulates the search's work.
	template <typename P = DynamicPrimitive, typename GridView>
	HAPPYRAY_HD inline bool FindClosestPointGrid(const SceneView	  &scene,
												 const GridView		  &grid,
												 const Vec3f		  &query,
												 float				   max_distance,
												 ClosestPointRecord	  *out,
												 ClosestPointCounters *counters = nullptr)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(
				scene.primitive_type,
				[&](auto tag)
				{
					using Q = typename decltype(tag)::type;
					return FindClosestPointGrid<Q>(scene, grid, query, max_distance, out, counters);
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
					if (counters && range.start != range.end)
					{
						++counters->leaves_visited;
						counters->prims_tested += range.end - range.start;
					}
					for (uint32_t i = range.start; i < range.end; ++i)
						ClosestPointPrimitive<P>(scene, LoadRO(prim_indices + i), query, &hit);
					return false;
				},
				counters);
			return ResolveClosestPoint<P>(scene, query, hit, out);
		}
	}

	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool FindClosestPointBruteForce(const SceneView		&scene,
													   const Vec3f			&query,
													   float				 max_distance,
													   ClosestPointRecord	*out,
													   ClosestPointCounters *counters = nullptr)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(
				scene.primitive_type,
				[&](auto tag)
				{
					using Q = typename decltype(tag)::type;
					return FindClosestPointBruteForce<Q>(scene, query, max_distance, out, counters);
				});
		}
		else
		{
			ClosestPointHit hit = InitialClosestPointHit(max_distance);
			if (counters)
				counters->prims_tested += scene.primitive_count;
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
											 ClosestPointRecord	   *out,
											 ClosestPointCounters  *counters = nullptr)
	{
		if constexpr (std::is_same_v<A, DynamicAccel>)
		{
			return DispatchAccel(
				accel.type,
				[&](auto tag)
				{ return FindClosestPoint<P, decltype(tag)>(scene, accel, query, max_distance, out, counters); });
		}
		else if constexpr (A::value == AccelType::kUniformGrid)
			return FindClosestPointGrid<P>(scene, accel.uniform_grid, query, max_distance, out, counters);
		else if constexpr (A::value == AccelType::kTwoLevelGrid)
			return FindClosestPointGrid<P>(scene, accel.two_level_grid, query, max_distance, out, counters);
		else if constexpr (A::value == AccelType::kIrregularGrid)
			return FindClosestPointGrid<P>(scene, accel.irregular_grid, query, max_distance, out, counters);
		else
			return FindClosestPointBruteForce<P>(scene, query, max_distance, out, counters);
	}

	// One FindClosestPoint per query, instantiated per primitive type and structure. Pointers must be
	// reachable by the active Thrust backend; misses keep a default record. `counters`, if given,
	// receives each query's work (from a separate instantiation, so the default path pays nothing).
	void FindClosestPoints(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   float				  max_distance,
						   ClosestPointRecord	 *out,
						   ClosestPointCounters	 *counters = nullptr);

} // namespace happyray
