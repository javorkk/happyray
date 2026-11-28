// Host/device k-nearest-primitive queries: the k primitives with the smallest unsigned distance to
// a query position, through any acceleration structure or by brute force (the correctness oracle).
#pragma once

#include "closest_point.h"

namespace happyray
{

	constexpr uint32_t kMaxKNearest = 64;

	namespace detail
	{
		struct KNearestEntry
		{
			float	 distance_squared;
			uint32_t prim;
		};

		// The best primitives so far, sorted by distance, then id, like ClosestPointPrimitive's ties.
		struct KNearestList
		{
			KNearestEntry entries[kMaxKNearest];
			uint32_t	  k		= 0;
			uint32_t	  count = 0;
			float		  max_distance_squared;

			HAPPYRAY_HD KNearestList(uint32_t k_, float max_distance)
				: k(k_), max_distance_squared(max_distance * max_distance)
			{
			}

			HAPPYRAY_HD float Bound() const
			{
				return count == k ? entries[k - 1].distance_squared : max_distance_squared;
			}

			// Grids reference a primitive from several cells, so repeated offers are dropped.
			template <typename P> HAPPYRAY_HD void Offer(const SceneView &scene, uint32_t prim, const Vec3f &query)
			{
				Vec3f		point;
				float		u = 0.0f, v = 0.0f;
				const float bound			 = Bound();
				const float distance_squared = PrimitiveTraits<P>::ClosestPoint(PrimitiveTraits<P>::Fetch(scene, prim),
																				query,
																				bound,
																				&point,
																				&u,
																				&v);
				if (!(distance_squared < INFINITY) || distance_squared > bound)
					return;
				for (uint32_t i = 0; i < count; ++i)
					if (entries[i].prim == prim)
						return;
				uint32_t slot = count;
				while (slot > 0 &&
					   (entries[slot - 1].distance_squared > distance_squared ||
						(entries[slot - 1].distance_squared == distance_squared && entries[slot - 1].prim > prim)))
					--slot;
				if (slot == k)
					return;
				for (uint32_t i = count < k ? count : k - 1; i > slot; --i)
					entries[i] = entries[i - 1];
				entries[slot] = {distance_squared, prim};
				count += count < k;
			}

			// Recomputes the winners' closest points (cheaper than carrying them through the search).
			template <typename P>
			HAPPYRAY_HD uint32_t Resolve(const SceneView &scene, const Vec3f &query, ClosestPointRecord *out) const
			{
				for (uint32_t i = 0; i < k; ++i)
				{
					out[i] = ClosestPointRecord();
					if (i >= count)
						continue;
					ClosestPointHit hit;
					hit.prim			 = entries[i].prim;
					hit.distance_squared = entries[i].distance_squared;
					PrimitiveTraits<P>::ClosestPoint(PrimitiveTraits<P>::Fetch(scene, hit.prim),
													 query,
													 INFINITY,
													 &hit.point,
													 &hit.u,
													 &hit.v);
					ResolveClosestPoint<P>(scene, query, hit, out + i);
				}
				return count;
			}
		};

		HAPPYRAY_HD inline uint32_t ClampK(uint32_t k)
		{
			return k < kMaxKNearest ? k : kMaxKNearest;
		}
	} // namespace detail

	// Writes the k nearest primitives within `max_distance` to `out[0, k)`, near to far, and returns
	// how many were found; the rest of `out` gets default records. `k` is clamped to kMaxKNearest.
	template <typename P = DynamicPrimitive, typename GridView>
	HAPPYRAY_HD inline uint32_t FindKNearestGrid(const SceneView	  &scene,
												 const GridView		  &grid,
												 const Vec3f		  &query,
												 uint32_t			   k,
												 float				   max_distance,
												 ClosestPointRecord	  *out,
												 ClosestPointCounters *counters = nullptr)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return FindKNearestGrid<Q>(scene, grid, query, k, max_distance, out, counters);
									 });
		}
		else
		{
			k = detail::ClampK(k);
			if (k == 0)
				return 0;
			detail::KNearestList list(k, max_distance);
			detail::SearchGrid(
				grid,
				query,
				[&]() { return list.Bound(); },
				[&](uint32_t prim) { list.Offer<P>(scene, prim, query); },
				counters);
			return list.Resolve<P>(scene, query, out);
		}
	}

	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline uint32_t FindKNearestBruteForce(const SceneView		&scene,
													   const Vec3f			&query,
													   uint32_t				 k,
													   float				 max_distance,
													   ClosestPointRecord	*out,
													   ClosestPointCounters *counters = nullptr)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return FindKNearestBruteForce<Q>(scene, query, k, max_distance, out, counters);
									 });
		}
		else
		{
			k = detail::ClampK(k);
			if (k == 0)
				return 0;
			detail::KNearestList list(k, max_distance);
			if (counters)
				counters->prims_tested += scene.primitive_count;
			for (uint32_t prim = 0; prim < scene.primitive_count; ++prim)
				list.Offer<P>(scene, prim, query);
			return list.Resolve<P>(scene, query, out);
		}
	}

	// `P` and `A` select the primitive type and acceleration structure as for FindClosestPoint.
	template <typename P = DynamicPrimitive, typename A = DynamicAccel>
	HAPPYRAY_HD inline uint32_t FindKNearest(const SceneView	   &scene,
											 const AccelStructures &accel,
											 const Vec3f		   &query,
											 uint32_t				k,
											 float					max_distance,
											 ClosestPointRecord	   *out,
											 ClosestPointCounters  *counters = nullptr)
	{
		if constexpr (std::is_same_v<A, DynamicAccel>)
		{
			return DispatchAccel(
				accel.type,
				[&](auto tag)
				{ return FindKNearest<P, decltype(tag)>(scene, accel, query, k, max_distance, out, counters); });
		}
		else if constexpr (A::value == AccelType::kUniformGrid)
			return FindKNearestGrid<P>(scene, accel.uniform_grid, query, k, max_distance, out, counters);
		else if constexpr (A::value == AccelType::kTwoLevelGrid)
			return FindKNearestGrid<P>(scene, accel.two_level_grid, query, k, max_distance, out, counters);
		else if constexpr (A::value == AccelType::kIrregularGrid)
			return FindKNearestGrid<P>(scene, accel.irregular_grid, query, k, max_distance, out, counters);
		else
			return FindKNearestBruteForce<P>(scene, query, k, max_distance, out, counters);
	}

	// One FindKNearest per query; query i writes `out[i * k, (i + 1) * k)` and, if given, `found[i]`.
	// Pointers must be reachable by the active Thrust backend. Throws if k > kMaxKNearest.
	void FindKNearestBatch(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   uint32_t				  k,
						   float				  max_distance,
						   ClosestPointRecord	 *out,
						   uint32_t				 *found	   = nullptr,
						   ClosestPointCounters	 *counters = nullptr);

} // namespace happyray
