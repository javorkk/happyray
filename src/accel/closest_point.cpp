// Batched closest-point queries.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "closest_point.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace happyray
{
	namespace
	{
		template <typename P, typename A, bool kCount> struct ClosestPointFunctor
		{
			SceneView			  scene;
			AccelStructures		  accel;
			const Vec3f			 *queries;
			float				  max_distance;
			ClosestPointCounters *counters;

			HAPPYRAY_HD ClosestPointRecord operator()(uint32_t i) const
			{
				ClosestPointRecord record;
				if constexpr (kCount)
				{
					ClosestPointCounters counted;
					FindClosestPoint<P, A>(scene, accel, queries[i], max_distance, &record, &counted);
					counters[i] = counted;
				}
				else
					FindClosestPoint<P, A>(scene, accel, queries[i], max_distance, &record);
				return record;
			}
		};

		template <typename P, typename A, bool kCount>
		void Run(const SceneView	   &scene,
				 const AccelStructures &accel,
				 const Vec3f		   *queries,
				 uint32_t				count,
				 float					max_distance,
				 ClosestPointRecord	   *out,
				 ClosestPointCounters  *counters)
		{
			const auto first = thrust::counting_iterator<uint32_t>(0);
			thrust::transform(thrust::device,
							  first,
							  first + count,
							  out,
							  ClosestPointFunctor<P, A, kCount>{scene, accel, queries, max_distance, counters});
		}
	} // namespace

	void FindClosestPoints(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   float				  max_distance,
						   ClosestPointRecord	 *out,
						   ClosestPointCounters	 *counters)
	{
		DispatchPrimitive(scene.primitive_type,
						  [&](auto prim)
						  {
							  DispatchAccel(
								  accel.type,
								  [&](auto structure)
								  {
									  using P = typename decltype(prim)::type;
									  using A = decltype(structure);
									  if (counters)
										  Run<P, A, true>(scene, accel, queries, count, max_distance, out, counters);
									  else
										  Run<P, A, false>(scene, accel, queries, count, max_distance, out, nullptr);
								  });
						  });
	}
} // namespace happyray
