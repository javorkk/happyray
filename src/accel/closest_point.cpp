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
		template <typename P, typename A> struct ClosestPointFunctor
		{
			SceneView		scene;
			AccelStructures accel;
			const Vec3f	   *queries;
			float			max_distance;

			HAPPYRAY_HD ClosestPointRecord operator()(uint32_t i) const
			{
				ClosestPointRecord record;
				FindClosestPoint<P, A>(scene, accel, queries[i], max_distance, &record);
				return record;
			}
		};
	} // namespace

	void FindClosestPoints(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   float				  max_distance,
						   ClosestPointRecord	 *out)
	{
		DispatchPrimitive(scene.primitive_type,
						  [&](auto prim)
						  {
							  DispatchAccel(accel.type,
											[&](auto structure)
											{
												using P			 = typename decltype(prim)::type;
												using A			 = decltype(structure);
												const auto first = thrust::counting_iterator<uint32_t>(0);
												thrust::transform(
													thrust::device,
													first,
													first + count,
													out,
													ClosestPointFunctor<P, A>{scene, accel, queries, max_distance});
											});
						  });
	}
} // namespace happyray
