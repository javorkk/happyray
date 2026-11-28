// Batched k-nearest-primitive queries.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "knn.h"

#include <stdexcept>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace happyray
{
	namespace
	{
		struct KNearestArgs
		{
			SceneView			  scene;
			AccelStructures		  accel;
			const Vec3f			 *queries;
			uint32_t			  k;
			float				  max_distance;
			ClosestPointRecord	 *out;
			uint32_t			 *found;
			ClosestPointCounters *counters;
		};

		template <typename P, typename A, bool kCount> struct KNearestFunctor
		{
			KNearestArgs args;

			HAPPYRAY_HD void operator()(uint32_t i) const
			{
				ClosestPointRecord *records = args.out + static_cast<size_t>(i) * args.k;
				const Vec3f		   &query	= args.queries[i];
				uint32_t			n;
				if constexpr (kCount)
				{
					ClosestPointCounters counted;
					n = FindKNearest<P, A>(args.scene, args.accel, query, args.k, args.max_distance, records, &counted);
					args.counters[i] = counted;
				}
				else
					n = FindKNearest<P, A>(args.scene, args.accel, query, args.k, args.max_distance, records);
				if (args.found)
					args.found[i] = n;
			}
		};

		template <typename P, typename A, bool kCount> void Run(const KNearestArgs &args, uint32_t count)
		{
			const auto first = thrust::counting_iterator<uint32_t>(0);
			thrust::for_each(thrust::device, first, first + count, KNearestFunctor<P, A, kCount>{args});
		}
	} // namespace

	void FindKNearestBatch(const SceneView		 &scene,
						   const AccelStructures &accel,
						   const Vec3f			 *queries,
						   uint32_t				  count,
						   uint32_t				  k,
						   float				  max_distance,
						   ClosestPointRecord	 *out,
						   uint32_t				 *found,
						   ClosestPointCounters	 *counters)
	{
		if (k > kMaxKNearest)
			throw std::invalid_argument("FindKNearestBatch: k exceeds kMaxKNearest");
		const KNearestArgs args{scene, accel, queries, k, max_distance, out, found, counters};
		DispatchPrimitive(scene.primitive_type,
						  [&](auto prim)
						  {
							  DispatchAccel(accel.type,
											[&](auto structure)
											{
												using P = typename decltype(prim)::type;
												using A = decltype(structure);
												if (counters)
													Run<P, A, true>(args, count);
												else
													Run<P, A, false>(args, count);
											});
						  });
	}
} // namespace happyray
