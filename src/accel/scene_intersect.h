// Shared host/device scene intersection and occlusion queries, independent of the integrator.
// Brute force is the correctness oracle for every acceleration structure.
#pragma once

#include "../app/scene.h"
#include "../core/primitive.h"
#include "../core/ray.h"
#include "../core/triangle.h"
#include "../core/vec.h"
#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace happyray
{

	// Finds the closest hit across every primitive in `scene`. Returns false if the ray hits
	// nothing.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool IntersectSceneBruteForce(const SceneView &scene, const Ray &ray, HitRecord *out_hit)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IntersectSceneBruteForce<Q>(scene, ray, out_hit);
									 });
		}
		else
		{
			ClosestHit hit;
			float	   t_max = ray.t_max;
			for (uint32_t prim = 0; prim < scene.primitive_count; ++prim)
				IntersectPrimitivesClosest<P>(scene, &prim, 0, 1, ray, &t_max, &hit);
			return ResolveClosestHit<P>(scene, ray, hit, out_hit);
		}
	}

	// Returns true if any primitive occludes `ray` (no need for the closest hit -- used for AO
	// occlusion tests, where only a boolean "something is in the way" result is needed).
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool IsOccludedBruteForce(const SceneView &scene, const Ray &ray)
	{
		if constexpr (std::is_same_v<P, DynamicPrimitive>)
		{
			return DispatchPrimitive(scene.primitive_type,
									 [&](auto tag)
									 {
										 using Q = typename decltype(tag)::type;
										 return IsOccludedBruteForce<Q>(scene, ray);
									 });
		}
		else
		{
			for (uint32_t prim = 0; prim < scene.primitive_count; ++prim)
			{
				if (TestPrimitivesInRangeOccluded<P>(scene, &prim, 0, 1, ray))
					return true;
			}
			return false;
		}
	}

	// Selects which acceleration structure (if any) IntersectScene/IsOccluded
	// trace against. Each later phase adds one more AccelType enum value and one more (optional)
	// *View member here -- the dispatch in IntersectScene/IsOccluded stays a simple switch, same
	// cost on CPU and GPU.
	enum class AccelType
	{
		kBruteForce,
		kUniformGrid,
		kTwoLevelGrid,
		kIrregularGrid
	};

	struct AccelStructures
	{
		AccelType		  type = AccelType::kBruteForce;
		UniformGridView	  uniform_grid;	  // Valid/meaningful only when type == kUniformGrid.
		TwoLevelGridView  two_level_grid; // Valid/meaningful only when type == kTwoLevelGrid.
		IrregularGridView irregular_grid; // Valid/meaningful only when type == kIrregularGrid.
	};

	// `P` selects the primitive type at compile time; DynamicPrimitive dispatches at runtime.
	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool
	IntersectScene(const SceneView &scene, const AccelStructures &accel, const Ray &ray, HitRecord *out_hit)
	{
		switch (accel.type)
		{
		case AccelType::kUniformGrid:
			return IntersectSceneGrid<P>(scene, accel.uniform_grid, ray, out_hit);
		case AccelType::kTwoLevelGrid:
			return IntersectSceneTwoLevelGrid<P>(scene, accel.two_level_grid, ray, out_hit);
		case AccelType::kIrregularGrid:
			return IntersectSceneIrregularGrid<P>(scene, accel.irregular_grid, ray, out_hit);
		case AccelType::kBruteForce:
		default:
			return IntersectSceneBruteForce<P>(scene, ray, out_hit);
		}
	}

	template <typename P = DynamicPrimitive>
	HAPPYRAY_HD inline bool IsOccluded(const SceneView &scene, const AccelStructures &accel, const Ray &ray)
	{
		switch (accel.type)
		{
		case AccelType::kUniformGrid:
			return IsOccludedGrid<P>(scene, accel.uniform_grid, ray);
		case AccelType::kTwoLevelGrid:
			return IsOccludedTwoLevelGrid<P>(scene, accel.two_level_grid, ray);
		case AccelType::kIrregularGrid:
			return IsOccludedIrregularGrid<P>(scene, accel.irregular_grid, ray);
		case AccelType::kBruteForce:
		default:
			return IsOccludedBruteForce<P>(scene, ray);
		}
	}

} // namespace happyray
