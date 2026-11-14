// Shared host/device scene intersection and occlusion queries, independent of the integrator:
// leaf-cell primitive tests and traversal visitors used by every grid, the per-grid and
// brute-force queries, and the AccelStructures dispatch. Brute force is the correctness oracle
// for every acceleration structure.
#pragma once

#include "../app/scene.h"
#include "../core/primitive.h"
#include "../core/primitive_traits.h"
#include "../core/ray.h"
#include "../core/triangle.h"
#include "../core/vec.h"
#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace happyray
{

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

	// Shared queries for all grid views; traversal overloads remain grid-specific.
	template <typename P = DynamicPrimitive, typename GridView>
	HAPPYRAY_HD inline bool
	IntersectSceneGrid(const SceneView &scene, const GridView &grid, const Ray &ray, HitRecord *out_hit)
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
			const uint32_t *prim_indices = [&]()
			{
				if constexpr (std::is_same_v<GridView, TwoLevelGridView>)
					return grid.second_level_prim_indices;
				else
					return grid.prim_indices;
			}();
			ClosestHit hit;
			detail::TraverseGrid(grid, ray, detail::ClosestHitVisitor<P>{scene, prim_indices, ray, hit});
			return ResolveClosestHit<P>(scene, ray, hit, out_hit);
		}
	}

	template <typename P = DynamicPrimitive, typename GridView>
	HAPPYRAY_HD inline bool IsOccludedGrid(const SceneView &scene, const GridView &grid, const Ray &ray)
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
			const uint32_t *prim_indices = [&]()
			{
				if constexpr (std::is_same_v<GridView, TwoLevelGridView>)
					return grid.second_level_prim_indices;
				else
					return grid.prim_indices;
			}();
			return detail::TraverseGrid(grid, ray, detail::OcclusionVisitor<P>{scene, prim_indices, ray});
		}
	}

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
			return IntersectSceneGrid<P>(scene, accel.two_level_grid, ray, out_hit);
		case AccelType::kIrregularGrid:
			return IntersectSceneGrid<P>(scene, accel.irregular_grid, ray, out_hit);
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
			return IsOccludedGrid<P>(scene, accel.two_level_grid, ray);
		case AccelType::kIrregularGrid:
			return IsOccludedGrid<P>(scene, accel.irregular_grid, ray);
		case AccelType::kBruteForce:
		default:
			return IsOccludedBruteForce<P>(scene, ray);
		}
	}

} // namespace happyray
