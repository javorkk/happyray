// Backend-private sort-based construction shared by uniform and coarse grids.
#pragma once

#include "uniform_grid.h"

#include <type_traits>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "../core/thrust_scratch.h"

namespace happyray
{
	namespace detail
	{
		template <CellOverlapMode Mode> using OverlapModeTag = std::integral_constant<CellOverlapMode, Mode>;

		// Calls `fn(OverlapModeTag<mode>{})`.
		template <typename Fn> decltype(auto) DispatchOverlapMode(CellOverlapMode mode, Fn &&fn)
		{
			switch (mode)
			{
			case CellOverlapMode::kAabb:
				return fn(OverlapModeTag<CellOverlapMode::kAabb>{});
			case CellOverlapMode::kHybrid:
				return fn(OverlapModeTag<CellOverlapMode::kHybrid>{});
			default:
				return fn(OverlapModeTag<CellOverlapMode::kExact>{});
			}
		}

		// Calls `fn(PrimitiveTag<P>{}, OverlapModeTag<mode>{})`.
		template <typename Fn> decltype(auto) DispatchBuild(PrimitiveType type, CellOverlapMode mode, Fn &&fn)
		{
			return DispatchPrimitive(
				type,
				[&](auto prim) { return DispatchOverlapMode(mode, [&](auto overlap) { return fn(prim, overlap); }); });
		}

		template <typename P> struct PrimitiveBounds
		{
			SceneView		 scene;
			HAPPYRAY_HD BBox operator()(uint32_t prim) const
			{
				return ComputePrimitiveBBox<P>(scene, prim);
			}
		};

		struct BoundsUnion
		{
			HAPPYRAY_HD BBox operator()(BBox a, const BBox &b) const
			{
				a.Grow(b);
				return a;
			}
		};

		// The counting pass reserves capacity; the writing pass emits fragments and pads unused
		// slots (kHybrid only) with invalid keys.
		template <typename P, CellOverlapMode Mode, bool Write> struct CellFragments
		{
			SceneView		scene;
			UniformGridView grid;
			const BBox	   *boxes;
			const uint32_t *offsets = nullptr;
			const uint32_t *counts	= nullptr;
			uint32_t	   *keys	= nullptr;
			uint32_t	   *prims	= nullptr;

			HAPPYRAY_HD uint32_t operator()(uint32_t prim) const
			{
				P primitive;
				if constexpr (Mode != CellOverlapMode::kAabb)
					primitive = PrimitiveTraits<P>::Fetch(scene, prim);
				if constexpr (!Write)
				{
					return CountOverlappedCells<P, Mode>(primitive,
														 boxes[prim],
														 grid.bounds_min,
														 grid.inv_cell_size,
														 grid.cell_size,
														 grid.resolution);
				}
				else
				{
					uint32_t	   offset = offsets[prim];
					const uint32_t end	  = offset + counts[prim];
					const uint32_t count =
						ForEachOverlappedCell<P, Mode>(primitive,
													   boxes[prim],
													   grid.bounds_min,
													   grid.inv_cell_size,
													   grid.cell_size,
													   grid.resolution,
													   [&](const Vec3i &cell)
													   {
														   keys[offset]	   = FlattenCellIndex(cell, grid.resolution);
														   prims[offset++] = prim;
													   });
					const uint32_t invalid =
						static_cast<uint32_t>(grid.resolution.x) * grid.resolution.y * grid.resolution.z;
					while (offset < end)
					{
						keys[offset]	= invalid;
						prims[offset++] = prim;
					}
					return count;
				}
			}
		};

		struct BuiltCellReferences
		{
			UniformGridView					 geometry;
			thrust::device_vector<uint32_t>	 prims;
			thrust::device_vector<CellRange> ranges;
			uint32_t						 reference_count = 0;
		};

		struct CellReferenceScratch
		{
			CachingAllocator				allocator;
			thrust::device_vector<BBox>		boxes;
			thrust::device_vector<uint32_t> counts, offsets, keys;
		};

		// Drops the invalid-key padding the kHybrid writing pass sorted to the end. Only `values`
		// (kept by the grid) is reallocated; keys are build-time temporaries.
		template <typename Policy>
		inline uint32_t TrimInvalidPairs(const Policy					 &policy,
										 thrust::device_vector<uint32_t> *keys,
										 thrust::device_vector<uint32_t> *values,
										 uint32_t						  invalid_key)
		{
			const uint32_t valid = static_cast<uint32_t>(
				thrust::lower_bound(policy, keys->begin(), keys->end(), invalid_key) - keys->begin());
			if (valid < keys->size())
			{
				keys->resize(valid);
				values->resize(valid);
				values->shrink_to_fit();
			}
			return valid;
		}

		template <typename P, CellOverlapMode Mode>
		BuiltCellReferences BuildCellReferences(const SceneView &scene, float density, CellReferenceScratch &scratch)
		{
			const auto			policy = thrust::device(scratch.allocator);
			BuiltCellReferences built;
			const uint32_t		n	  = scene.primitive_count;
			const auto			first = thrust::counting_iterator<uint32_t>(0);
			Reserve(&scratch.boxes, n);
			thrust::transform(policy, first, first + n, scratch.boxes.begin(), PrimitiveBounds<P>{scene});
			BBox bounds =
				n > 0 ? thrust::reduce(policy, scratch.boxes.begin(), scratch.boxes.begin() + n, BBox(), BoundsUnion{})
					  : BBox();
			const float diagonal = Length(bounds.Extent());
			const float epsilon	 = diagonal > 0.0f ? diagonal * 1e-4f : 1e-4f;
			bounds.min_bound -= Vec3f(epsilon);
			bounds.max_bound += Vec3f(epsilon);
			auto &grid			 = built.geometry;
			grid.bounds_min		 = bounds.min_bound;
			grid.bounds_max		 = bounds.max_bound;
			grid.resolution		 = ComputeGridResolution(bounds, n, density);
			grid.cell_size		 = Vec3f(bounds.Extent().x / grid.resolution.x,
										 bounds.Extent().y / grid.resolution.y,
										 bounds.Extent().z / grid.resolution.z);
			grid.inv_cell_size	 = Vec3f(1.0f / grid.cell_size.x, 1.0f / grid.cell_size.y, 1.0f / grid.cell_size.z);
			const uint32_t cells = static_cast<uint32_t>(grid.resolution.x) * grid.resolution.y * grid.resolution.z;

			Reserve(&scratch.counts, n);
			Reserve(&scratch.offsets, n);
			thrust::device_vector<uint32_t> &counts	 = scratch.counts;
			thrust::device_vector<uint32_t> &offsets = scratch.offsets;
			const BBox						*boxes	 = thrust::raw_pointer_cast(scratch.boxes.data());
			thrust::transform(policy,
							  first,
							  first + n,
							  counts.begin(),
							  CellFragments<P, Mode, false>{scene, grid, boxes});
			thrust::exclusive_scan(policy, counts.begin(), counts.begin() + n, offsets.begin());
			const uint32_t pairs = n > 0 ? offsets[n - 1] + counts[n - 1] : 0u;
			Reserve(&scratch.keys, pairs);
			scratch.keys.resize(pairs);
			built.prims.resize(pairs);
			thrust::for_each(policy,
							 first,
							 first + n,
							 CellFragments<P, Mode, true>{scene,
														  grid,
														  boxes,
														  thrust::raw_pointer_cast(offsets.data()),
														  thrust::raw_pointer_cast(counts.data()),
														  thrust::raw_pointer_cast(scratch.keys.data()),
														  thrust::raw_pointer_cast(built.prims.data())});
			thrust::sort_by_key(policy, scratch.keys.begin(), scratch.keys.end(), built.prims.begin());
			built.reference_count = pairs;
			if constexpr (Mode == CellOverlapMode::kHybrid)
				built.reference_count = TrimInvalidPairs(policy, &scratch.keys, &built.prims, cells);
			built.ranges.assign(cells, CellRange{});
			thrust::for_each(policy,
							 first,
							 first + built.reference_count,
							 CellRangeExtractor{thrust::raw_pointer_cast(scratch.keys.data()),
												thrust::raw_pointer_cast(built.ranges.data()),
												built.reference_count,
												cells});
			return built;
		}

		inline BuiltCellReferences
		BuildCellReferences(const SceneView &scene, float density, CellOverlapMode mode, CellReferenceScratch &scratch)
		{
			return DispatchBuild(
				scene.primitive_type,
				mode,
				[&](auto prim, auto overlap)
				{
					return BuildCellReferences<typename decltype(prim)::type, decltype(overlap)::value>(scene,
																										density,
																										scratch);
				});
		}
	} // namespace detail
} // namespace happyray
