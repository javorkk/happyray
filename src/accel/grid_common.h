// Backend-private sort-based construction shared by uniform and coarse grids.
#pragma once

#include "uniform_grid.h"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace happyray
{
	namespace detail
	{
		struct TriangleBounds
		{
			SceneView		 scene;
			HAPPYRAY_HD BBox operator()(uint32_t tri) const
			{
				return ComputeTriangleBBox(scene, tri);
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

		// Counting and writing use identical geometry; only writing filters fast
		// AABB capacities with SAT and pads unused slots.
		template <bool Write> struct CellFragments
		{
			SceneView		scene;
			UniformGridView grid;
			const BBox	   *boxes;
			const uint32_t *offsets	  = nullptr;
			uint32_t	   *counts	  = nullptr;
			uint32_t	   *keys	  = nullptr;
			uint32_t	   *triangles = nullptr;

			HAPPYRAY_HD uint32_t operator()(uint32_t tri) const
			{
				const Triangle triangle = FetchTriangle(scene, tri);
				if constexpr (!Write)
				{
					return CountTriangleCellIntersections(triangle,
														  BBoxExtractor<Triangle>::Get(triangle),
														  grid.bounds_min,
														  grid.inv_cell_size,
														  grid.cell_size,
														  grid.resolution);
				}
				else
				{
					uint32_t	   offset = offsets[tri];
					const uint32_t end	  = offset + counts[tri];
					counts[tri]			  = ForEachIntersectedCell(triangle,
														   boxes[tri],
														   grid.bounds_min,
														   grid.inv_cell_size,
														   grid.cell_size,
														   grid.resolution,
														   [&](const Vec3i &cell)
														   {
															   keys[offset] = FlattenCellIndex(cell, grid.resolution);
															   triangles[offset++] = tri;
														   });
					const uint32_t invalid =
						static_cast<uint32_t>(grid.resolution.x) * grid.resolution.y * grid.resolution.z;
					while (offset < end)
					{
						keys[offset]		= invalid;
						triangles[offset++] = tri;
					}
					return counts[tri];
				}
			}
		};

		struct BuiltCellReferences
		{
			UniformGridView					 geometry;
			thrust::device_vector<BBox>		 boxes;
			thrust::device_vector<uint32_t>	 keys;
			thrust::device_vector<uint32_t>	 triangles;
			thrust::device_vector<CellRange> ranges;
			uint32_t						 reference_count = 0;
		};

		inline BuiltCellReferences BuildCellReferences(const SceneView &scene, float density)
		{
			BuiltCellReferences built;
			const uint32_t		n	  = scene.triangle_count;
			const auto			first = thrust::counting_iterator<uint32_t>(0);
			built.boxes.resize(n);
			thrust::transform(first, first + n, built.boxes.begin(), TriangleBounds{scene});
			BBox bounds =
				n > 0 ? thrust::reduce(built.boxes.begin(), built.boxes.end(), BBox(), BoundsUnion{}) : BBox();
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

			thrust::device_vector<uint32_t> counts(n), offsets(n);
			const BBox					   *boxes = thrust::raw_pointer_cast(built.boxes.data());
			thrust::transform(first, first + n, counts.begin(), CellFragments<false>{scene, grid, boxes});
			thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin());
			const uint32_t pairs = n > 0 ? offsets.back() + counts.back() : 0u;
			built.keys.resize(pairs);
			built.triangles.resize(pairs);
			thrust::for_each(first,
							 first + n,
							 CellFragments<true>{scene,
												 grid,
												 boxes,
												 thrust::raw_pointer_cast(offsets.data()),
												 thrust::raw_pointer_cast(counts.data()),
												 thrust::raw_pointer_cast(built.keys.data()),
												 thrust::raw_pointer_cast(built.triangles.data())});
#if HAPPYRAY_FAST_CELL_COUNT
			built.reference_count = thrust::reduce(counts.begin(), counts.end(), 0u);
#else
			built.reference_count = pairs;
#endif
			thrust::sort_by_key(built.keys.begin(), built.keys.end(), built.triangles.begin());
			built.ranges.assign(cells, CellRange{});
			thrust::for_each(first,
							 first + pairs,
							 CellRangeExtractor{thrust::raw_pointer_cast(built.keys.data()),
												thrust::raw_pointer_cast(built.ranges.data()),
												pairs,
												cells});
			return built;
		}
	} // namespace detail
} // namespace happyray
