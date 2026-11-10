// Thrust implementation of TwoLevelGrid (see two_level_grid.h), following the legacy
// TLGridSortBuilder: build the top level directly, then build each nested grid from its owning
// cell's sorted primitive range. Only the nested grids retain explicit begin/end ranges.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "two_level_grid.h"
#include "grid_common.h"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace happyray
{
	namespace
	{

		// Computes each top-level cell's cubic second-level resolution.
		struct ComputeTopLevelCellFunctor
		{
			Vec3f			 top_bounds_min;
			Vec3f			 top_cell_size;
			int				 top_resolution_x;
			int				 top_resolution_y;
			const CellRange *top_ranges;
			float			 second_level_density;
			uint32_t		 dense_cell_threshold;

			HAPPYRAY_HD TopLevelCell operator()(uint32_t c) const
			{
				const int	   cz  = static_cast<int>(c) / (top_resolution_x * top_resolution_y);
				const int	   rem = static_cast<int>(c) % (top_resolution_x * top_resolution_y);
				const int	   cy  = rem / top_resolution_x;
				const int	   cx  = rem % top_resolution_x;
				const Vec3f	   cell_min(top_bounds_min.x + cx * top_cell_size.x,
										top_bounds_min.y + cy * top_cell_size.y,
										top_bounds_min.z + cz * top_cell_size.z);
				const Vec3f	   cell_max		   = cell_min + top_cell_size;
				const uint32_t primitive_count = top_ranges[c].end - top_ranges[c].start;

				const Vec3i computed_res =
					(primitive_count <= dense_cell_threshold)
						? Vec3i(1, 1, 1)
						: ComputeGridResolution(BBox(cell_min, cell_max), primitive_count, second_level_density);

				TopLevelCell info;
				const int	 resolution = computed_res.x > computed_res.y
											  ? (computed_res.x > computed_res.z ? computed_res.x : computed_res.z)
											  : (computed_res.y > computed_res.z ? computed_res.y : computed_res.z);
				info.resolution			= primitive_count == 0 ? -resolution : resolution;
				return info;
			}
		};

		struct ExtractNumCellsFunctor
		{
			HAPPYRAY_HD uint32_t operator()(const TopLevelCell &info) const
			{
				const uint32_t resolution = static_cast<uint32_t>(info.Resolution());
				return resolution * resolution * resolution;
			}
		};

		struct SetCellRangeOffsetFunctor
		{
			HAPPYRAY_HD TopLevelCell operator()(TopLevelCell info, uint32_t offset) const
			{
				info.cell_range_offset = offset;
				return info;
			}
		};

		template <typename P, CellOverlapMode Mode, bool Write> struct Fragments
		{
			SceneView			scene;
			const uint32_t	   *top_cell_keys;
			const uint32_t	   *top_prim_indices;
			const TopLevelCell *top_level_cells;
			Vec3f				top_bounds_min, top_cell_size, top_inv_cell_size;
			int					top_resolution_x, top_resolution_y;
			uint32_t			top_cell_count;
			const BBox		   *prim_bboxes;
			const uint32_t	   *inclusive_offsets  = nullptr;
			uint32_t		   *exact_counts	   = nullptr;
			uint32_t			invalid_cell_index = 0;
			uint32_t		   *out_cell_indices   = nullptr;
			uint32_t		   *out_prim_indices   = nullptr;

			HAPPYRAY_HD uint32_t operator()(uint32_t i) const
			{
				const uint32_t owner = top_cell_keys[i];
				if (owner >= top_cell_count)
				{
					if constexpr (Write)
						exact_counts[i] = 0;
					return 0;
				}
				const TopLevelCell &info = top_level_cells[owner];
				const int			cx	 = static_cast<int>(owner % top_resolution_x);
				const int			cy	 = static_cast<int>((owner / top_resolution_x) % top_resolution_y);
				const int			cz	 = static_cast<int>(owner / (top_resolution_x * top_resolution_y));
				const Vec3f			cell_min(top_bounds_min.x + cx * top_cell_size.x,
									 top_bounds_min.y + cy * top_cell_size.y,
									 top_bounds_min.z + cz * top_cell_size.z);
				const int			resolution = info.Resolution();
				const Vec3f			sub_inv(top_inv_cell_size.x * resolution,
									top_inv_cell_size.y * resolution,
									top_inv_cell_size.z * resolution);
				const Vec3f			sub_cell_size(top_cell_size.x / resolution,
										  top_cell_size.y / resolution,
										  top_cell_size.z / resolution);
				const uint32_t		tri = top_prim_indices[i];
				P					primitive;
				if constexpr (Mode != CellOverlapMode::kAabb)
					primitive = PrimitiveTraits<P>::Fetch(scene, tri);
				const BBox &box = prim_bboxes[tri];
				if constexpr (!Write)
				{
					return detail::CountOverlappedCells<P, Mode>(primitive,
																 box,
																 cell_min,
																 sub_inv,
																 sub_cell_size,
																 Vec3i(resolution, resolution, resolution));
				}
				else
				{
					const uint32_t write_begin = i == 0 ? 0u : inclusive_offsets[i - 1];
					const uint32_t write_end   = inclusive_offsets[i];
					uint32_t	   offset	   = write_begin;
					exact_counts[i]			   = detail::ForEachOverlappedCell<P, Mode>(
						   primitive,
						   box,
						   cell_min,
						   sub_inv,
						   sub_cell_size,
						   Vec3i(resolution, resolution, resolution),
						   [&](const Vec3i &cell)
						   {
							   out_cell_indices[offset] =
								   info.cell_range_offset +
								   FlattenCellIndex(cell, Vec3i(resolution, resolution, resolution));
							   out_prim_indices[offset] = tri;
							   ++offset;
						   });
					while (offset < write_end)
					{
						out_cell_indices[offset] = invalid_cell_index;
						out_prim_indices[offset] = tri;
						++offset;
					}
					return exact_counts[i];
				}
			}
		};

		struct ComputeTopLevelCellRefCountFunctor
		{
			const CellRange *top_ranges;
			const uint32_t	*pair_counts;

			HAPPYRAY_HD uint32_t operator()(uint32_t cell) const
			{
				uint32_t count = 0;
				for (uint32_t i = top_ranges[cell].start; i < top_ranges[cell].end; ++i)
					count += pair_counts[i];
				return count;
			}
		};

		struct TwoLevelGridSahFunctor
		{
			const TopLevelCell *top_level_cells;
			const uint32_t	   *top_level_prim_ref_counts;
			Vec3f				top_cell_size;
			SahCostConstants	costs;

			HAPPYRAY_HD float operator()(uint32_t t) const
			{
				const TopLevelCell &cell = top_level_cells[t];
				if (cell.IsEmpty())
					return 0.0f;
				const int	resolution = cell.Resolution();
				const auto	num_local  = static_cast<uint32_t>(resolution) * resolution * resolution;
				const Vec3f cell_size(top_cell_size.x / resolution,
									  top_cell_size.y / resolution,
									  top_cell_size.z / resolution);
				return HalfSurfaceArea(cell_size) *
					   (static_cast<float>(num_local) * costs.traversal +
						static_cast<float>(top_level_prim_ref_counts[t]) * costs.intersection);
			}
		};

	} // namespace

	struct TwoLevelGrid::Impl
	{
		GridBoundsView						top_level;
		thrust::device_vector<TopLevelCell> top_level_cells;
		thrust::device_vector<uint32_t>		top_level_prim_ref_counts;
		thrust::device_vector<CellRange>	second_level_cell_ranges;
		thrust::device_vector<uint32_t>		second_level_prim_indices;

		TwoLevelGridView View() const
		{
			TwoLevelGridView view;
			view.top_level				   = top_level;
			view.top_level_cells		   = thrust::raw_pointer_cast(top_level_cells.data());
			view.second_level_cell_ranges  = thrust::raw_pointer_cast(second_level_cell_ranges.data());
			view.second_level_prim_indices = thrust::raw_pointer_cast(second_level_prim_indices.data());
			return view;
		}

		template <typename P, CellOverlapMode Mode>
		void Build(const SceneView &scene,
				   float			top_level_density,
				   float			second_level_density,
				   uint32_t			dense_cell_threshold);
	};

	TwoLevelGrid::TwoLevelGrid(const SceneView &scene,
							   float			top_level_density,
							   float			second_level_density,
							   uint32_t			dense_cell_threshold,
							   CellOverlapMode	mode)
		: impl_(std::make_unique<Impl>())
	{
		detail::DispatchBuild(scene.primitive_type,
							  mode,
							  [&](auto prim, auto overlap)
							  {
								  impl_->Build<typename decltype(prim)::type, decltype(overlap)::value>(
									  scene,
									  top_level_density,
									  second_level_density,
									  dense_cell_threshold);
							  });
	}

	template <typename P, CellOverlapMode Mode>
	void TwoLevelGrid::Impl::Build(const SceneView &scene,
								   float			top_level_density,
								   float			second_level_density,
								   uint32_t			dense_cell_threshold)
	{
		auto		coarse			   = detail::BuildCellReferences<P, Mode>(scene, top_level_density);
		auto	   &prim_bboxes		   = coarse.boxes;
		auto	   &top_cell_keys	   = coarse.keys;
		auto	   &top_prim_indices   = coarse.prims;
		auto	   &top_cell_ranges	   = coarse.ranges;
		const auto &geometry		   = coarse.geometry;
		top_level					   = {geometry.bounds_min,
										  geometry.bounds_max,
										  geometry.cell_size,
										  geometry.inv_cell_size,
										  geometry.resolution};
		const uint32_t num_top_cells   = static_cast<uint32_t>(top_cell_ranges.size());
		const uint32_t total_top_pairs = static_cast<uint32_t>(top_cell_keys.size());

		const GridBoundsView				top_view = View().top_level;
		thrust::device_vector<TopLevelCell> top_level_cells(num_top_cells);
		thrust::transform(thrust::counting_iterator<uint32_t>(0),
						  thrust::counting_iterator<uint32_t>(num_top_cells),
						  top_level_cells.begin(),
						  ComputeTopLevelCellFunctor{top_view.bounds_min,
													 top_view.cell_size,
													 top_view.resolution.x,
													 top_view.resolution.y,
													 thrust::raw_pointer_cast(top_cell_ranges.data()),
													 second_level_density,
													 dense_cell_threshold});

		thrust::device_vector<uint32_t> num_second_cells(num_top_cells);
		thrust::transform(top_level_cells.begin(),
						  top_level_cells.end(),
						  num_second_cells.begin(),
						  ExtractNumCellsFunctor());
		thrust::device_vector<uint32_t> second_cell_offsets(num_top_cells);
		thrust::exclusive_scan(num_second_cells.begin(), num_second_cells.end(), second_cell_offsets.begin());
		const uint32_t total_second_cells = second_cell_offsets.back() + num_second_cells.back();
		thrust::transform(top_level_cells.begin(),
						  top_level_cells.end(),
						  second_cell_offsets.begin(),
						  top_level_cells.begin(),
						  SetCellRangeOffsetFunctor());

		thrust::device_vector<uint32_t> second_pair_counts(total_top_pairs);
		thrust::transform(thrust::counting_iterator<uint32_t>(0),
						  thrust::counting_iterator<uint32_t>(total_top_pairs),
						  second_pair_counts.begin(),
						  Fragments<P, Mode, false>{scene,
													thrust::raw_pointer_cast(top_cell_keys.data()),
													thrust::raw_pointer_cast(top_prim_indices.data()),
													thrust::raw_pointer_cast(top_level_cells.data()),
													top_view.bounds_min,
													top_view.cell_size,
													top_view.inv_cell_size,
													top_view.resolution.x,
													top_view.resolution.y,
													num_top_cells,
													thrust::raw_pointer_cast(prim_bboxes.data())});
		thrust::inclusive_scan(second_pair_counts.begin(), second_pair_counts.end(), second_pair_counts.begin());
		const uint32_t					total_second_pairs = total_top_pairs > 0 ? second_pair_counts.back() : 0u;
		thrust::device_vector<uint32_t> second_cell_keys(total_second_pairs);
		thrust::device_vector<uint32_t> second_prim_indices(total_second_pairs);
		thrust::device_vector<uint32_t> exact_second_pair_counts(total_top_pairs, 0u);
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(total_top_pairs),
						 Fragments<P, Mode, true>{scene,
												  thrust::raw_pointer_cast(top_cell_keys.data()),
												  thrust::raw_pointer_cast(top_prim_indices.data()),
												  thrust::raw_pointer_cast(top_level_cells.data()),
												  top_view.bounds_min,
												  top_view.cell_size,
												  top_view.inv_cell_size,
												  top_view.resolution.x,
												  top_view.resolution.y,
												  num_top_cells,
												  thrust::raw_pointer_cast(prim_bboxes.data()),
												  thrust::raw_pointer_cast(second_pair_counts.data()),
												  thrust::raw_pointer_cast(exact_second_pair_counts.data()),
												  total_second_cells,
												  thrust::raw_pointer_cast(second_cell_keys.data()),
												  thrust::raw_pointer_cast(second_prim_indices.data())});
		thrust::device_vector<uint32_t> top_level_prim_ref_counts(num_top_cells);
		thrust::transform(
			thrust::counting_iterator<uint32_t>(0),
			thrust::counting_iterator<uint32_t>(num_top_cells),
			top_level_prim_ref_counts.begin(),
			ComputeTopLevelCellRefCountFunctor{thrust::raw_pointer_cast(top_cell_ranges.data()),
											   thrust::raw_pointer_cast(exact_second_pair_counts.data())});
		thrust::sort_by_key(second_cell_keys.begin(), second_cell_keys.end(), second_prim_indices.begin());
		uint32_t valid_second_pairs = total_second_pairs;
		if constexpr (Mode == CellOverlapMode::kHybrid)
			valid_second_pairs = detail::TrimInvalidPairs(&second_cell_keys, &second_prim_indices, total_second_cells);
		second_level_cell_ranges.assign(total_second_cells, CellRange{0u, 0u});
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(valid_second_pairs),
						 detail::CellRangeExtractor{thrust::raw_pointer_cast(second_cell_keys.data()),
													thrust::raw_pointer_cast(second_level_cell_ranges.data()),
													valid_second_pairs,
													total_second_cells});

		this->top_level_cells			= std::move(top_level_cells);
		this->top_level_prim_ref_counts = std::move(top_level_prim_ref_counts);
		second_level_prim_indices		= std::move(second_prim_indices);
	}

	TwoLevelGrid::~TwoLevelGrid() = default;

	uint32_t TwoLevelGrid::LeafCellCount() const
	{
		return static_cast<uint32_t>(impl_->second_level_cell_ranges.size());
	}

	uint32_t TwoLevelGrid::SecondLevelPrimIndexCount() const
	{
		return static_cast<uint32_t>(impl_->second_level_prim_indices.size());
	}

	float TwoLevelGrid::ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const
	{
		const TwoLevelGridView view = impl_->View();
		const auto num_top_cells	= static_cast<uint32_t>(view.top_level.resolution.x) * view.top_level.resolution.y *
								   view.top_level.resolution.z;
		float total = HalfSurfaceArea(view.top_level.cell_size) * static_cast<float>(num_top_cells) * costs.traversal;
		if (num_top_cells > 0)
		{
			total += thrust::transform_reduce(
				thrust::counting_iterator<uint32_t>(0),
				thrust::counting_iterator<uint32_t>(num_top_cells),
				TwoLevelGridSahFunctor{thrust::raw_pointer_cast(impl_->top_level_cells.data()),
									   thrust::raw_pointer_cast(impl_->top_level_prim_ref_counts.data()),
									   view.top_level.cell_size,
									   costs},
				0.0f,
				thrust::plus<float>());
		}
		return total / HalfSurfaceArea(scene_bounds.Extent());
	}

	TwoLevelGridView TwoLevelGrid::View() const
	{
		return impl_->View();
	}

} // namespace happyray
