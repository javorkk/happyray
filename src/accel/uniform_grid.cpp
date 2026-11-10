// Uniform grid ownership; construction is shared with the hierarchical coarse grid.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "grid_common.h"

namespace happyray
{
	struct UniformGrid::Impl
	{
		UniformGridView					 geometry;
		uint32_t						 prim_reference_count = 0;
		thrust::device_vector<CellRange> cell_ranges;
		thrust::device_vector<uint32_t>	 prim_indices;

		UniformGridView View() const
		{
			UniformGridView view = geometry;
			view.cell_ranges	 = thrust::raw_pointer_cast(cell_ranges.data());
			view.prim_indices	 = thrust::raw_pointer_cast(prim_indices.data());
			return view;
		}
	};

	CellOverlapMode DefaultCellOverlapMode()
	{
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
		return CellOverlapMode::kHybrid;
#else
		return CellOverlapMode::kExact;
#endif
	}

	const char *CellOverlapModeName(CellOverlapMode mode)
	{
		switch (mode)
		{
		case CellOverlapMode::kAabb:
			return "aabb";
		case CellOverlapMode::kHybrid:
			return "hybrid";
		default:
			return "exact";
		}
	}

	UniformGrid::UniformGrid(const SceneView &scene, float density, CellOverlapMode mode)
		: impl_(std::make_unique<Impl>())
	{
		auto built					= detail::BuildCellReferences(scene, density, mode);
		impl_->geometry				= built.geometry;
		impl_->prim_reference_count = built.reference_count;
		impl_->cell_ranges			= std::move(built.ranges);
		impl_->prim_indices			= std::move(built.prims);
	}

	UniformGrid::~UniformGrid() = default;

	UniformGridView UniformGrid::View() const
	{
		return impl_->View();
	}

	uint32_t UniformGrid::PrimIndexCount() const
	{
		return static_cast<uint32_t>(impl_->prim_indices.size());
	}

	float UniformGrid::ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const
	{
		const UniformGridView view	= View();
		const uint32_t		  cells = static_cast<uint32_t>(view.resolution.x) * view.resolution.y * view.resolution.z;
		return HalfSurfaceArea(view.cell_size) *
			   (static_cast<float>(cells) * costs.traversal +
				static_cast<float>(impl_->prim_reference_count) * costs.intersection) /
			   HalfSurfaceArea(scene_bounds.Extent());
	}
} // namespace happyray
