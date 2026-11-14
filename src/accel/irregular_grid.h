// Irregular grid: a uniform top-level grid whose cells each own a dense power-of-2 local grid,
// refined by HAGrid's merge (combine neighbors while split-SAH cost doesn't rise) and expand
// (grow cells into neighbors whose references are a subset of their own) passes. Since all
// top-level cells have the same size, their power-of-2 resolutions share one integer lattice
// (`global_resolution`), so cells may merge/expand across top-level boundaries using exact
// integer bounds.
#pragma once

#include <cstdint>
#include <memory>

#include "../app/scene.h"
#include "../core/ray.h"
#include "../core/triangle.h"
#include "../core/vec.h"
#include "sah_cost.h"
#include "uniform_grid.h"

namespace happyray
{

	// `shift`/`global_offset` map local voxel v to shared-lattice coordinate
	// global_offset + (v << shift).
	struct IrregularCellInfo
	{
		Vec3f	 bounds_min;
		Vec3f	 cell_size; // Of one *finest* local voxel (pre-merge granularity).
		Vec3f	 inv_cell_size;
		Vec3i	 resolution		= Vec3i(1, 1, 1); // Already rounded up to a power of 2.
		uint32_t entries_offset = 0;
		Vec3i	 shift;
		Vec3i	 global_offset;
	};

	// One post-merge/expand cell; several finest voxels may resolve to it. Bounds are in the
	// shared lattice so merge/expand arithmetic stays exact.
	struct alignas(16) IrregularCell
	{
		Vec3i	  min;
		Vec3i	  max; // Exclusive, like a UniformGridView cell range.
		CellRange references;
	};

	// 8-byte traversal digest of IrregularCellInfo (cf. HAGrid's voxel map Entry). `packed` holds
	// 5-bit fields: shift.xyz, then log2(resolution.xy) (z isn't needed to flatten).
	struct alignas(8) IrregularTopCell
	{
		uint32_t entries_offset = 0;
		uint32_t packed			= 0;
	};

	HAPPYRAY_HD inline int IrregularLog2(int x)
	{
		int log = 0;
		while ((1 << log) < x)
		{
			++log;
		}
		return log;
	}

	HAPPYRAY_HD inline IrregularTopCell MakeIrregularTopCell(const IrregularCellInfo &info)
	{
		IrregularTopCell top;
		top.entries_offset = info.entries_offset;
		top.packed		   = static_cast<uint32_t>(info.shift.x) | (static_cast<uint32_t>(info.shift.y) << 5) |
					 (static_cast<uint32_t>(info.shift.z) << 10) |
					 (static_cast<uint32_t>(IrregularLog2(info.resolution.x)) << 15) |
					 (static_cast<uint32_t>(IrregularLog2(info.resolution.y)) << 20);
		return top;
	}

	struct IrregularGridView
	{
		UniformGridView			 top_level;
		const IrregularCellInfo *cell_infos = nullptr; // One per top-level cell (build-time data).
		const IrregularTopCell	*top_cells	= nullptr; // One per top-level cell (traversal lookup).
		// Indexed by info.entries_offset + local voxel; yields an absolute local_cells index, which
		// after merging may belong to a different top-level cell.
		const uint32_t		*local_entries = nullptr;
		const IrregularCell *local_cells   = nullptr; // Combined across every top-level cell.
		const uint32_t		*prim_indices  = nullptr; // Combined sorted leaf CSR payload.
		// global_resolution[axis] == resolution[axis] << shift[axis] for every top-level cell.
		Vec3f global_cell_size;
		Vec3i global_resolution = Vec3i(1, 1, 1);
		Vec3f global_inv_cell_size;
		Vec3i global_shift;
	};

	namespace detail
	{

		// Shared-lattice voxel -> owning cell: top-level cell from the high bits, local voxel from
		// the low bits (HAGrid's lookup_entry).
		HAPPYRAY_HD inline uint32_t LookupIrregularCell(const IrregularGridView &grid, const Vec3i &voxel)
		{
			const int			   tx = voxel.x >> grid.global_shift.x;
			const int			   ty = voxel.y >> grid.global_shift.y;
			const int			   tz = voxel.z >> grid.global_shift.z;
			const IrregularTopCell top =
				LoadRO(grid.top_cells +
					   FlattenCellIndex(tx, ty, tz, grid.top_level.resolution.x, grid.top_level.resolution.y));
			const uint32_t p	 = top.packed;
			const int	   lx	 = (voxel.x & (grid.global_resolution.x - 1)) >> (p & 31u);
			const int	   ly	 = (voxel.y & (grid.global_resolution.y - 1)) >> ((p >> 5) & 31u);
			const int	   lz	 = (voxel.z & (grid.global_resolution.z - 1)) >> ((p >> 10) & 31u);
			const int	   log_x = static_cast<int>((p >> 15) & 31u);
			const int	   log_y = static_cast<int>((p >> 20) & 31u);
			return LoadRO(grid.local_entries + top.entries_offset + static_cast<uint32_t>(lx) +
						  (static_cast<uint32_t>(ly) << log_x) + (static_cast<uint32_t>(lz) << (log_x + log_y)));
		}

		// The exit axis steps past the far plane; other axes are re-derived from the exit point but
		// clamped to the cell's extent and to never move backwards, so rounding can't skip or stall.
		HAPPYRAY_HD inline int NextIrregularVoxel(float t_cell,
												  float t_exit,
												  float voxel_offset,
												  float voxel_scale,
												  bool	positive,
												  int	voxel,
												  int	cell_min,
												  int	cell_max)
		{
			if (t_cell == t_exit)
			{
				return positive ? cell_max : cell_min - 1;
			}
			const int exit_voxel =
				ClampInt(static_cast<int>(voxel_offset + voxel_scale * t_exit), cell_min, cell_max - 1);
			return positive ? (exit_voxel > voxel ? exit_voxel : voxel) : (exit_voxel < voxel ? exit_voxel : voxel);
		}

		// Flat traversal over the shared lattice (HAGrid's traverse.cu). Returns true iff `visit` does.
		template <typename VisitFn>
		HAPPYRAY_HD inline bool TraverseGrid(const IrregularGridView &grid, const Ray &ray, VisitFn &&visit)
		{
			const UniformGridView &top	   = grid.top_level;
			const Vec3f			  &org	   = ray.origin;
			const Vec3f			  &dir	   = ray.direction;
			const Vec3f			   inv_dir = SafeRcp(dir);
			float				   t_start, t_end;
			if (!IntersectRayBoxRcp(org,
									dir,
									inv_dir,
									top.bounds_min,
									top.bounds_max,
									ray.t_min,
									ray.t_max,
									&t_start,
									&t_end))
			{
				return false;
			}
			const Vec3i	 dims(top.resolution.x << grid.global_shift.x,
							  top.resolution.y << grid.global_shift.y,
							  top.resolution.z << grid.global_shift.z);
			const Vec3f &gmin  = top.bounds_min;
			const Vec3f &gsize = grid.global_cell_size;
			const Vec3f &ginv  = grid.global_inv_cell_size;
			const Vec3f	 start = (org + dir * t_start - gmin) * ginv;
			Vec3i		 voxel(ClampInt(static_cast<int>(start.x), 0, dims.x - 1),
						   ClampInt(static_cast<int>(start.y), 0, dims.y - 1),
						   ClampInt(static_cast<int>(start.z), 0, dims.z - 1));

			const bool	pos_x = dir.x >= 0.0f, pos_y = dir.y >= 0.0f, pos_z = dir.z >= 0.0f;
			const float v_scale_x = dir.x * ginv.x, v_offset_x = (org.x - gmin.x) * ginv.x;
			const float v_scale_y = dir.y * ginv.y, v_offset_y = (org.y - gmin.y) * ginv.y;
			const float v_scale_z = dir.z * ginv.z, v_offset_z = (org.z - gmin.z) * ginv.z;

			float ray_t_max = ray.t_max;
			for (;;)
			{
				const IrregularCell cell = LoadRO(grid.local_cells + LookupIrregularCell(grid, voxel));

				const float tcx	   = dir.x != 0.0f
										 ? ((pos_x ? cell.max.x : cell.min.x) * gsize.x + gmin.x - org.x) * inv_dir.x
										 : INFINITY;
				const float tcy	   = dir.y != 0.0f
										 ? ((pos_y ? cell.max.y : cell.min.y) * gsize.y + gmin.y - org.y) * inv_dir.y
										 : INFINITY;
				const float tcz	   = dir.z != 0.0f
										 ? ((pos_z ? cell.max.z : cell.min.z) * gsize.z + gmin.z - org.z) * inv_dir.z
										 : INFINITY;
				const float t_exit = std::fmin(tcx, std::fmin(tcy, tcz));

				if (visit(cell.references, &ray_t_max))
				{
					return true;
				}
				if (ray_t_max <= t_exit)
				{
					return false;
				}

				voxel.x =
					NextIrregularVoxel(tcx, t_exit, v_offset_x, v_scale_x, pos_x, voxel.x, cell.min.x, cell.max.x);
				voxel.y =
					NextIrregularVoxel(tcy, t_exit, v_offset_y, v_scale_y, pos_y, voxel.y, cell.min.y, cell.max.y);
				voxel.z =
					NextIrregularVoxel(tcz, t_exit, v_offset_z, v_scale_z, pos_z, voxel.z, cell.min.z, cell.max.z);
				if (static_cast<unsigned>(voxel.x) >= static_cast<unsigned>(dims.x) ||
					static_cast<unsigned>(voxel.y) >= static_cast<unsigned>(dims.y) ||
					static_cast<unsigned>(voxel.z) >= static_cast<unsigned>(dims.z))
				{
					return false;
				}
			}
		}

	} // namespace detail

	class IrregularGrid
	{
	  public:
		// Densities are target cells per primitive (see ComputeGridResolution).
		static constexpr float kDefaultTopLevelDensity	  = 0.12f;
		static constexpr float kDefaultSecondLevelDensity = 2.4f;
		// `merge_alpha`: stop merging once a round shrinks a cell's local cell count by less than
		// (1 - merge_alpha). `expand_iterations`: each pass grows cells by at most one neighbor layer.
		explicit IrregularGrid(const SceneView &scene,
							   float			top_level_density	 = kDefaultTopLevelDensity,
							   float			second_level_density = kDefaultSecondLevelDensity,
							   uint32_t			dense_cell_threshold = 16,
							   float			merge_alpha			 = 0.7f,
							   int				expand_iterations	 = 2,
							   CellOverlapMode	mode				 = DefaultCellOverlapMode());
		~IrregularGrid();
		IrregularGrid(const IrregularGrid &)			= delete;
		IrregularGrid &operator=(const IrregularGrid &) = delete;

		IrregularGridView View() const;

		// Distinct post-merge cells (several local_entries may share one).
		uint32_t LeafCellCount() const;

		uint32_t TopLevelPrimIndexCount() const;
		uint32_t LocalEntryCount() const;
		uint32_t PrimIndexCount() const;

		float ComputeSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const;

	  private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

} // namespace happyray
