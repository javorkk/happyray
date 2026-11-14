// Render engine behind a plain-C++ interface; the Thrust backend (CUDA or TBB) is chosen at
// build time.
#pragma once

#include <memory>
#include <vector>

#include "../accel/sah_cost.h"
#include "../accel/intersect_common.h"
#include "../app/scene.h"
#include "../core/vec.h"
#include "ao_integrator.h"
#include "camera.h"
#include "collision_detector.h"
#include "simple_integrator.h"

namespace happyray
{

	// Target cells per primitive for each grid level (see ComputeGridResolution); a uniform grid
	// uses `top_level` as its only density.
	struct GridDensities
	{
		float top_level	   = 0;
		float second_level = 0; // Unused for a uniform grid.
	};

	struct GridBuildSettings
	{
		// Zero selects the renderer's default density for this grid type.
		GridDensities densities;
		uint32_t	  dense_cell_threshold = 16;
		float		  merge_alpha		   = 0.7f; // Irregular grid only.
		int			  expand_iterations	   = 2;	   // Irregular grid only.
	};

	class Renderer
	{
	  public:
		// Copies `scene` to backend memory.
		explicit Renderer(const FlatScene &scene);
		~Renderer();
		Renderer(const Renderer &)			  = delete;
		Renderer &operator=(const Renderer &) = delete;

		// Also regenerates the per-pixel jitter table (see random.h).
		void ResizeFramebuffer(int width, int height);

		// Invalidates stale grids and rebuilds the active one.
		void UpdateScene(const FlatScene &scene);

		// Keeps the pixel jitter table.
		void ResetAccumulation();

		uint32_t SampleCount() const;

		// No-op for brute force. Settings are kept per grid for scene updates and density tuning.
		void BuildAccelerationStructure(GridBuildSettings settings = {});
		bool HasAccelerationStructure() const;

		// Refreshes collisions if the grid is built; does not build it or reset accumulation.
		void	  SetAccelType(AccelType type);
		AccelType GetAccelType() const;

		GridDensities CurrentGridDensities() const;
		void RebuildGridDensities(GridDensities densities);

		// SAH cost of the active, built grid (see sah_cost.h).
		float ComputeCurrentGridSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const;

		// Reset accumulation after changing the camera or integrator configuration.
		void RenderFrame(const Camera &camera, const AOIntegrator &integrator);
		void RenderFrame(const Camera &camera, const SimpleIntegrator &integrator);

		// The first frame after reset uses the simple integrator; later frames blend the AO average
		// over it with weight n/(n+32), n = AO frame count.
		void RenderProgressiveFrame(const Camera &camera, const AOIntegrator &ao, const SimpleIntegrator &simple);

		const std::vector<Vec3f> &ResolvedPixels() const;

		// Each grid build detects collisions between objects over its leaf cells; colliding
		// primitives render red. Returns the smallest other object id per primitive, or kNoCollision.
		std::vector<uint32_t> CollisionIds() const;
		const CollisionStats &LastCollisionStats() const;

		// While disabled (e.g. paused animation), builds keep the last results. UpdateScene still
		// clears them. Enabled by default.
		void SetCollisionDetectionEnabled(bool enabled);
		bool CollisionDetectionEnabled() const;

		// Ignores CollisionDetectionEnabled(); no-op for brute force or an unbuilt grid.
		void DetectCollisions();
		void ClearCollisions();

		// Changing it rebuilds the active grid.
		void			SetCellOverlapMode(CellOverlapMode mode);
		CellOverlapMode GetCellOverlapMode() const;

		// Point/curve radius in pixels at each primitive's distance from the camera passed to
		// UpdateViewDependentGeometry; <= 0 keeps the scene's radii. Default: 2.
		void  SetScreenSpaceRadius(float pixels);
		float ScreenSpaceRadius() const;

		// Recomputes point/curve radii for `camera` and the current framebuffer height, then
		// rebuilds the active grid. Returns false (no-op) for other primitive types.
		bool UpdateViewDependentGeometry(const Camera &camera);

	  private:
		void RebuildGrids();

		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

	// "CUDA" or "TBB".
	const char *RendererBackendName();

} // namespace happyray
