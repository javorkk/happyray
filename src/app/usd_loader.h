// OpenUSD loader for meshes (triangulated, normals recomputed, transforms baked, Z-up rotated to
// Y-up), points, and basis curves (converted to cubic Bezier segments). A stage must hold one
// primitive type. The primitive set is fixed by the first frame; later frames must keep point and
// segment counts. Frames are the union of geometry and xform time samples in range, keeping every
// `stride`-th. Requires -DHAPPYRAY_ENABLE_USD=ON.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "scene.h"

namespace happyray
{

	struct UsdLoadOptions
	{
		std::optional<double> start_time; // Inclusive, in stage time codes (default: first sample).
		std::optional<double> end_time;	  // Inclusive (default: last sample).
		uint32_t			  stride = 1; // Keep every `stride`-th time sample (0 is treated as 1).
	};

	bool IsUsdPath(const std::string &path);

	bool UsdSupported();

	// `first_scene` gets the first frame (for camera framing and statistics).
	bool LoadUsdFrames(const std::string	  &path,
					   const UsdLoadOptions	  &options,
					   std::vector<FlatScene> *frames,
					   Scene				  *first_scene);

} // namespace happyray
