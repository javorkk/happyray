// OpenUSD (.usd/.usda/.usdc/.usdz) mesh, points, curves, and animation loader.
//
// Reads every visible UsdGeomMesh prim (default/render purpose, instance proxies included) at
// each requested time code and converts it to triangles: polygons are fan-triangulated, hole
// faces are dropped, winding follows the mesh's orientation and its world transform's handedness,
// and smooth area-weighted vertex normals are recomputed per frame (authored normals are
// ignored, since face-varying normals cannot be shared with the point buffers). World
// transforms are baked in, and Z-up stages are rotated to Y-up. Each source prim gets a Scene
// material carrying its average `displayColor` (light grey when absent).
// UsdGeomPoints are loaded as native points, and UsdGeomBasisCurves as native cubic Bezier
// segments. Linear, Bezier, B-spline, and Catmull-Rom bases are supported. Point/curve widths are
// converted from diameters to radii. A stage must contain only one primitive type because the
// renderer's Scene representation cannot mix triangles, points, and curves.
//
// The set of primitives is fixed by the first frame: visibility and purpose are evaluated there.
// Later frames must preserve mesh/point source point counts and the generated segment count for
// each curve source.
//
// Frames are the union of the authored time samples of loaded geometry and of every transformable
// prim's xform ops within [start_time, end_time], keeping every `stride`-th one.
// A stage with no time samples in that range yields a single frame at the default time.
//
// Needs OpenUSD at build time (-DHAPPYRAY_ENABLE_USD=ON); without it, UsdSupported() returns
// false and loading fails with an explanatory message on stderr.
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

	// True for the file extensions OpenUSD reads (.usd, .usda, .usdc, .usdz).
	bool IsUsdPath(const std::string &path);

	// True if this build was compiled with OpenUSD support.
	bool UsdSupported();

	// Loads the frames of the USD file at `path` into `frames` and the first frame into
	// `first_scene` (used for default camera framing and scene statistics). Returns false (and
	// logs to stderr) if the file cannot be opened, contains no meshes, or violates the
	// consistency rules above; `frames` is then left empty.
	bool LoadUsdFrames(const std::string	  &path,
					   const UsdLoadOptions	  &options,
					   std::vector<FlatScene> *frames,
					   Scene				  *first_scene);

} // namespace happyray
