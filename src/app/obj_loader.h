// Minimal custom Wavefront OBJ parser (no third-party dependency, by design).
//
// Supported: v / vn / vt, triangle and polygon (fan-triangulated) `f` lines with v, v/vt,
// v//vn and v/vt/vn vertex references (including negative/relative indices), `usemtl` to tag
// triangles with a material name, and `mtllib`-referenced .mtl files for flat Lambertian
// diffuse colors (`Kd r g b`, resolved relative to the .obj file's own directory). All faces
// are merged into a single Mesh.
//
// Known limitations (left for later phases, as more material/multi-object support becomes
// needed):
//   - Only `Kd` (diffuse color) is read from .mtl files; everything else (Ka, Ks, Ns, map_Kd,
//     illum, ...) is ignored.
//   - `o` / `g` lines do not split the result into separate Mesh objects.
//   - Vertex normals are computed (smooth, area-weighted) only if the file supplies none at
//     all; a file that supplies normals for some vertices but not others is not specially
//     handled beyond what the `vn` indices request.
#pragma once

#include <string>

#include "mesh.h"

namespace happyray {

// Parses the OBJ file at `path` and appends a single Mesh (plus any `usemtl`/`mtllib`-derived
// materials) to `scene`. Returns false (and logs to stderr) on failure to open/read the file.
// After normal generation, bisects longest triangle edges until none exceeds 0.1 times
// the scene bounding-box diagonal, interpolating vertex attributes and preserving materials.
// Subdivision failures throw rather than returning partially processed geometry as success.
bool LoadObj(const std::string& path, Scene* scene);

}  // namespace happyray
