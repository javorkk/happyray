// Minimal Wavefront OBJ parser. Reads v/vn/vt, polygon faces (fan-triangulated, negative indices
// allowed), usemtl, and the Kd color from mtllib files. All faces go into one Mesh; each distinct
// (`o`, `g`) name pair becomes an object id for collision detection.
#pragma once

#include <string>

#include "../core/mesh.h"

namespace happyray
{

	// Generates smooth normals if the file has none, then bisects edges longer than 0.1x the scene
	// diagonal. Returns false on I/O failure.
	bool LoadObj(const std::string &path, Scene *scene);

} // namespace happyray
