// Collision detection between the objects of a scene (OBJ `o` statements, see
// SceneView::object_ids), reusing the leaf cells of an already-built acceleration structure as
// the broad phase. Ported from string2shape's CollisionDetector
// (javorkk/string2shape, obj2string/CollisionDetector.cu):
//
//  1. CellTrimmer: discards every leaf cell whose primitives all belong to a single object.
//  2. The surviving cells are compacted (copy_if NonEmptyRange).
//  3. CollisionOperator: tests every pair of primitives from different objects inside the
//     remaining cells for bounding-box overlap (any primitive type), parallelized per cell
//     reference rather than per cell for load balance.
//
// Instead of string2shape's object/object adjacency matrix, the result is recorded per
// primitive: collision_ids[prim] is the smallest id of another object `prim` collides with, or
// kNoCollision (scene.h). Like Renderer and the grids, this is plain Thrust compiled for
// whichever backend (CUDA or TBB) the build selects; all pointers must be backend memory.
#pragma once

#include <cstdint>

#include "../accel/irregular_grid.h"
#include "../accel/two_level_grid.h"
#include "../accel/uniform_grid.h"
#include "../app/scene.h"

namespace happyray
{

	struct CollisionStats
	{
		uint32_t object_count = 0;
		// Leaf cells holding primitives of at least two different objects.
		uint32_t candidate_cell_count	   = 0;
		uint32_t colliding_primitive_count = 0;
		float	 total_ms				   = 0.0f;
	};

	class CollisionDetector
	{
	  public:
		// `scene.object_ids` must be non-null (unless object_count < 2), with ids in
		// [0, object_count). Overwrites all `scene.primitive_count` entries of
		// `collision_ids`. Scenes with fewer than two objects skip the detection.
		CollisionStats ComputeCollisions(const SceneView   &scene,
										 uint32_t			object_count,
										 const UniformGrid &grid,
										 uint32_t		   *collision_ids) const;
		CollisionStats ComputeCollisions(const SceneView	&scene,
										 uint32_t			 object_count,
										 const TwoLevelGrid &grid,
										 uint32_t			*collision_ids) const;
		CollisionStats ComputeCollisions(const SceneView	 &scene,
										 uint32_t			  object_count,
										 const IrregularGrid &grid,
										 uint32_t			 *collision_ids) const;
	};

} // namespace happyray
