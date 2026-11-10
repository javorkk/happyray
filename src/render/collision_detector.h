// Collision detection between the objects of a scene (OBJ `o` statements, see
// SceneView::object_ids), reusing the leaf cells of an already-built acceleration structure as
// the broad phase. Ported from string2shape's CollisionDetector
// (javorkk/string2shape, obj2string/CollisionDetector.cu):
//
//  1. CellTrimmer: discards every leaf cell whose primitives all belong to a single object.
//  2. The surviving cells are compacted (copy_if NonEmptyRange).
//  3. The surviving cells' references are copied out and sorted by (cell, object id), so each
//     cell's references are grouped by object (the grid itself is left untouched).
//  4. CollisionOperator: tests every pair of primitives from different objects inside the
//     remaining cells for bounding-box overlap (any primitive type, including kBox, for which the
//     test is exact), parallelized per cell reference rather than per cell for load balance.
//     Each reference starts after its own object's run, so same-object pairs are never visited.
//     It counts the tests it executes.
//
// Instead of string2shape's object/object adjacency matrix, the result is recorded per
// primitive: collision_ids[prim] is the smallest id of another object `prim` collides with, or
// kNoCollision (scene.h). Like Renderer and the grids, this is plain Thrust compiled for
// whichever backend (CUDA or TBB) the build selects; all pointers must be backend memory.
#pragma once

#include <cstdint>
#include <memory>

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
		// Work done by the overlap tests: pairs of primitives from different objects actually
		// tested (a pair sharing k cells may be tested up to k times; tests that could not lower
		// either recorded id are skipped) and how many of those tests found overlapping boxes.
		// Skipping depends on thread scheduling, so both vary slightly between runs.
		uint64_t candidate_pair_count = 0;
		uint64_t colliding_pair_count = 0;
		float	 total_ms			  = 0.0f;
	};

	// Keeps its temporary buffers (and Thrust's own scratch memory) allocated between calls,
	// growing them as needed, so repeated detections (e.g. once per animation frame) don't pay
	// for allocations. Not thread-safe: use one detector per thread.
	class CollisionDetector
	{
	  public:
		CollisionDetector();
		~CollisionDetector();
		CollisionDetector(CollisionDetector &&) noexcept;
		CollisionDetector &operator=(CollisionDetector &&) noexcept;
		CollisionDetector(const CollisionDetector &)			= delete;
		CollisionDetector &operator=(const CollisionDetector &) = delete;

		// `scene.object_ids` must be non-null (unless object_count < 2), with ids in
		// [0, object_count). Overwrites all `scene.primitive_count` entries of
		// `collision_ids`. Scenes with fewer than two objects skip the detection.
		// Instantiated for UniformGrid, TwoLevelGrid, and IrregularGrid.
		template <typename Grid>
		CollisionStats ComputeCollisions(const SceneView &scene,
										 uint32_t		  object_count,
										 const Grid		 &grid,
										 uint32_t		 *collision_ids);

	  private:
		struct Scratch;
		std::unique_ptr<Scratch> scratch_;
	};

} // namespace happyray
