// Collision detection between scene objects (SceneView::object_ids), using a built grid's leaf
// cells as the broad phase (port of string2shape's CollisionDetector): trim single-object cells,
// sort the remaining references by (cell, object), then test bounding boxes of primitive pairs
// from different objects, one work item per reference for load balance.
// collision_ids[prim] is the smallest id of another object `prim` collides with, or kNoCollision.
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
		// Pair tests executed (skipped when they couldn't lower a recorded id, so counts vary
		// slightly between runs) and how many found overlapping boxes.
		uint64_t candidate_pair_count = 0;
		uint64_t colliding_pair_count = 0;
		float	 total_ms			  = 0.0f;
	};

	// Keeps scratch buffers between calls. Not thread-safe.
	class CollisionDetector
	{
	  public:
		CollisionDetector();
		~CollisionDetector();
		CollisionDetector(CollisionDetector &&) noexcept;
		CollisionDetector &operator=(CollisionDetector &&) noexcept;
		CollisionDetector(const CollisionDetector &)			= delete;
		CollisionDetector &operator=(const CollisionDetector &) = delete;

		// `scene.object_ids` ids must lie in [0, object_count). Overwrites all of `collision_ids`.
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
