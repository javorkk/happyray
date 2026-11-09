// Scene views, host-owned flattened geometry, subdivision, and camera/AO helpers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../render/camera.h"
#include "mesh.h"
#include "primitive.h"
#include "triangle.h"
#include "vec.h"

namespace happyray
{

	// Per-primitive collision id meaning "this primitive does not collide with any other object".
	constexpr uint32_t kNoCollision = 0xffffffffu;

	// Non-owning view over either host or device memory. Holds one primitive type: an indexed
	// triangle soup, or one of the point/sphere/curve arrays.
	struct SceneView
	{
		const Vec3f	   *positions	 = nullptr;
		const Vec3f	   *normals		 = nullptr;
		const uint32_t *indices		 = nullptr; // 3 per triangle, indexing positions/normals.
		const uint32_t *material_ids = nullptr; // 1 per primitive.
		// FlattenScene supplies a default-white entry for scenes with no materials.
		const Vec3f	   *material_diffuse_colors = nullptr;
		uint32_t		primitive_count			= 0;
		const uint32_t *object_ids				= nullptr; // 1 per primitive; may be null.
		// 1 per primitive (the smallest id of another object it intersects, or kNoCollision); may be
		// null. Colliding primitives are shaded with `collided_material_id` instead of their own.
		const uint32_t	  *collision_ids		= nullptr;
		uint32_t		   collided_material_id = 0;
		PrimitiveType	   primitive_type		= PrimitiveType::kTriangle;
		const Point		  *points				= nullptr;
		const Sphere	  *spheres				= nullptr;
		const BezierCurve *curves				= nullptr;
	};

	HAPPYRAY_HD inline Triangle FetchTriangle(const SceneView &scene, uint32_t tri)
	{
		Triangle t;
		t.vtx[0] = scene.positions[scene.indices[tri * 3 + 0]];
		t.vtx[1] = scene.positions[scene.indices[tri * 3 + 1]];
		t.vtx[2] = scene.positions[scene.indices[tri * 3 + 2]];
		return t;
	}

	// The material a hit on `prim` is shaded with: swapped for the collided material if collision
	// detection found the primitive intersecting another object.
	HAPPYRAY_HD inline uint32_t FetchMaterialId(const SceneView &scene, uint32_t prim)
	{
		if (scene.collision_ids && scene.collision_ids[prim] != kNoCollision)
		{
			return scene.collided_material_id;
		}
		return scene.material_ids[prim];
	}

	// Host-owned geometry shared by CPU rendering and GPU upload.
	struct FlatScene
	{
		PrimitiveType		  primitive_type = PrimitiveType::kTriangle;
		std::vector<Vec3f>	  positions;
		std::vector<Vec3f>	  normals;
		std::vector<uint32_t> indices;
		std::vector<Point>	  points;
		std::vector<Sphere>	  spheres;
		std::vector<BezierCurve> curves;
		std::vector<uint32_t>	 material_ids; // 1 per primitive.
		std::vector<Vec3f>		 material_diffuse_colors;
		// 1 per primitive, unique across meshes; empty means a single object.
		std::vector<uint32_t> object_ids;

		uint32_t PrimitiveCount() const
		{
			switch (primitive_type)
			{
			case PrimitiveType::kPoint:
				return static_cast<uint32_t>(points.size());
			case PrimitiveType::kSphere:
				return static_cast<uint32_t>(spheres.size());
			case PrimitiveType::kBezierCurve:
				return static_cast<uint32_t>(curves.size());
			default:
				return static_cast<uint32_t>(indices.size() / 3);
			}
		}

		SceneView View() const
		{
			SceneView view;
			view.positions				 = positions.data();
			view.normals				 = normals.data();
			view.indices				 = indices.data();
			view.material_ids			 = material_ids.data();
			view.material_diffuse_colors = material_diffuse_colors.data();
			view.object_ids				 = object_ids.empty() ? nullptr : object_ids.data();
			view.primitive_count		 = PrimitiveCount();
			view.primitive_type			 = primitive_type;
			view.points					 = points.data();
			view.spheres				 = spheres.data();
			view.curves					 = curves.data();
			return view;
		}
	};

	// Concatenates meshes, offsetting indices by the running vertex count and object ids by the
	// running object count (a mesh without object ids counts as one object). Non-triangle
	// primitives form a single object.
	inline FlatScene FlattenScene(const Scene &scene)
	{
		FlatScene flat;
		flat.primitive_type = scene.Type();
		if (flat.primitive_type != PrimitiveType::kTriangle)
		{
			if (!scene.meshes.empty() || (!scene.points.empty()) + (!scene.spheres.empty()) + (!scene.curves.empty()) > 1)
				throw std::invalid_argument("Scenes must hold a single primitive type");
			flat.points		  = scene.points;
			flat.spheres	  = scene.spheres;
			flat.curves		  = scene.curves;
			flat.material_ids = scene.primitive_material_ids;
			flat.material_ids.resize(flat.PrimitiveCount(), 0u);
			flat.object_ids.assign(flat.PrimitiveCount(), 0u);
		}
		uint32_t  vertex_offset = 0;
		uint32_t  object_offset = 0;
		for (const Mesh &mesh : scene.meshes)
		{
			flat.positions.insert(flat.positions.end(), mesh.positions.begin(), mesh.positions.end());
			for (const VertexAttributes &attributes : mesh.attributes)
			{
				flat.normals.push_back(attributes.normal);
			}
			for (uint32_t index : mesh.indices)
			{
				flat.indices.push_back(index + vertex_offset);
			}
			for (uint32_t material_id : mesh.material_ids)
			{
				flat.material_ids.push_back(material_id);
			}
			uint32_t mesh_object_count = 0;
			for (uint32_t tri = 0; tri < mesh.TriangleCount(); ++tri)
			{
				const uint32_t object_id = tri < mesh.object_ids.size() ? mesh.object_ids[tri] : 0;
				flat.object_ids.push_back(object_id + object_offset);
				mesh_object_count = std::max(mesh_object_count, object_id + 1);
			}
			object_offset += mesh_object_count;
			vertex_offset += static_cast<uint32_t>(mesh.positions.size());
		}

		for (const Material &material : scene.materials)
		{
			flat.material_diffuse_colors.push_back(material.diffuse_color);
		}
		if (flat.material_diffuse_colors.empty())
		{
			// Material-less OBJ faces still reference material id 0.
			flat.material_diffuse_colors.push_back(Vec3f(1.0f, 1.0f, 1.0f));
		}
		return flat;
	}

	inline double SceneEdgeLengthSquared(const Vec3f &a, const Vec3f &b)
	{
		const double x = static_cast<double>(a.x) - b.x;
		const double y = static_cast<double>(a.y) - b.y;
		const double z = static_cast<double>(a.z) - b.z;
		return x * x + y * y + z * z;
	}

	// Keep the original scene-wide limit fixed throughout longest-edge bisection.
	inline void SubdivideLargeTriangles(Scene *scene)
	{
		for (const Mesh &mesh : scene->meshes)
		{
			for (const Vec3f &p : mesh.positions)
			{
				if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
					throw std::runtime_error("Triangle subdivision requires finite vertex positions");
			}
		}
		const BBox bounds = scene->WorldBounds();
		if (bounds.IsEmpty())
			return;
		const double limit_squared = 0.01 * SceneEdgeLengthSquared(bounds.min_bound, bounds.max_bound);
		if (!std::isfinite(limit_squared))
			throw std::runtime_error("Triangle subdivision requires finite scene bounds");
		if (limit_squared == 0)
			return;

		for (Mesh &mesh : scene->meshes)
		{
			for (size_t tri = 0; tri < mesh.indices.size() / 3;)
			{
				const uint32_t indices[3] = {mesh.indices[tri * 3],
											 mesh.indices[tri * 3 + 1],
											 mesh.indices[tri * 3 + 2]};
				int			   edge		  = 0;
				double		   longest	  = 0;
				for (int i = 0; i < 3; ++i)
				{
					const double length =
						SceneEdgeLengthSquared(mesh.positions[indices[i]], mesh.positions[indices[(i + 1) % 3]]);
					if (length > longest)
					{
						longest = length;
						edge	= i;
					}
				}
				if (longest <= limit_squared)
				{
					++tri;
					continue;
				}

				const uint32_t			a = indices[edge], b = indices[(edge + 1) % 3];
				const uint32_t			c				  = indices[(edge + 2) % 3];
				const Vec3f				midpoint_position = mesh.positions[a] * 0.5f + mesh.positions[b] * 0.5f;
				const VertexAttributes &va				  = mesh.attributes[a];
				const VertexAttributes &vb				  = mesh.attributes[b];
				VertexAttributes		midpoint;
				midpoint.normal = va.normal * 0.5f + vb.normal * 0.5f;
				midpoint.uv		= va.uv * 0.5f + vb.uv * 0.5f;
				if (SceneEdgeLengthSquared(midpoint_position, mesh.positions[a]) == 0 ||
					SceneEdgeLengthSquared(midpoint_position, mesh.positions[b]) == 0)
					throw std::runtime_error("Triangle subdivision cannot represent an edge midpoint");
				if (mesh.positions.size() >= std::numeric_limits<uint32_t>::max() ||
					mesh.indices.size() / 3 >= std::numeric_limits<uint32_t>::max())
					throw std::overflow_error("Triangle subdivision exceeds 32-bit mesh limits");
				const uint32_t mid = static_cast<uint32_t>(mesh.positions.size());
				mesh.positions.push_back(midpoint_position);
				mesh.attributes.push_back(midpoint);
				mesh.indices[tri * 3]	  = a;
				mesh.indices[tri * 3 + 1] = mid;
				mesh.indices[tri * 3 + 2] = c;
				mesh.indices.insert(mesh.indices.end(), {mid, b, c});
				if (!mesh.material_ids.empty())
				{
					const uint32_t material = mesh.material_ids[tri];
					mesh.material_ids.push_back(material);
				}
				if (!mesh.object_ids.empty())
				{
					const uint32_t object = mesh.object_ids[tri];
					mesh.object_ids.push_back(object);
				}
			}
		}
	}

	// Frames the camera on the scene's bounding box so freshly-loaded OBJ files are visible
	// without manual positioning.
	inline Camera MakeDefaultCamera(const Scene &scene)
	{
		Camera	   camera;
		const BBox bounds = scene.WorldBounds();
		if (bounds.IsEmpty())
		{
			return camera;
		}
		const Vec3f center = bounds.Center();
		const Vec3f extent = bounds.Extent();
		const float radius = 0.5f * Length(extent);
		camera.position	   = center + Vec3f(0.0f, radius * 0.4f, radius * 2.2f + 1e-3f);
		camera.yaw_deg	   = -90.0f;
		camera.pitch_deg   = -10.0f;
		return camera;
	}

	// The default ambient-occlusion occlusion-ray reach for a scene of a given size: 0.1x the
	// longest side of its world-space bounding box. See AOIntegrator::max_distance
	// (src/render/ao_integrator.h) for where this gets applied.
	inline float DefaultMaxDistance(const Scene &scene)
	{
		const BBox bounds = scene.WorldBounds();
		if (bounds.IsEmpty())
		{
			return 1.0f;
		}
		const Vec3f extent		 = bounds.Extent();
		const float longest_side = std::fmax(extent.x, std::fmax(extent.y, extent.z));
		return longest_side * 0.1f;
	}

	// A scene.obj's optional sibling camera config, e.g. "assets/ogre.obj" ->
	// "assets/ogre.camera.cfg". Doesn't check the file exists -- LoadCameraConfig does that.
	inline std::string CameraConfigPathFor(const std::string &scene_path)
	{
		const std::string suffix = ".obj";
		if (scene_path.size() >= suffix.size() &&
			scene_path.compare(scene_path.size() - suffix.size(), suffix.size(), suffix) == 0)
		{
			return scene_path.substr(0, scene_path.size() - suffix.size()) + ".camera.cfg";
		}
		return scene_path + ".camera.cfg";
	}

	// Loads a happyray-style camera.cfg ("position x y z", "orientation x y z" (the forward/view
	// direction, not necessarily normalized), "up x y z", "FOV deg", "resX"/"resY" -- "rotation" is
	// accepted but ignored, since HappyRay has no initial-rotation-then-reset step). Returns false
	// (leaving `camera` untouched) if the file doesn't exist.
	//
	// HappyRay's Camera only supports yaw/pitch (no roll -- its "up" is always derived from yaw/pitch
	// applied to world-+Y, see camera.h), so only the forward direction's yaw/pitch is reproduced
	// exactly; an "up" vector that isn't already close to world +Y (i.e. a camera.cfg with
	// intentional roll/bank) can't be represented and is otherwise ignored.
	//
	// If `out_width`/`out_height` are non-null and the file has a "resX"/"resY" line, they're set to
	// those values (left untouched otherwise); the caller decides how to combine that with any other
	// resolution source (e.g. a command-line override should still win). Pass nullptr (the default)
	// to ignore resX/resY entirely, as benchmark.cpp does (its resolution comes from --width/--height
	// instead).
	inline bool
	LoadCameraConfig(const std::string &path, Camera *camera, int *out_width = nullptr, int *out_height = nullptr)
	{
		std::FILE *file = std::fopen(path.c_str(), "r");
		if (!file)
		{
			return false;
		}

		bool  have_orientation = false;
		Vec3f orientation(1.0f, 0.0f, 0.0f);
		char  line_buffer[256];
		while (std::fgets(line_buffer, sizeof(line_buffer), file))
		{
			std::istringstream line(line_buffer);
			std::string		   keyword;
			line >> keyword;
			if (keyword == "position")
			{
				line >> camera->position.x >> camera->position.y >> camera->position.z;
			}
			else if (keyword == "orientation")
			{
				line >> orientation.x >> orientation.y >> orientation.z;
				have_orientation = true;
			}
			else if (keyword == "FOV" || keyword == "fov")
			{
				line >> camera->vertical_fov_deg;
			}
			else if (keyword == "resX" && out_width)
			{
				line >> *out_width;
			}
			else if (keyword == "resY" && out_height)
			{
				line >> *out_height;
			}
			// "up"/"rotation" (and anything else) are intentionally ignored -- see this function's
			// header comment.
		}
		std::fclose(file);

		if (have_orientation)
		{
			const Vec3f		forward	  = Normalize(orientation);
			constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
			camera->pitch_deg		  = std::asin(forward.y) * kRadToDeg;
			camera->yaw_deg			  = std::atan2(forward.z, forward.x) * kRadToDeg;
		}
		return true;
	}

} // namespace happyray
