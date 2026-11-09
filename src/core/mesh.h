// In-memory scene representation produced by the OBJ loader (and, later, other importers like
// USD). Positions and vertex attributes live in separate, equally sized buffers sharing
// the same triangle indices.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bbox.h"
#include "bezier_curve.h"
#include "point.h"
#include "primitive.h"
#include "sphere.h"
#include "vec.h"

namespace happyray
{

	struct VertexAttributes
	{
		Vec3f normal;
		Vec2f uv;
	};

	struct Material
	{
		std::string name;
		// Flat Lambertian diffuse color, parsed from the OBJ's mtllib ("Kd r g b"). Defaults to
		// white (full albedo) when no material/Kd is available, so AO shading multiplied by this
		// color is a no-op for un-textured/material-less geometry.
		Vec3f diffuse_color = Vec3f(1.0f, 1.0f, 1.0f);
	};

	struct Mesh
	{
		std::string					  name;
		std::vector<Vec3f>			  positions;
		std::vector<VertexAttributes> attributes;	// One entry per position, including at seams.
		std::vector<uint32_t>		  indices;		// 3 per triangle, indexing positions and attributes.
		std::vector<uint32_t>		  material_ids; // One entry per triangle, indexing into Scene::materials.
		// One entry per triangle: the OBJ object (`o` statement) it belongs to, counted from 0 within
		// this mesh. Empty means the whole mesh is a single object.
		std::vector<uint32_t> object_ids;

		uint32_t TriangleCount() const
		{
			return static_cast<uint32_t>(indices.size() / 3);
		}
	};

	// Holds a single primitive type: either triangle meshes or one of the non-triangle arrays.
	struct Scene
	{
		std::vector<Mesh>		 meshes;
		std::vector<Material>	 materials;
		std::vector<Point>		 points;
		std::vector<Sphere>		 spheres;
		std::vector<BezierCurve> curves;
		// One entry per point/sphere/curve, indexing into `materials`.
		std::vector<uint32_t> primitive_material_ids;

		PrimitiveType Type() const
		{
			if (!points.empty())
				return PrimitiveType::kPoint;
			if (!spheres.empty())
				return PrimitiveType::kSphere;
			if (!curves.empty())
				return PrimitiveType::kBezierCurve;
			return PrimitiveType::kTriangle;
		}

		BBox WorldBounds() const
		{
			BBox bounds;
			for (const Mesh &mesh : meshes)
			{
				for (const Vec3f &position : mesh.positions)
				{
					bounds.Grow(position);
				}
			}
			for (const Point &point : points)
				bounds.Grow(BBoxExtractor<Point>::Get(point));
			for (const Sphere &sphere : spheres)
				bounds.Grow(BBoxExtractor<Sphere>::Get(sphere));
			for (const BezierCurve &curve : curves)
				bounds.Grow(BBoxExtractor<BezierCurve>::Get(curve));
			return bounds;
		}

		uint32_t PrimitiveCount() const
		{
			return static_cast<uint32_t>(points.size() + spheres.size() + curves.size()) + TriangleCount();
		}

		uint32_t TriangleCount() const
		{
			uint32_t total = 0;
			for (const Mesh &mesh : meshes)
			{
				total += mesh.TriangleCount();
			}
			return total;
		}
	};

} // namespace happyray
