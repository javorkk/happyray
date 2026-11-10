// Generic fixed-size vertex bundle, adopted from javorkk/string2shape's Primitive<N> concept.
#pragma once

#include <cstdint>

#include "vec.h"

namespace happyray
{

	enum class PrimitiveType : uint32_t
	{
		kTriangle,
		kPoint,
		kSphere,
		kBezierCurve,
		kBox
	};

	inline const char *PrimitiveTypeName(PrimitiveType type)
	{
		switch (type)
		{
		case PrimitiveType::kPoint:
			return "point";
		case PrimitiveType::kSphere:
			return "sphere";
		case PrimitiveType::kBezierCurve:
			return "curve";
		case PrimitiveType::kBox:
			return "box";
		default:
			return "triangle";
		}
	}

	template <int kNumVerticesT> struct Primitive
	{
		static constexpr int kNumVertices = kNumVerticesT;
		Vec3f				 vtx[kNumVerticesT];
	};

} // namespace happyray
