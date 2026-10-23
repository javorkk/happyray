// Generic fixed-size vertex bundle, adopted from javorkk/string2shape's Primitive<N> concept:
// a primitive is nothing more than its vertex count and its corners, with no notion of what
// shape it represents. BBoxExtractor (bbox.h) is written purely in terms of this
// interface, so any new primitive shape gets bounding-box support for free, with no per-shape
// bbox code of its own.
#pragma once

#include "vec.h"

namespace happyray {

template <int kNumVerticesT>
struct Primitive {
    static constexpr int kNumVertices = kNumVerticesT;
    Vec3 vtx[kNumVerticesT];
};

// The only primitive HappyRay currently traces.
using Triangle = Primitive<3>;

}  // namespace happyray
