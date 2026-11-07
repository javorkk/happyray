// Generic fixed-size vertex bundle, adopted from javorkk/string2shape's Primitive<N> concept.
#pragma once

#include "vec.h"

namespace happyray {

template <int kNumVerticesT>
struct Primitive {
    static constexpr int kNumVertices = kNumVerticesT;
    Vec3f vtx[kNumVerticesT];
};

}  // namespace happyray
