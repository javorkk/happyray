#ifdef _MSC_VER
#pragma once
#endif

#ifndef PRIMITIVE_HPP_INCLUDED_DEB13D93_5EFB_42A7_A8DC_97946CEBD3A5
#define PRIMITIVE_HPP_INCLUDED_DEB13D93_5EFB_42A7_A8DC_97946CEBD3A5

#include <vector_types.h>

template<int taNumVertices>
class Primitive
{
public:
    static const int NUM_VERTICES = taNumVertices;
    float3 vtx[taNumVertices];
};


#endif // PRIMITIVE_HPP_INCLUDED_DEB13D93_5EFB_42A7_A8DC_97946CEBD3A5
