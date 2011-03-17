#ifdef _MSC_VER
#pragma once
#endif

#ifndef TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035
#define TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035


#include "CUDAStdAfx.h"
#include "RT/Primitive/Primitive.hpp"

typedef Primitive<3> Triangle;

struct ShevtsovTriAccel
{
    float   nu;
    float   nv;
    float   np;
    float   pu;
    float   pv;
    float   e0u;
    float   e0v;
    float   e1u;
    float   e1v;
    int    dimW;
    int    dimU;
    int    dimV;

    ShevtsovTriAccel(){}

    ShevtsovTriAccel(Triangle aTriangle);
};

struct WoopTriAccel
{
    float data[12];

    WoopTriAccel(){}

    WoopTriAccel(const Triangle& aTriangle);
};

#endif // TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035
