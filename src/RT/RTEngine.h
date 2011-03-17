#ifdef _MSC_VER
#pragma once
#endif

#ifndef RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
#define RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5

#include "CUDAStdAfx.h"
#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UGridSortBuilder.h"

#include "Application/WFObject.hpp"

class StaticRTEngine
{
public:
    static float                                        sGridDensity;
    static int                                          sFrameId;
    static PrimitiveArray<Triangle>                     sTriangleArray;
    static UniformGridMemoryManager                     sUGridMemoryManager;
    static UGridSortBuilder<Triangle>                   sGridBuilder;

    static void init(const WFObject& aScene);
};

#endif // RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
