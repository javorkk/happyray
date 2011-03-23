#ifdef _MSC_VER
#pragma once
#endif

#ifndef RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
#define RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5

#include "CUDAStdAfx.h"
#include "Application/WFObject.hpp"
#include "RT/Structure/FrameBuffer.h"

class RTEngine
{
public:

    static void init();
    
    static void upload(
        const WFObject& aFrame1,
        const WFObject& aFrame2,
        const float aCoeff);

    static void buildAccStruct();
    
    static void setCamera(
        const float3& aPosition,
        const float3& aOrientation,
        const float3& aUp,
        const float   aFOV,
        const int     aX,
        const int     aY );

    static void renderFrame(FrameBuffer& aFrameBuffer, const int aImageId);

    static void cleanup();
};

#endif // RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
