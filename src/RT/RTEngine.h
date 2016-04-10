/****************************************************************************/
/* Copyright (c) 2011, Javor Kalojanov
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
/****************************************************************************/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
#define RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5

#include "CUDAStdAfx.h"
#include "Application/WFObject.hpp"
#include "RT/Structure/FrameBuffer.h"
#include "RT/Primitive/LightSource.hpp"

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

    static void setLights(const AreaLightSourceCollection&);

    static void renderFrame(FrameBuffer& aFrameBuffer, const int aImageId, const int aRenderMode);

    static void cleanup();

    static float getBoundingBoxDiagonalLength();
};

#endif // RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
