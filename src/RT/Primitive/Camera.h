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

#ifndef CAMERA_H_59133955_487F_44EB_A559_6E592D36A521
#define CAMERA_H_59133955_487F_44EB_A559_6E592D36A521

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

//A perspective camera implementation
class Camera
{
public:
	float3 mCenter;
	float3 mTopLeft;
	float3 mStepX, mStepY;

    HOST DEVICE void init(const float3 &aCenter, const float3 &aForward, const float3 &aUp, 
        float aVertOpeningAngInGrad, const uint aResX, const uint aResY)
    {
        mCenter = aCenter;

        float aspect_ratio = static_cast<float>(aResX) / static_cast<float>(aResY);

        float3 forward_axis = ~aForward;
        float3 right_axis = ~(forward_axis % aUp);
        float3 up_axis = ~(forward_axis % right_axis);

        float angleInRad = aVertOpeningAngInGrad * static_cast<float>(M_PI) / 180.f;
        float3 row_vector = 2.f * right_axis * tanf(angleInRad / 2.f) * aspect_ratio;
        float3 col_vector = 2.f * up_axis * tanf(angleInRad / 2.f);

        mStepX = row_vector / static_cast<float>(aResX);
        mStepY = col_vector / static_cast<float>(aResY);
        mTopLeft = forward_axis - row_vector / 2.f - col_vector / 2.f;

    }

    DEVICE float3 getPosition() const
    {
        return mCenter;
    }

	DEVICE float3 getDirection(float aX, float aY) const
    {
		return mTopLeft + aX * mStepX + aY * mStepY;
	}
};

//Pixel sampling strategies
template<int taNumSamplesX, int taNumSamplesY>
class RegularPixelSampler
{
public:
    float resX, resY;

    DEVICE void getPixelCoords(float aPixelId, float aSampleId,
        float& oPixelX, float& oPixelY) const
    {
        //compute in-pixel offsets
        float tmp =
            static_cast<float>(static_cast<int>(aSampleId) % (taNumSamplesX * taNumSamplesY))
            / (float) taNumSamplesX;
        //float tmp = aSampleId / (float) taNumSamplesX;
        oPixelY = (truncf(tmp) + 0.5f) / (float) taNumSamplesY;
        oPixelX = tmp - truncf(tmp) + 0.5f / (float) taNumSamplesX;

        //compute pixel coordinates (and sum up with the offsets)
        tmp = aPixelId / resX;
        oPixelY += truncf(tmp);
        oPixelX += (tmp - truncf(tmp)) * resX;
    }

};

class RandomPixelSampler
{
public:
    float resX, resY;

    //oPixelX and oPixelY have to be initialized with the random offsets
    DEVICE void getPixelCoords(float aPixelId, float& oPixelX, float& oPixelY) const
    {
        //compute pixel coordinates (and sum up with the offsets)
        float tmp = aPixelId / resX;
        oPixelY = truncf(tmp) + oPixelY;
        oPixelX = (tmp - truncf(tmp)) * resX + oPixelX;
    }

};

class GaussianPixelSampler
{
public:
    float resX, resY;

    //oPixelX and oPixelY have to be initialized with the random offsets
    DEVICE void getPixelCoords(float aPixelId, float& oPixelX, float& oPixelY) const
    {
        //compute pixel coordinates (and sum up with the offsets)
        float tmp = aPixelId / resX;
        float rad = 0.5f * sqrtf(-logf(1-oPixelX));
        float angle = 2.f * M_PI * oPixelY;
        oPixelY = truncf(tmp) + rad * sinf(angle);
        oPixelX = (tmp - truncf(tmp)) * resX + rad * cosf(angle);
    }

};

#endif // CAMERA_H_59133955_487F_44EB_A559_6E592D36A521
