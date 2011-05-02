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

#ifndef FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC
#define FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

struct FrameBuffer
{
    uint resX, resY;
    float3* deviceData;

    FrameBuffer():resX(0),resY(0),deviceData(NULL)
    {}

    DEVICE float3& operator[](const uint aId)
    {
        return deviceData[aId];
    }

    DEVICE const float3& operator[](const uint aId) const
    {
        return deviceData[aId];
    }

    HOST void init(uint aResX, uint aResY)
    {
        if(aResX != resX || aResY != resY)
        {
            MY_CUDA_SAFE_CALL( cudaFree(deviceData) );
            resX = aResX;
            resY = aResY;
            MY_CUDA_SAFE_CALL( 
                cudaMalloc((void**) &deviceData, aResX * aResY * sizeof(float3)));
            MY_CUDA_SAFE_CALL( cudaMemset(deviceData, 0 , aResX * aResY * sizeof(float3)));
        }
    }

    HOST void download(float3* aTarget) const
    {
        MY_CUDA_SAFE_CALL(
            cudaMemcpy(
                aTarget,
                deviceData,
                resX * resY * sizeof(float3),
                cudaMemcpyDeviceToHost) );
    }

    HOST void download(float3* aTarget, const int aResX, const int aResY) const
    {
        MY_CUDA_SAFE_CALL(
            cudaMemcpy(
            aTarget,
            deviceData,
            aResX * aResY * sizeof(float3),
            cudaMemcpyDeviceToHost) );
    }

    HOST void setZero()
    {
        MY_CUDA_SAFE_CALL(
            cudaMemset(
                (void*) deviceData,
                0,
                resX * resY * sizeof(float3)));
    }

    HOST void cleanup()
    {
        MY_CUDA_SAFE_CALL( cudaFree(deviceData) );
    }
};

#endif // FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC
