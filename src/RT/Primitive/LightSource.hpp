/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
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

#ifndef LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
#define LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "Utils/Scan.h"

struct AreaLightSource
{
    float3 position, normal, intensity, edge1, edge2;

    DEVICE HOST float3 AreaLightSource::getPoint(float aXCoord, float aYCoord) const
    {
        float3 result;
        result = position + aXCoord * edge1 + aYCoord * edge2;
        return result;
    }

    DEVICE HOST float AreaLightSource::getArea() const
    {
        return len((edge1 % edge2));
    }

    DEVICE HOST void AreaLightSource::init(
        const float3& aVtx0, const float3& aVtx1,
        const float3& aVtx2, const float3& aVtx3,
        const float3& aIntensity, const float3& aNormal)
    {
        position = aVtx0;
        normal = aNormal;
        edge1 = aVtx1 - aVtx0;
        edge2 = aVtx3 - aVtx0;
        intensity = aIntensity;
    }

};


class AreaLightSourceCollection
{
    size_t mSize;
    float* mCDF;
    float* mWeights;
    AreaLightSource* mLights;
public:

#ifndef __CUDA_ARCH__
    AreaLightSourceCollection()
        :mSize(0u), mCDF(NULL), mWeights(NULL), mLights(NULL)
    {}
#endif

    DEVICE AreaLightSource& getLight(float aRandNum)
    {
        for(int i=0; i < mSize - 1; ++i)
        {
            if(aRandNum < mCDF[i] / mCDF[mSize - 1])
                return mLights[i];
        }

        return mLights[mSize-1];
    }

    DEVICE AreaLightSource& getLightWithID(size_t aId)
    {
        if(aId >= mSize)
            return mLights[0];

        return mLights[aId];
    }

    DEVICE float getWeight(size_t aId)
    {
        if(aId >= mSize)
            return 0.f;
        return mWeights[aId];
    }

    DEVICE float getTotalWeight()
    {
        return mCDF[mSize - 1];
    }

    HOST DEVICE size_t size() const
    {
        return mSize;
    }

    HOST void cleanup()
    {
        mSize = 0;
        MY_CUDA_SAFE_CALL( cudaFree(mCDF) );
        MY_CUDA_SAFE_CALL( cudaFree(mWeights) );
        MY_CUDA_SAFE_CALL( cudaFree(mLights) );
    }

    HOST void upload(float aWeight, AreaLightSource& aLS)
    {
        if(mSize == 0u)
        {
            mSize = 1u;
            MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mCDF,     sizeof(float)) );
            MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mWeights, sizeof(float)) );
            MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mLights,  sizeof(AreaLightSource)) );
            MY_CUDA_SAFE_CALL( cudaMemcpy(mWeights, &aWeight, sizeof(float), cudaMemcpyHostToDevice) );
            MY_CUDA_SAFE_CALL( cudaMemcpy(mCDF, mWeights, sizeof(float), cudaMemcpyDeviceToDevice) );
            MY_CUDA_SAFE_CALL( cudaMemcpy(mLights, &aLS, sizeof(AreaLightSource), cudaMemcpyHostToDevice) );
            return;
        }

        ++mSize;
        float* oldCDF = mCDF;
        float* oldWeights = mWeights;
        AreaLightSource* oldLights = mLights;
        //allocate space
        MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mCDF,     sizeof(float)*mSize) );
        MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mWeights, sizeof(float)*mSize) );
        MY_CUDA_SAFE_CALL( cudaMalloc((void**)&mLights,  sizeof(AreaLightSource)*mSize) );
        //copy existing data
        MY_CUDA_SAFE_CALL( cudaMemcpy(mWeights, oldWeights, sizeof(float)*(mSize-1), cudaMemcpyDeviceToDevice) );
        MY_CUDA_SAFE_CALL( cudaMemcpy(mLights, oldLights, sizeof(AreaLightSource)*(mSize-1), cudaMemcpyDeviceToDevice) );
        //free old storage
        MY_CUDA_SAFE_CALL( cudaFree(oldCDF) );
        MY_CUDA_SAFE_CALL( cudaFree(oldWeights) );
        MY_CUDA_SAFE_CALL( cudaFree(oldLights) );
        //add new item
        MY_CUDA_SAFE_CALL( cudaMemcpy(mWeights + mSize - 1, &aWeight, sizeof(float), cudaMemcpyHostToDevice) );
        MY_CUDA_SAFE_CALL( cudaMemcpy(mLights+ mSize - 1, &aLS, sizeof(AreaLightSource), cudaMemcpyHostToDevice) );
        //compute new Cumulative Distribution Function
        MY_CUDA_SAFE_CALL( cudaMemcpy(mCDF,mWeights, sizeof(float)*mSize, cudaMemcpyDeviceToDevice) );
        InclusiveFloatScan scan;
        scan(mCDF, (uint)mSize);
    }

    HOST void normalizeALSIntensities();

};


#endif // LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
