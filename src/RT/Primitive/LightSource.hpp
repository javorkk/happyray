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

#define MAX_DIMENSION(aX, aY, aZ)	                           \
    (aX > aY) ? ((aX > aZ) ? 0 : 2)	: ((aY > aZ) ? 1 : 2)

struct AreaLightSource
{
    float3 position, normal, intensity, edge1, edge2;
    int e1MaxDimension, e2MaxDimension;

    DEVICE HOST float3 AreaLightSource::getPoint(float aXCoord, float aYCoord) const
    {
        float3 result;
        result = position + aXCoord * edge1 + aYCoord * edge2;
        return result;
    }

    DEVICE HOST bool AreaLightSource::isOnLS(float3 aPt) const
    {
        bool onPlane = dot(aPt - position, normal) < 0.0001f;

        if(!onPlane) return false;

        //precomputed values for isOnLS
        const int e1MaxDimension = MAX_DIMENSION(fabsf(edge1.x), fabsf(edge1.y), fabsf(edge1.z));
        const int e2MaxDimension = MAX_DIMENSION(fabsf(edge2.x), fabsf(edge2.y), fabsf(edge2.z));


        float e1x = toPtr(edge1)[e1MaxDimension];
        float e1y = toPtr(edge1)[e2MaxDimension];
        float e2x = toPtr(edge2)[e1MaxDimension];
        float e2y = toPtr(edge2)[e2MaxDimension];
        float px = toPtr(position)[e1MaxDimension];
        float py = toPtr(position)[e2MaxDimension];
        float xx = toPtr(aPt)[e1MaxDimension];
        float xy = toPtr(aPt)[e2MaxDimension]; 

        float beta = (xy - py) - (e1y / e1x) * (xx - px);
        beta /= e2y - e2x * (e1y / e1x);
        float alpha = ((xx - px) - beta * e2x) / e1x;
        bool inside = alpha > -0.0001f && alpha < 1.0001f && beta > -0.001f && beta < 1.0001f; 
        return inside;
    }

    DEVICE HOST float AreaLightSource::getArea() const
    {
        return len((edge1 % edge2));
    }

    DEVICE HOST void AreaLightSource::create(
        const float3& aVtx0, const float3& aVtx1,
        const float3& aVtx2, const float3& aVtx3,
        const float3& aIntensity, const float3& aNormal)
    {
        position = aVtx0;
        normal = aNormal;
        edge1 = aVtx1 - aVtx0;
        edge2 = aVtx3 - aVtx0;
        intensity = aIntensity;
        init();
    }

    DEVICE HOST void init()
    {}

};


class AreaLightSourceCollection
{
    int mSize;
    float* mCDF;
    float* mWeights;
    AreaLightSource* mLights;
public:

#ifndef __CUDA_ARCH__
    AreaLightSourceCollection()
        :mSize(0u), mCDF(NULL), mWeights(NULL), mLights(NULL)
    {}
#endif

    DEVICE AreaLightSource getLight(float aRandNum, int& oId)
    {
        for(int i=0; i < mSize - 1; ++i)
        {
            if(aRandNum < mCDF[i] / mCDF[mSize - 1])
            {
                oId = i;
                return mLights[i];
            }
        }
        oId = mSize-1;
        return mLights[mSize-1];
    }

    DEVICE AreaLightSource getLightWithID(int aId)
    {
        if(aId >= mSize)
            return mLights[0];

        return mLights[aId];
    }

    DEVICE float getWeight(int aId)
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
        if(mSize == 0)
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

#undef MAX_DIMENSION

#endif // LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
