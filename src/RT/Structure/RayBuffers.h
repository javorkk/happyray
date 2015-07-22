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

#ifndef RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5
#define RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

class SimpleRayBuffer
{
    //static const int ELEMENTSIZE = 8 + 1; //+ 1 for instance id
    void* mMemoryPtr;
public:
    SimpleRayBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void setMemPtr(void* aPtr)
    {
        mMemoryPtr = aPtr;
    }

    HOST DEVICE void* getData()
    {
        return mMemoryPtr;
    }

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        oRayOrg = loadOrigin(aRayId, aNumRays);
        oRayDir = loadDirection(aRayId, aNumRays);

        return loadDistance(aRayId, aNumRays);
    }


    HOST DEVICE void store(const float3& aRayOrg, const float3& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)
    {
        float3 hitPoint = aRayOrg + (aRayT - EPS) * aRayDir;

        float* rayOut = ((float*)mMemoryPtr) + aRayId;
        *rayOut = hitPoint.x;
        rayOut += aNumRays;
        *rayOut = hitPoint.y;
        rayOut += aNumRays;
        *rayOut = hitPoint.z;

        rayOut += aNumRays;
        *rayOut = aRayDir.x;
        rayOut += aNumRays;
        *rayOut = aRayDir.y;
        rayOut += aNumRays;
        *rayOut = aRayDir.z;

        rayOut += aNumRays;
        *rayOut = aRayT;

        rayOut += aNumRays;
        *((uint*)rayOut) = aBestHit;

    }

    HOST DEVICE void storeInput(const float3& aRayOrg, const float3& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)
    {
        //float3 hitPoint = aRayOrg + (aRayT - EPS) * aRayDir;

        float* rayOut = ((float*)mMemoryPtr) + aRayId;
        *rayOut = aRayOrg.x;
        rayOut += aNumRays;
        *rayOut = aRayOrg.y;
        rayOut += aNumRays;
        *rayOut = aRayOrg.z;

        rayOut += aNumRays;
        *rayOut = aRayDir.x;
        rayOut += aNumRays;
        *rayOut = aRayDir.y;
        rayOut += aNumRays;
        *rayOut = aRayDir.z;

        rayOut += aNumRays;
        *rayOut = FLT_MAX;

        rayOut += aNumRays;
        *((uint*)rayOut) = aBestHit;

    }

    HOST DEVICE void load(float3& oRayOrg, float3& oRayDir, float& oRayT,
        uint& oBestHit, const uint aRayId, const uint aNumRays)
    {
        float* rayOut = ((float*)mMemoryPtr) + aRayId;
        oRayOrg.x = *rayOut;
        rayOut += aNumRays;
        oRayOrg.y = *rayOut;
        rayOut += aNumRays;
        oRayOrg.z = *rayOut;

        rayOut += aNumRays;
        oRayDir.x = *rayOut;
        rayOut += aNumRays;
        oRayDir.y = *rayOut;
        rayOut += aNumRays;
        oRayDir.z = *rayOut;

        rayOut += aNumRays;
        oRayT = *rayOut;

        rayOut += aNumRays;
        oBestHit = *((uint*)rayOut);

    }

    HOST DEVICE float3 loadOrigin(const uint aRayId, const uint aNumRays)
    {
        float3 retval;

        float* ptr = (float*)mMemoryPtr + aRayId;
        retval.x = *ptr;
        ptr += aNumRays;
        retval.y = *ptr;
        ptr += aNumRays;
        retval.z = *ptr;

        return retval;
    }

    HOST DEVICE float3 loadDirection(const uint aRayId, const uint aNumRays)
    {
        float3 retval;

        float* ptr = (float*)mMemoryPtr + aRayId + aNumRays * 3;
        retval.x = *ptr;
        ptr += aNumRays;
        retval.y = *ptr;
        ptr += aNumRays;
        retval.z = *ptr;

        return retval;
    }

    HOST DEVICE void storeDistance(const float aRayT, const uint aRayId, 
        const uint aNumRays)
    {
        *((float*)mMemoryPtr + aRayId + aNumRays * 6) = aRayT;
    }

    HOST DEVICE float loadDistance(const uint aRayId, const uint aNumRays)
    {
        return *((float*)mMemoryPtr + aRayId + aNumRays * 6);
    }

    HOST DEVICE uint loadBestHit(const uint aRayId, const uint aNumRays)
    {
        return *((uint*)mMemoryPtr + aRayId + aNumRays * 7);
    }

    HOST DEVICE uint loadBestHitInstance(const uint aRayId, const uint aNumRays)
    {
        return *((uint*)mMemoryPtr + aRayId + aNumRays * 8);
    }

    HOST DEVICE void storeBestHitInstance(const uint aId, const uint aRayId,
        const uint aNumRays)
    {
        *((float*)mMemoryPtr + aRayId + aNumRays * 8) = aId;
    }
};


class DirectIlluminationBuffer
{
protected:
    void* mMemoryPtr;
public:
    DirectIlluminationBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData() const
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const float3& aRayOrg, const float3& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)  const
    {
        float3 outDir;
        if (aRayT < 0.9999f)
        {
            if(aRayT > 0.f)
            {
                //if occluded, overwrite the radiance
                float3* rayOut = ((float3*)mMemoryPtr) + aRayId;
                *rayOut = rep(0.f);
            }
            //else hit point was on light source
            outDir = rep(0.f);
        }
        else
        {
            //hit point receives direct illumination
            outDir = ~aRayDir;
        }

        float3* rayOut = ((float3*)((char*)mMemoryPtr + aNumRays * sizeof(float3)) + aRayId);
        *rayOut = outDir;

    }

    HOST DEVICE void storeLSIntensity(const float3& aVal, const uint aSampleId, const uint aNumSamples)  const
    {
        //if not occluded, store distance to light source
        float3* rayOut = ((float3*)mMemoryPtr) + aSampleId;
        *rayOut = aVal;
    }

    HOST DEVICE float3 loadLSIntensity(const uint aSampleId) const
    {
        return *((float3*)mMemoryPtr + aSampleId);
    }    
    
    HOST DEVICE float3 loadDirToLS(const uint aSampleId, const uint aNumSamples) const
    {
        return *((float3*)((char*)mMemoryPtr + aNumSamples * sizeof(float3)) + aSampleId);
    }    

    HOST DEVICE uint loadBestHitInstance(const uint aRayId, const uint aNumRays)
    {
        return (uint)-1;//dummy
    }

    HOST DEVICE void storeBestHitInstance(const uint aId, const uint aRayId,
        const uint aNumRays)
    {
        //dummy
    }

};

class AOIlluminationBuffer: public DirectIlluminationBuffer
{
public:
    const float UNOCCLUDED_RAY_LENGTH;

    AOIlluminationBuffer(void* aMemPtr, float aRayT = 0.9999f): DirectIlluminationBuffer(aMemPtr),
        UNOCCLUDED_RAY_LENGTH(aRayT)
    {}

    HOST DEVICE void store(const float3& aRayOrg, const float3& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)  const
    {
        if (aRayT < UNOCCLUDED_RAY_LENGTH)
        {
            //if occluded, overwrite the radiance
            float3* rayOut = ((float3*)mMemoryPtr) + aRayId;
            *rayOut = rep(0.f);
        }
    }

    HOST DEVICE uint loadBestHitInstance(const uint aRayId, const uint aNumRays)
    {
        return (uint)-1;//dummy
    }

    HOST DEVICE void storeBestHitInstance(const uint aId, const uint aRayId,
        const uint aNumRays)
    {
        //dummy
    }

};

class ImportanceBuffer
{
    void* mMemoryPtr;
public:
    ImportanceBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData() const
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const float3& aImportance, const uint aRayId) const
    {
        //if not occluded, store distance to light source
        float3* out = ((float3*)mMemoryPtr) + aRayId;
        *out = aImportance;
    }

    HOST DEVICE float3 loadImportance(const uint aSampleId) const
    {
        return *((float3*)mMemoryPtr + aSampleId);
    }    

};

#endif // RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5
