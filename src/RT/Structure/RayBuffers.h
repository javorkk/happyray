#ifdef _MSC_VER
#pragma once
#endif

#ifndef RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5
#define RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

class SimpleRayBuffer
{
    static const int ELEMENTSIZE = 8;
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
};


class OcclusionRayBuffer
{
    void* mMemoryPtr;
public:
    OcclusionRayBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData()
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const float3& aRayOrg, const float3& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)
    {
        //if not occluded, store distance to light source
        float3* rayOut = ((float3*)mMemoryPtr) + aRayId;
        if (aRayT >= 0.9999f)
        {
            *rayOut = aRayDir;
        }
        else
        {
            *rayOut = rep(FLT_MAX);
        }
    }

    HOST DEVICE float3 loadLigtVec(const uint aSampleId)
    {
        return *((float3*)mMemoryPtr + aSampleId);
    }    
    
};

class RefractionRayBuffer
{
    void* mMemoryPtr;
public:
    RefractionRayBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData()
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const float3& aAttenuation, const uint aRayId)
    {
        //if not occluded, store distance to light source
        float3* out = ((float3*)mMemoryPtr) + aRayId;
        *out = aAttenuation;
    }

    HOST DEVICE float3 loadAttenuation(const uint aSampleId)
    {
        return *((float3*)mMemoryPtr + aSampleId);
    }    

};

#endif // RAYBUFFERS_H_4B1C9F62_3AE1_4DAA_9F06_1053FA37C0D5
