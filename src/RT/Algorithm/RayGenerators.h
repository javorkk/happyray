#ifdef _MSC_VER
#pragma once
#endif

#ifndef RAYGENERATORS_H_1FCDF790_9537_4905_AB8B_BCE345A4E231
#define RAYGENERATORS_H_1FCDF790_9537_4905_AB8B_BCE345A4E231

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "Utils/RandomNumberGenerators.hpp"
#include "Utils/HemisphereSamplers.hpp"
#include "RT/Primitive/Camera.h"
#include "RT/Primitive/LightSource.hpp"

template< class tPixelSampler, bool taSafe >
class RegularPrimaryRayGenerator
{
public:
    Camera dcCamera;
    tPixelSampler dcRegularPixelSampler;
    int dcImageId;

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        oRayOrg = dcCamera.getPosition();

        float screenX, screenY;
        if (taSafe)//compile-time decision
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)(min(aRayId, aNumRays - 1u)), (float)dcImageId, screenX,
                screenY);
        }
        else
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)aRayId, (float)dcImageId, screenX, screenY);
        }
        oRayDir = dcCamera.getDirection(screenX, screenY);

        return FLT_MAX;
    }
};

template < class tPixelSampler, bool taSafe >
class RandomPrimaryRayGenerator
{

public:
    Camera dcCamera;
    tPixelSampler dcRandomPixelSampler;
    int dcImageId;

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        //typedef KISSRandomNumberGenerator       t_RNG;

        //t_RNG genRand(
        //    1236789u + aRayId * 977u + aRayId,
        //    369u + aRayId * 35537u + globalThreadId1D(719u),
        //    351288629u + aRayId + globalThreadId1D(45751u),        
        //    416191069u );

        typedef SimpleRandomNumberGenerator     t_RNG;
        t_RNG genRand(globalThreadId1D() * globalThreadId1D() * globalThreadId1D() + 
            1236789u + aRayId * 35537u * dcImageId +
            (aRayId + dcImageId * aNumRays) * 
            (aRayId + dcImageId * aNumRays) *
            (aRayId + dcImageId * aNumRays) );


        float screenX = genRand();
        float screenY = genRand();

        if (taSafe)//compile-time decision
        {
            dcRandomPixelSampler.getPixelCoords(
                (float)(min(aRayId, aNumRays - 1u)), screenX, screenY);
        }
        else
        {
            dcRandomPixelSampler.getPixelCoords((float)aRayId, screenX, screenY);
        }

        oRayDir = dcCamera.getDirection(screenX, screenY);
        oRayOrg = dcCamera.getPosition();

        return FLT_MAX;
    }
};

template<class tRayBuffer, int taResX, int taResY>
class AreaLightShadowRayGenerator
{
    tRayBuffer mBuffer;

public:
    AreaLightSource dcLightSource;
    int dcImageId;

    AreaLightShadowRayGenerator(const tRayBuffer& aBuffer,
        const AreaLightSource& aLS):mBuffer(aBuffer), dcLightSource(aLS)
    {}

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        uint myPixelIndex = aRayId / dcSamples;
        uint numPixels = dcNumPixels;

        float rayT = mBuffer.loadDistance(myPixelIndex, numPixels);

        //typedef KISSRandomNumberGenerator       t_RNG;

        //t_RNG genRand(  3643u + aRayId * 4154207u + aRayId,
        //    1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
        //    331801u + aRayId + globalThreadId1D(91438197u),
        //    10499029u );

        StratifiedSampleGenerator<taResX,taResY> 
            sampleGenerator( 3643u + aRayId * 4154207u * dcSeed + aRayId,
            1761919u + aRayId * 2746753u * dcSeed /*+ globalThreadId1D(8116093u)*/,
            331801u + aRayId + dcSeed /*+ globalThreadId1D(91438197u)*/,
            10499029u );
        
        float r1 = (float)(aRayId % taResX); 
        float r2 = (float)((aRayId / taResX) % taResY);

        sampleGenerator(r1, r2);

        oRayDir = dcLightSource.getPoint(r1, r2);

        oRayOrg = mBuffer.loadOrigin(myPixelIndex, numPixels);
        //+ (rayT - EPS) * mBuffer.loadDirection(myPixelIndex, numPixels);

        oRayDir = oRayDir - oRayOrg;

        if (rayT >= FLT_MAX)
        {
            return 0.5f;
        }
        else
        {
            return FLT_MAX;
        }
    }
};

template<class tRayBuffer>
class RayLoader
{
    tRayBuffer mBuffer;
public:
    RayLoader(const tRayBuffer& aBuffer):mBuffer(aBuffer)
    {}
public:
    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        oRayOrg = mBuffer.loadOrigin(aRayId, aNumRays);
        oRayDir = mBuffer.loadDirection(aRayId, aNumRays);

        return mBuffer.loadDistance(aRayId, aNumRays);
    }
};

template<class tRayBuffer, class tPrimitiveStorage>
class DiffuseReflectionRayGenerator
{
    tRayBuffer mBuffer;
    tPrimitiveStorage mStorage;

public:
    DiffuseReflectionRayGenerator(const tRayBuffer& aBuffer, const tPrimitiveStorage& aStorage):
      mBuffer(aBuffer), mStorage(aStorage)
    {}

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        float rayT = mBuffer.loadDistance(aRayId, dcNumRays);
        
        if (rayT >= FLT_MAX)
        {
            return -1.f;
        }

        oRayOrg = mBuffer.loadOrigin(aRayId, dcNumRays);
        oRayDir = mBuffer.loadDirection(aRayId, dcNumRays);


        CosineHemisphereSampler genDir;

        //typedef KISSRandomNumberGenerator       t_RNG;

        //t_RNG genRand(  3643u + aRayId * 4154207u + aRayId,
        //    1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
        //    331801u + aRayId + globalThreadId1D(91438197u),
        //    10499029u );

        //float rand1 = genRand();
        //float rand2 = genRand();

        HaltonNumberGenerator genQuasiRand;
        float rand1 = genQuasiRand(aRayId, 0, dcPrimesRCP);
        float rand2 = genQuasiRand(aRayId, 1, dcPrimesRCP);

        uint bestHit = mStorage.indices[mBuffer.loadBestHit(aRayId, dcNumRays)];
        float3 normal = mStorage(bestHit).vertices[0];
        float3 edge1 = mStorage(bestHit).vertices[1];
        float3 edge2 = mStorage(bestHit).vertices[2];

        edge1 = edge1 - normal;
        edge2 = edge2 - normal;

        normal = ~(edge1 % edge2);

        if (dot(oRayDir,normal) > 0.f)
        {
            normal = -normal;
        }

        getLocalCoordinates(normal, edge1, edge2);


        float3 randDir = genDir(rand1, rand2);

        oRayDir = edge1 * randDir.x + edge2 * randDir.y + normal * randDir.z;

        return FLT_MAX;

    }
};

#endif // RAYGENERATORS_H_1FCDF790_9537_4905_AB8B_BCE345A4E231
