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

#ifndef RAYGENERATORS_H_1FCDF790_9537_4905_AB8B_BCE345A4E231
#define RAYGENERATORS_H_1FCDF790_9537_4905_AB8B_BCE345A4E231

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "Utils/RandomNumberGenerators.hpp"
#include "Utils/HemisphereSamplers.hpp"
#include "RT/Primitive/Camera.h"
#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/RayBuffers.h"

template< class tPixelSampler, bool taSafe >
class RegularPrimaryRayGenerator
{
public:
    Camera dcCamera;
    tPixelSampler dcRegularPixelSampler;
    int sampleId;

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        oRayOrg = dcCamera.getPosition();

        float screenX, screenY;
        if (taSafe)//compile-time decision
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)(min(aRayId, aNumRays - 1u)), (float)sampleId, screenX,
                screenY);
        }
        else
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)aRayId, (float)sampleId, screenX, screenY);
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
    int sampleId;

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
            1236789u + aRayId * 35537u * sampleId +
            (aRayId + sampleId * aNumRays) * 
            (aRayId + sampleId * aNumRays) *
            (aRayId + sampleId * aNumRays) );


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

template<int taResX, int taResY>
class AreaLightShadowRayGenerator
{
    SimpleRayBuffer mBuffer;
    DirectIlluminationBuffer mOcclusionBuffer;

public:
    AreaLightSourceCollection lightSources;
    int sampleId;

    AreaLightShadowRayGenerator(
        const SimpleRayBuffer& aBuffer,
        const DirectIlluminationBuffer& aOccBuff,
        const AreaLightSourceCollection& aLSCollection,
        int aImageId):mBuffer(aBuffer), mOcclusionBuffer(aOccBuff), lightSources(aLSCollection), sampleId(aImageId)
    {}

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumShadowRays)
    {
        uint numPixels = aNumShadowRays / (taResX * taResY);
        uint myPixelIndex = aRayId / (taResX * taResY);

        float rayT = mBuffer.loadDistance(myPixelIndex, numPixels);

        if (rayT >= FLT_MAX)
        {
            return 0.5f;
        }

        typedef KISSRandomNumberGenerator       t_RNG;

        t_RNG genRand(  3643u + aRayId * 4154207u * sampleId + aRayId,
            1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
            331801u + aRayId + sampleId + globalThreadId1D(91438197u),
            10499029u );

        int lightSourceId = 0;
        AreaLightSource lightSource = lightSources.getLight(genRand(), lightSourceId);

        oRayOrg = mBuffer.loadOrigin(myPixelIndex, numPixels);
        //+ (rayT - EPS) * mBuffer.loadDirection(myPixelIndex, numPixels);

        float3 lsRadiance;

        bool isOnLS = lightSource.isOnLS(oRayOrg);
        if (isOnLS)
        {
            oRayDir = - mBuffer.loadDirection(myPixelIndex, numPixels);
            float cosLightNormal = dot(oRayDir,lightSource.normal);
            float receivesEnergy = (cosLightNormal > 0.f) ? .5f : 0.f; //0.5f is power heuristic n = 0
            lsRadiance = lightSource.intensity * receivesEnergy;
        }
        else
        {
            StratifiedSampleGenerator<taResX,taResY> 
                sampleGenerator( 3643u + aRayId * 4154207u * sampleId + aRayId,
                1761919u + aRayId * 2746753u + sampleId /*+ globalThreadId1D(8116093u)*/,
                331801u + aRayId * sampleId  + sampleId/*+ globalThreadId1D(91438197u)*/,
                10499029u );

            float r1 = (float)(aRayId % taResX); 
            float r2 = (float)((aRayId / taResX) % taResY);

            sampleGenerator(r1, r2);

            oRayDir = lightSource.getPoint(r1, r2);
            oRayDir = oRayDir - oRayOrg;

            float attenuation = 1.f / dot(oRayDir,oRayDir);

            //normalize
            float3 DirN = oRayDir * sqrtf(attenuation);

            float cosLightNormal = dot(-lightSource.normal, DirN);

            lsRadiance = lightSource.intensity *
                lightSource.getArea() *
                attenuation * 
                fmaxf(0.f, cosLightNormal) *
                0.5f; //0.5f is power heuristic n = 0
        }

        mOcclusionBuffer.storeLSIntensity(lsRadiance, aRayId, aNumShadowRays);


        return  (isOnLS) ? -1.f : FLT_MAX;

    }
};

#include "RT/Structure/PrimitiveArray.h"

template<class tPrimitive, int taResX>
class AmbientOcclusionRayGenerator
{
    void* mMemoryPtr;
public:

    AmbientOcclusionRayGenerator(
        const SimpleRayBuffer                          aBuffer,
        const AOIlluminationBuffer                     aOccBuff,
        const PrimitiveArray<tPrimitive>               aPrimitiveStorage,
        const VtxAttributeArray<tPrimitive, float3>    aNormalStorage,
        int                                            aImageId,
        void* aMemPtr)
    {
        void* hostPtr;
        MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&hostPtr, getParametersSize(), cudaHostAllocDefault));
        memcpy(hostPtr, (void*)&aBuffer, sizeof(SimpleRayBuffer));        
        void* nextSlot = (char*)hostPtr + sizeof(SimpleRayBuffer);
        memcpy(nextSlot, (void*)&aOccBuff, sizeof(AOIlluminationBuffer));
        nextSlot = (char*)nextSlot + sizeof(AOIlluminationBuffer);
        memcpy(nextSlot, (void*)&aPrimitiveStorage, sizeof(PrimitiveArray<tPrimitive>));
        nextSlot = (char*)nextSlot + sizeof(PrimitiveArray<tPrimitive>);
        memcpy(nextSlot, (void*)&aNormalStorage, sizeof(VtxAttributeArray<tPrimitive, float3>));
        nextSlot = (char*)nextSlot + sizeof(VtxAttributeArray<tPrimitive, float3>);
        memcpy(nextSlot, (void*)&aImageId, sizeof(int));

        MY_CUDA_SAFE_CALL(cudaMemcpy(aMemPtr, hostPtr, getParametersSize(), cudaMemcpyHostToDevice));
        MY_CUDA_SAFE_CALL(cudaFreeHost(hostPtr));
        MY_CUT_CHECK_ERROR("Upload failed!\n");
        mMemoryPtr = aMemPtr;

    }
    
    static int getParametersSize()
    {
         return 2*sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>) + sizeof(VtxAttributeArray<tPrimitive, float3>) + sizeof(int);
    }
    DEVICE SimpleRayBuffer* getBuffer()
    {
        return (SimpleRayBuffer*)mMemoryPtr;
    }

    DEVICE AOIlluminationBuffer* getOcclusionBuffer()
    {
        return (AOIlluminationBuffer*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer));
    }

    DEVICE PrimitiveArray<tPrimitive>* getPrimitiveStorage()
    {
        return (PrimitiveArray<tPrimitive>*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer));
    }

    DEVICE VtxAttributeArray<tPrimitive, float3>* getNormalStorage()
    {
        return (VtxAttributeArray<tPrimitive, float3>*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>));
    }

    DEVICE int getSampleId()
    {
        return *(int*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>) + sizeof(VtxAttributeArray<tPrimitive, float3>));

    }

    DEVICE float operator()(float3& oRayOrg, float3& oRayDir, const uint aRayId,
        const uint aNumShadowRays)
    {
        //////////////////////////////////////////////////////////////////////////
        //load hit point data
        uint numPixels = aNumShadowRays / taResX;
        uint myPixelIndex = aRayId / taResX;

        float rayT;
        uint bestHit = 0u;
        
        getBuffer()->load(oRayOrg, oRayDir, rayT, bestHit, myPixelIndex, numPixels);

        if (rayT >= FLT_MAX || bestHit >= getPrimitiveStorage()->numPrimitives)
        {
            return 0.0f;
        }

        //////////////////////////////////////////////////////////////////////////
        //compute surface normal
        tPrimitive prim = (*getPrimitiveStorage())[bestHit];
        float3& vert0 = prim.vtx[0];
        float3& vert1 = prim.vtx[1];
        float3& vert2 = prim.vtx[2];

        float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

        //Compute barycentric coordinates
        vert0 = vert0 - oRayOrg;
        vert1 = vert1 - oRayOrg;
        vert2 = vert2 - oRayOrg;

        float3 n0 = vert1 % vert2;
        float3 n1 = vert2 % vert0;

        float twiceSabc_RCP = lenRCP(realNormal);
        float u = len(n0) * twiceSabc_RCP;
        float v = len(n1) * twiceSabc_RCP;

        //VtxAttribStruct<tPrimitive, float3> normals;
        //normals = (*getNormalStorage())[bestHit];
        //float3& normal0 = normals.data[0];
        //float3& normal1 = normals.data[1];
        //float3& normal2 = normals.data[2];

        float3 normal = ~realNormal; /* ~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);  */
        if(dot(normal, oRayDir) > 0.f)
            normal = -normal;
        //////////////////////////////////////////////////////////////////////////
        //generate random direction and transform it in global coordinates
        typedef KISSRandomNumberGenerator       t_RNG;

        t_RNG genRand(  3643u + aRayId * 4154207u * getSampleId() + aRayId,
            1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
            331801u + aRayId + getSampleId() + globalThreadId1D(91438197u),
            10499029u );

        CosineHemisphereSampler getRandDirection;
        float3 tmpDir = getRandDirection(genRand(), genRand());
        float3 tangent, binormal;
        getLocalCoordinates(normal, tangent, binormal);

        oRayDir = ~(normal * tmpDir.z + tangent * tmpDir.x + binormal * tmpDir.y);
        oRayOrg += 0.01f * getOcclusionBuffer()->UNOCCLUDED_RAY_LENGTH * oRayDir;

        float3 lsRadiance = rep(1.f);
        getOcclusionBuffer()->storeLSIntensity(lsRadiance, aRayId, aNumShadowRays);


        return  FLT_MAX;

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
