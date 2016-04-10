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

#ifndef TLGRIDHIERACHYAOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C
#define TLGRIDHIERACHYAOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C

#include "CUDAStdAfx.h"
#include "DeviceConstants.h"
#include "Core/Algebra.hpp"
#include "Core/Flags.hpp"

#include "RT/Primitive/Material.hpp"
#include "RT/Structure/FrameBuffer.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/RayBuffers.h"
#include "RT/Structure/MemoryManager.h"

#include "RT/Algorithm/RayTracingKernels.h"
#include "Utils/RandomNumberGenerators.hpp"

#include "RT/Structure/TwoLevelGridHierarchy.h"

static const int NUMAMBIENTOCCLUSIONSAMPLES_TLGH = 1;

//#define USE_3D_TEXTURE //instead of scene materials
#ifdef USE_3D_TEXTURE
typedef TexturedPhongMaterial t_Material;
#else
typedef PhongMaterial t_Material;
#endif

//////////////////////////////////////////////////////////////////////////////////////////
//in DeviceConstants.h:
//DEVICE_NO_INLINE CONSTANT uint                                        dcNumPixels;
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//Ambient Occlusion Shading Kernel
//////////////////////////////////////////////////////////////////////////////////////////
//assumes that block size is multiple of number of ambient occlusion samples per hit point
template< class tPrimitive >
GLOBAL void computeAOIlluminationTLGH(
    PrimitiveArray<tPrimitive>              aStorage,
    VtxAttributeArray<tPrimitive, float3>   aNormalStorage,
    PrimitiveAttributeArray<t_Material>     aMaterialStorage,
    SimpleRayBuffer                         aInputBuffer,
    AOIlluminationBuffer                    aOcclusionBuffer,
    GeometryInstance*                       aInstancesPtr,
    //int                                     aNumInstances,
    FrameBuffer                             oFrameBuffer,
    int                                     dcNumRays, //AO rays
    int                                     dcImageId)
{
    extern SHARED float3 sharedVec[];
    sharedVec[threadId1D()] = rep(0.f);

    float3 rayOrg = rep(0.f);
    float3 rayDir = rep(1.f);
    uint numIterations = dcNumRays - dcNumRays % blockSize() + blockSize();
    for (uint myRayIndex = globalThreadId1D();
        myRayIndex < numIterations;
        myRayIndex += numThreads())
    {

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT = FLT_MAX;
        uint  bestHit = 0u;
        const uint myPixelIndex = min(dcNumPixels - 1, myRayIndex / NUMAMBIENTOCCLUSIONSAMPLES_TLGH);

        SYNCTHREADS;

        //////////////////////////////////////////////////////////////////////////
        //load occlusion information in shared memory
        if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = aOcclusionBuffer.loadLSIntensity(myRayIndex);
            aInputBuffer.load(rayOrg, rayDir, rayT, bestHit, myPixelIndex, dcNumPixels);
        }
        //////////////////////////////////////////////////////////////////////////


        SYNCTHREADS;

        if (myRayIndex < dcNumRays && rayT < FLT_MAX)
        {
            if (sharedVec[threadId1D()].x + sharedVec[threadId1D()].y + sharedVec[threadId1D()].z > 0.f && bestHit < aStorage.numPrimitives)
            {
                tPrimitive prim = aStorage[bestHit];
                float3& vert0 = prim.vtx[0];
                float3& vert1 = prim.vtx[1];
                float3& vert2 = prim.vtx[2];

                float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

                float3 rayOrgT = rayOrg;
                const uint instanceId = aInputBuffer.loadBestHitInstance(myPixelIndex, dcNumPixels);
                if (instanceId < (uint)-1)
                {
                    rayOrgT = aInstancesPtr[instanceId].transformPointToLocal(rayOrgT);
                }

                //Compute barycentric coordinates
                vert0 = vert0 - rayOrgT;
                vert1 = vert1 - rayOrgT;
                vert2 = vert2 - rayOrgT;

                float3 n0 = vert1 % vert2;
                float3 n1 = vert2 % vert0;

                float twiceSabc_RCP = lenRCP(realNormal);
                float u = len(n0) * twiceSabc_RCP;
                float v = len(n1) * twiceSabc_RCP;

                /*VtxAttribStruct<tPrimitive, float3> normals;
                normals = aNormalStorage[bestHit];
                float3& normal0 = normals.data[0];
                float3& normal1 = normals.data[1];
                float3& normal2 = normals.data[2];*/

                float3 normal = ~realNormal;//~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);

                if (instanceId < (uint)-1)
                {
                    normal = aInstancesPtr[instanceId].transformNormalToGlobal(normal);
                }

                t_Material material = aMaterialStorage[bestHit];
                float3 diffReflectance = material.getDiffuseReflectance(rayOrg.x, rayOrg.y, rayOrg.z);

                sharedVec[threadId1D()].x *= diffReflectance.x;
                sharedVec[threadId1D()].y *= diffReflectance.y;
                sharedVec[threadId1D()].z *= diffReflectance.z;

                float sinZNormal = fmaxf(0.f, sqrtf(1.f - normal.z));
                float cosEyeNormal = fabsf(dot(rayDir, normal));
                sharedVec[threadId1D()].x *= (0.7f * cosEyeNormal + 0.3f * sinZNormal);
                sharedVec[threadId1D()].y *= (0.7f * cosEyeNormal + 0.2f * sinZNormal);
                sharedVec[threadId1D()].z *= (0.9f * cosEyeNormal /*+ 0.1f * sinZNormal + 0.1f * (1.f  - sinZNormal)*/);
                //////////////////////////////////////////////////////////////////////////
                //Triangle edges
                //sharedVec[threadId1D()].x = u;//1.f;// M_PI_RCP; //u;
                //sharedVec[threadId1D()].y = v;//1.f;//M_PI_RCP; //v;
                //sharedVec[threadId1D()].z = 1.f-u-v;//1.f;//M_PI_RCP; //1.f - u - v;
                //bool edgePt = u <= 0.01f || v <= 0.01f || u + v >= 0.99f;
                //float edgeColorFactor = edgePt ? 0.f : 1.f;
                //sharedVec[threadId1D()] *= edgeColorFactor;
                //////////////////////////////////////////////////////////////////////////
                //sharedVec[threadId1D()] = rep(0.01f*rayT);
            }
        }
        else if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = rep(1.f);//Background color
        }//endif hit point exists

        SYNCTHREADS;

        //one thread per pixel instead of per occlusion sample
        if (myRayIndex < dcNumRays && myRayIndex % NUMAMBIENTOCCLUSIONSAMPLES_TLGH == 0u)
        {
            float3 oRadiance = rep(0.f);

            for (uint i = 0; i < NUMAMBIENTOCCLUSIONSAMPLES_TLGH; ++i)
            {
                oRadiance = oRadiance + sharedVec[threadId1D() + i]
                    * 1.f / (float)NUMAMBIENTOCCLUSIONSAMPLES_TLGH;
            }

            float newSampleWeight = 1.f / (float)(dcImageId + 1);
            float oldSamplesWeight = 1.f - newSampleWeight;


            oFrameBuffer[myPixelIndex] =
                oFrameBuffer[myPixelIndex] * oldSamplesWeight +
                oRadiance * newSampleWeight;
        }

    }
}

GLOBAL void initInstanceIdBuffer(SimpleRayBuffer aRayBuffer, int dcNumRays)
{
    for (uint myRayIndex = globalThreadId1D();
        myRayIndex < dcNumRays;
        myRayIndex += numThreads())
    {
        aRayBuffer.storeBestHitInstance((uint)-1, myRayIndex, dcNumRays);
    }
}

template<class tPrimitive, int taResX>
class TLGHAmbientOcclusionRayGenerator
{
    void* mMemoryPtr;
public:

    TLGHAmbientOcclusionRayGenerator(
        const SimpleRayBuffer                          aBuffer,
        const AOIlluminationBuffer                     aOccBuff,
        const PrimitiveArray<tPrimitive>               aPrimitiveStorage,
        const VtxAttributeArray<tPrimitive, float3>    aNormalStorage,
        int                                            aImageId,
        GeometryInstance*                              aInstancesPtr,
        //int                                            aNumInstances,
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
        nextSlot = (char*)nextSlot + sizeof(int);
        memcpy(nextSlot, (void*)&aInstancesPtr, sizeof(GeometryInstance*));
        //nextSlot = (char*)nextSlot + sizeof(GeometryInstance*);
        //memcpy(nextSlot, (void*)&aNumInstances, sizeof(int));

        MY_CUDA_SAFE_CALL(cudaMemcpy(aMemPtr, hostPtr, getParametersSize(), cudaMemcpyHostToDevice));
        MY_CUDA_SAFE_CALL(cudaFreeHost(hostPtr));
        MY_CUT_CHECK_ERROR("Upload failed!\n");
        mMemoryPtr = aMemPtr;

    }

    static int getParametersSize()
    {
        return 2 * sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>) + sizeof(VtxAttributeArray<tPrimitive, float3>) + sizeof(int) + sizeof(GeometryInstance*) /*+ sizeof(int)*/;
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

    DEVICE GeometryInstance* getInstances()
    {
        return (GeometryInstance*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>) + sizeof(VtxAttributeArray<tPrimitive, float3>) + sizeof(int));
    }

    //DEVICE int getNumInstances()
    //{
    //    return *(int*)((char*)mMemoryPtr + sizeof(SimpleRayBuffer) + sizeof(AOIlluminationBuffer) + sizeof(PrimitiveArray<tPrimitive>) + sizeof(VtxAttributeArray<tPrimitive, float3>) + sizeof(int) + sizeof(GeometryInstance*));
    //}

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

        GeometryInstance* instancesPtr = getInstances();

        float3 rayOrgT = oRayOrg;
        const uint instanceId = getBuffer()->loadBestHitInstance(myPixelIndex, numPixels);
        if (instanceId < (uint)-1)
        {
            rayOrgT = instancesPtr[instanceId].transformPointToLocal(rayOrgT);
        }

        //Compute barycentric coordinates
        vert0 = vert0 - rayOrgT;
        vert1 = vert1 - rayOrgT;
        vert2 = vert2 - rayOrgT;

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

        if (instanceId < (uint)-1)
        {
            normal = instancesPtr[instanceId].transformNormalToGlobal(normal);
        }

        //if (dot(normal, oRayDir) > 0.f)
        //    normal = -normal;



        //////////////////////////////////////////////////////////////////////////
        //generate random direction and transform it in global coordinates
        typedef KISSRandomNumberGenerator       t_RNG;

        t_RNG genRand(3643u + aRayId * 4154207u * getSampleId() + aRayId,
            1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
            331801u + aRayId + getSampleId() + globalThreadId1D(91438197u),
            10499029u);

        CosineHemisphereSampler getRandDirection;
        float3 tmpDir = getRandDirection(genRand(), genRand());
        float3 tangent, binormal;
        getLocalCoordinates(normal, tangent, binormal);

        oRayDir = ~(normal * tmpDir.z + tangent * tmpDir.x + binormal * tmpDir.y);
        oRayOrg += 0.01f * getOcclusionBuffer()->UNOCCLUDED_RAY_LENGTH * normal;

        float3 lsRadiance = rep(1.f);
        getOcclusionBuffer()->storeLSIntensity(lsRadiance, aRayId, aNumShadowRays);


        return  FLT_MAX;

    }
};

template<
    class tPrimitive,
        template <class, class, bool> class tTraverser,
    class tPrimaryIntersector,
    class tAlternativeIntersector>

class TLGHAOIntegrator
{
    int* mGlobalMemoryPtr;
    size_t mGlobalMemorySize;
    int* mAORayGeneratorMemoryPtr;
    size_t mAORayGeneratorMemorySize;
    cudaEvent_t mTrace, mShade;
    float mAlpha;

public:
    typedef RandomPrimaryRayGenerator< GaussianPixelSampler, true > t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                                         t_RayBuffer;
    typedef tPrimaryIntersector                                     t_Intersector;
    typedef TwoLevelGridHierarchy                                   t_AccelerationStructure;
    typedef TLGHAmbientOcclusionRayGenerator < tPrimitive, NUMAMBIENTOCCLUSIONSAMPLES_TLGH >  t_AORayGenerator;
    typedef AOIlluminationBuffer                                    t_AOcclusionBuffer;

    t_RayBuffer             rayBuffer;


    TLGHAOIntegrator(float aAlpha = 10.f) :rayBuffer(t_RayBuffer(NULL)),
        mGlobalMemorySize(0u), mAORayGeneratorMemorySize(0u), mAlpha(aAlpha)
    {}

    ~TLGHAOIntegrator()
    {
        cleanup();
    }

    HOST void setAlpha(float aAlpha)
    {
        mAlpha = aAlpha;
    }

    HOST void integrate(
        PrimitiveArray<tPrimitive>&                     aStorage,
        VtxAttributeArray<tPrimitive, float3>&          aNormalStorage,
        PrimitiveAttributeArray<t_Material>&         aMaterialStorage,
        t_AccelerationStructure&                        aAccStruct,
        t_PrimaryRayGenerator&                          aRayGenerator,
        FrameBuffer&                                    aFrameBuffer,
        const int                                       aImageId
        )
    {
        const uint sharedMemoryTrace = SHARED_MEMORY_TRACE;

        const uint numPixels = aFrameBuffer.resX * aFrameBuffer.resY;
        const uint globalMemorySize = sizeof(uint) +    //Persistent threads
            numPixels * sizeof(float3) +                //rayBuffer : rayOrg
            numPixels * sizeof(float3) +                //rayBuffer : rayDir
            numPixels * sizeof(float) +                 //rayBuffer : rayT
            numPixels * sizeof(uint) +                  //rayBuffer : primitive Id
            numPixels * sizeof(uint) +                  //rayBuffer : instance Id
            numPixels * sizeof(float3) +                //importanceBuffer : importance
            numPixels * NUMAMBIENTOCCLUSIONSAMPLES_TLGH * (sizeof(float3)) + //ambient occlusion buffer: intensity               
            0u;

        MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);
        MemoryManager::allocateDeviceArray((void**)&mAORayGeneratorMemoryPtr, t_AORayGenerator::getParametersSize(), (void**)&mAORayGeneratorMemoryPtr, mAORayGeneratorMemorySize);

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //primary rays
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        MY_CUDA_SAFE_CALL(cudaMemset(mGlobalMemoryPtr, 0, sizeof(uint)));
        rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);
        dim3 blocks(numPixels / 512u + 1u);
        dim3 threads(512u);
        initInstanceIdBuffer << < blocks, threads >> >(rayBuffer, numPixels);

        MY_CUT_CHECK_ERROR("Setting memory for ray tracing failed!\n");

        dim3 threadBlockTrace(RENDERTHREADSX, RENDERTHREADSY);
        dim3 blockGridTrace(RENDERBLOCKSX, RENDERBLOCKSY);

        cudaEventCreate(&mTrace);

        trace<tPrimitive, TwoLevelGridHierarchy, t_PrimaryRayGenerator, t_RayBuffer, tTraverser, tPrimaryIntersector, false>
            << < blockGridTrace, threadBlockTrace, sharedMemoryTrace >> >(
            aStorage,
            aRayGenerator,
            aAccStruct,
            rayBuffer,
            numPixels,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing primary rays failed!\n");

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //Ambient Occlusion rays
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        MY_CUDA_SAFE_CALL(cudaMemcpyToSymbol(dcNumPixels, &numPixels, sizeof(uint)));

        t_AOcclusionBuffer occlusionBuffer(mGlobalMemoryPtr +
            1 +                             //Persistent threads
            numPixels * 3 +                 //rayBuffer : rayOrg
            numPixels * 3 +                 //rayBuffer : rayDir
            numPixels +                     //rayBuffer : rayT
            numPixels +                     //rayBuffer : primitive Id
            numPixels +                     //rayBuffer : instance Id
            numPixels * 3 +                 //importanceBuffer : importance
            0u,
            mAlpha);

        t_AORayGenerator   ambientOcclusionRayGenerator(
            rayBuffer,
            occlusionBuffer,
            aStorage,
            aNormalStorage,
            aImageId,
            aAccStruct.instances,
            //aAccStruct.numInstances,
            mAORayGeneratorMemoryPtr);


        MY_CUDA_SAFE_CALL(cudaMemset(mGlobalMemoryPtr, 0, sizeof(uint)));
        const uint numAORays = numPixels * NUMAMBIENTOCCLUSIONSAMPLES_TLGH;

        trace<tPrimitive, TwoLevelGridHierarchy, t_AORayGenerator, t_AOcclusionBuffer, tTraverser, tPrimaryIntersector, true>
            << < blockGridTrace, threadBlockTrace, sharedMemoryTrace >> >(
            aStorage,
            ambientOcclusionRayGenerator,
            aAccStruct,
            occlusionBuffer,
            numAORays,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing ambient occlusion rays failed!\n");

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //AO illumination
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        dim3 threadBlockDI(24 * NUMAMBIENTOCCLUSIONSAMPLES_TLGH);
        dim3 blockGridDI(120);

        const uint sharedMemoryShade =
            threadBlockDI.x * sizeof(float3) +   //light vector   
            0u;

        computeAOIlluminationTLGH < tPrimitive >
            << < blockGridDI, threadBlockDI, sharedMemoryShade >> >(
            aStorage,
            aNormalStorage,
            aMaterialStorage,
            rayBuffer,
            occlusionBuffer,
            aAccStruct.instances,
            //aAccStruct.numInstances,
            aFrameBuffer,
            numAORays,
            aImageId);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Computing ambient illumination failed!\n");

        cudaEventDestroy(mTrace);
    }

    HOST void cleanup()
    {
        if (mGlobalMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL(cudaFree(mGlobalMemoryPtr));
            mGlobalMemoryPtr = NULL;
            mGlobalMemorySize = 0u;
        }
        if (mAORayGeneratorMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL(cudaFree(mAORayGeneratorMemoryPtr));
            mAORayGeneratorMemoryPtr = NULL;
            mAORayGeneratorMemorySize = 0u;
        }
    }

};


#endif // TLGRIDHIERACHYAOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C