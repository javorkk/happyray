#ifdef _MSC_VER
#pragma once
#endif

#ifndef AOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C
#define AOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C

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

static const int NUMAMBIENTOCCLUSIONSAMPLES  = 1;

//////////////////////////////////////////////////////////////////////////
//in DeviceConstants.h:
//DEVICE_NO_INLINE CONSTANT uint                            dcNumPixels;
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//Ambient Occlusion Shading Kernel
//////////////////////////////////////////////////////////////////////////////////////////

//assumes that block size is multiple of number of ambient occlusion samples per hit point
template< class tPrimitive >
GLOBAL void computeAOIllumination(
    PrimitiveArray<tPrimitive>              aStorage,
    VtxAttributeArray<tPrimitive, float3>   aNormalStorage,
    SimpleRayBuffer                         aInputBuffer,
    DirectIlluminationBuffer                aOcclusionBuffer,
    FrameBuffer                             oFrameBuffer,
    int                                     dcNumRays, //AO rays
    int                                     dcImageId)
{
    extern SHARED float3 sharedVec[];
    sharedVec[threadId1D()] = rep(0.f);

    float3 rayOrg = rep(0.f);
    float3 rayDir = rep(1.f);

    for(uint myRayIndex = globalThreadId1D();
        myRayIndex - threadId1D() < dcNumRays;
        myRayIndex += numThreads())
    {

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT = FLT_MAX;
        uint  bestHit = 0u;
        const uint myPixelIndex = min(dcNumPixels - 1, myRayIndex / NUMAMBIENTOCCLUSIONSAMPLES);

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

        if (myRayIndex < dcNumRays && rayT < FLT_MAX )
        {
            if (sharedVec[threadId1D()].x + sharedVec[threadId1D()].y + sharedVec[threadId1D()].z > 0.f && bestHit < aStorage.numPrimitives)
            {
                tPrimitive prim = aStorage[bestHit];
                float3& vert0 = prim.vtx[0];
                float3& vert1 = prim.vtx[1];
                float3& vert2 = prim.vtx[2];

                float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

                //Compute barycentric coordinates
                vert0 = vert0 - rayOrg;
                vert1 = vert1 - rayOrg;
                vert2 = vert2 - rayOrg;

                float3 n0 = vert1 % vert2;
                float3 n1 = vert2 % vert0;

                float twiceSabc_RCP = lenRCP(realNormal);
                float u = len(n0) * twiceSabc_RCP;
                float v = len(n1) * twiceSabc_RCP;

                VtxAttribStruct<tPrimitive, float3> normals;
                normals = aNormalStorage[bestHit];
                float3& normal0 = normals.data[0];
                float3& normal1 = normals.data[1];
                float3& normal2 = normals.data[2];

                float3 normal = ~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);

                PhongMaterial material = dcMaterialStorage[bestHit];
                float3 diffReflectance = material.getDiffuseReflectance();

                sharedVec[threadId1D()].x *= diffReflectance.x;
                sharedVec[threadId1D()].y *= diffReflectance.y;
                sharedVec[threadId1D()].z *= diffReflectance.z;

            }
        }
        else if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = rep(0.f);
        }//endif hit point exists

        SYNCTHREADS;

        //one thread per pixel instead of per occlusion sample
        if (myRayIndex < dcNumRays && myRayIndex % NUMAMBIENTOCCLUSIONSAMPLES == 0u )
        {
            float3 oRadiance = rep(0.f);

            for(uint i = 0; i < NUMAMBIENTOCCLUSIONSAMPLES; ++i)
            {
                oRadiance =  oRadiance + sharedVec[threadId1D() + i] 
                * 1.f / (float)NUMAMBIENTOCCLUSIONSAMPLES;
            }

            float newSampleWeight = 1.f / (float)(dcImageId + 1);
            float oldSamplesWeight = 1.f - newSampleWeight;

            oFrameBuffer[myPixelIndex] =
                oFrameBuffer[myPixelIndex] * oldSamplesWeight +
                oRadiance * newSampleWeight;
        }

    }
}

template<
    class tPrimitive,
    class tAccelerationStructure,
        template <class, class, bool> class tTraverser,
    class tPrimaryIntersector,
    class tAlternativeIntersector>

class AOIntegrator
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
    typedef tAccelerationStructure                                  t_AccelerationStructure;
    typedef AmbientOcclusionRayGenerator < tPrimitive, NUMAMBIENTOCCLUSIONSAMPLES >  t_AORayGenerator;
    typedef DirectIlluminationBuffer                                t_OcclusionBuffer;

    t_RayBuffer             rayBuffer;


    AOIntegrator(float aAlpha = 10.f):rayBuffer(t_RayBuffer(NULL)), 
        mGlobalMemorySize(0u), mAORayGeneratorMemorySize(0u)
    {}

    ~AOIntegrator()
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
        PrimitiveAttributeArray<PhongMaterial>&         aMaterialStorage,
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
            numPixels * sizeof(float3) +                //importanceBuffer : importance
            numPixels * NUMAMBIENTOCCLUSIONSAMPLES * (sizeof(float3) + sizeof(float3)) + //ambient occlusion buffer: intensity and dummy direction               
            0u;

        MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);
        MemoryManager::allocateDeviceArray((void**)&mAORayGeneratorMemoryPtr, t_AORayGenerator::getParametersSize(), (void**)&mAORayGeneratorMemoryPtr, mAORayGeneratorMemorySize);

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //primary rays
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );
        rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);

        dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );

        cudaEventCreate(&mTrace);

        trace<tPrimitive, tAccelerationStructure, t_PrimaryRayGenerator, t_RayBuffer, tTraverser, tPrimaryIntersector, false>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>(
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
        MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumPixels", &numPixels, sizeof(uint)) );

        t_OcclusionBuffer occlusionBuffer(mGlobalMemoryPtr + 
            1 +                             //Persistent threads
            numPixels * 3 +                 //rayBuffer : rayOrg
            numPixels * 3 +                 //rayBuffer : rayDir
            numPixels +                     //rayBuffer : rayT
            numPixels +                     //rayBuffer : primitive Id
            numPixels * 3 +                 //importanceBuffer : importance
            0u,
            mAlpha);

        t_AORayGenerator   ambientOcclusionRayGenerator(
            rayBuffer,
            occlusionBuffer,
            aStorage,
            aNormalStorage,
            aImageId,
            mAORayGeneratorMemoryPtr);


        MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );
        const uint numAORays = numPixels * NUMAMBIENTOCCLUSIONSAMPLES;

        trace<tPrimitive, tAccelerationStructure, t_AORayGenerator, t_OcclusionBuffer, tTraverser, tPrimaryIntersector, true>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>(
            aStorage,
            ambientOcclusionRayGenerator,
            aAccStruct,
            occlusionBuffer,
            numAORays,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing shadow rays failed!\n");

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //direct illumination at the end of each path
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        dim3 threadBlockDI( 24*NUMAMBIENTOCCLUSIONSAMPLES );
        dim3 blockGridDI  ( 120 );

        const uint sharedMemoryShade =
            threadBlockDI.x * sizeof(float3) +   //light vector   
            0u;

        computeAOIllumination < tPrimitive >
            <<< blockGridDI, threadBlockDI, sharedMemoryShade>>>(
            aStorage,
            aNormalStorage,
            rayBuffer,
            occlusionBuffer,
            aFrameBuffer,
            numAORays,
            aImageId);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Computing direct illumination failed!\n");

        cudaEventDestroy(mTrace);
    }

    HOST void cleanup()
    {
        if(mGlobalMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL( cudaFree(mGlobalMemoryPtr));
            mGlobalMemoryPtr = NULL;
            mGlobalMemorySize = 0u;
        }
        if(mAORayGeneratorMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL( cudaFree(mAORayGeneratorMemoryPtr));
            mAORayGeneratorMemoryPtr = NULL;
            mAORayGeneratorMemorySize = 0u;
        }
    }

};


#endif // AOINTEGRATOR_H_INCLUDED_D0183711_2FA1_4513_A892_14302C600A8C
