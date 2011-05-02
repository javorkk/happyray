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

#ifndef SIMPLEINTEGRATOR_H_DA829C75_BB8A_4034_B5FC_77B50B840948
#define SIMPLEINTEGRATOR_H_DA829C75_BB8A_4034_B5FC_77B50B840948

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Structure/FrameBuffer.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/RayBuffers.h"
#include "RT/Structure/MemoryManager.h"

#include "RT/Algorithm/RayTracingKernels.h"

template< 
    class tPrimitive,
    class tMaterialStorage,
    class tInputBuffer >
GLOBAL void simpleShade(
        PrimitiveArray<tPrimitive>      aStorage,
        PrimitiveAttributeArray<tPrimitive, float3>      aNormalStorage,
        tInputBuffer                    aInputBuffer,
        FrameBuffer                     oFrameBuffer,
        const int                       aImageId,
        int*                            aGlobalMemoryPtr)
{
    extern SHARED uint sharedMem[];

    float3* rayOrg = (float3*)(sharedMem);
    float3* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

    const uint aNumRays = oFrameBuffer.resX * oFrameBuffer.resY;

    for(uint myRayIndex = globalThreadId1D(); myRayIndex < aNumRays;
        myRayIndex += numThreads())
    {
        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT;
        uint  bestHit;

        aInputBuffer.load(rayOrg[threadId1D()], rayDir[threadId1D()],
            rayT, bestHit, myRayIndex, aNumRays);
        //////////////////////////////////////////////////////////////////////////


        float3 oRadiance;
        if (rayT < FLT_MAX)
        {
            //oRadiance = rep(rayT);
            tPrimitive prim = aStorage[bestHit];
            float3& vert0 = prim.vtx[0];
            float3& vert1 = prim.vtx[1];
            float3& vert2 = prim.vtx[2];

            float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

            //Compute barycentric coordinates
            vert0 = vert0 - rayOrg[threadId1D()];
            vert1 = vert1 - rayOrg[threadId1D()];
            vert2 = vert2 - rayOrg[threadId1D()];

            float3 n0 = vert1 % vert2;
            float3 n1 = vert2 % vert0;

            float twiceSabc_RCP = lenRCP(realNormal);
            float u = len(n0) * twiceSabc_RCP;
            float v = len(n1) * twiceSabc_RCP;

            AttribStruct<tPrimitive, float3> normals;
            normals = aNormalStorage[bestHit];
            float3& normal0 = normals.data[0];
            float3& normal1 = normals.data[1];
            float3& normal2 = normals.data[2];

            float3 normal = ~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);

            float3 diffReflectance;
            diffReflectance.x = u;//1.f;// M_PI_RCP; //u;
            diffReflectance.y = v;//1.f;//M_PI_RCP; //v;
            diffReflectance.z = 1.f-u-v;//1.f;//M_PI_RCP; //1.f - u - v;

            oRadiance =  diffReflectance * fmaxf(0.f, fabsf(dot(-normal,~rayDir[threadId1D()])));

        }
        else
        {
            oRadiance.x = 0.15f;
            oRadiance.y = 0.2f;
            oRadiance.z = 0.3f;
        }

        float newSampleWeight = 1.f / (float)(aImageId + 1);
        float oldSamplesWeight = 1.f - newSampleWeight;

        oFrameBuffer.deviceData[myRayIndex] = 
            oFrameBuffer.deviceData[myRayIndex] * oldSamplesWeight +
            oRadiance * newSampleWeight;

    }
}


template<
    class tPrimitive,
    class tPrimaryRayGenerator,
    class tAccelerationStructure,
    class tPrimaryIntersector,
    class tAlternativeIntersector>

class SimpleIntegrator
{
    int* mGlobalMemoryPtr;
    size_t mGlobalMemorySize;
    cudaEvent_t mTrace, mShade;
public:
    typedef tPrimaryRayGenerator            t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                 t_RayBuffer;
    typedef tPrimaryIntersector             t_Intersector;
    typedef tAccelerationStructure          t_AccelerationStructure;

    t_RayBuffer rayBuffer;

    SimpleIntegrator():rayBuffer(t_RayBuffer(NULL)), mGlobalMemorySize(0u)
    {}

    ~SimpleIntegrator()
    {
        if(mGlobalMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL( cudaFree(mGlobalMemoryPtr));
        }

    }

    HOST void tracePrimary(
        PrimitiveArray<tPrimitive>&     aStorage,
        t_AccelerationStructure&        aAccStruct,
        t_PrimaryRayGenerator&          aRayGenerator,
        FrameBuffer&                    aFrameBuffer
        )
    {

        const uint sharedMemoryTrace = SHARED_MEMORY_TRACE;

        const uint numRays = aFrameBuffer.resX * aFrameBuffer.resY;
        const uint globalMemorySize = sizeof(uint) +    //Persistent threads
            numRays * sizeof(float3) +                   //rayOrg
            numRays * sizeof(float3) +                   //rayDir
            numRays * sizeof(float) +                   //rayT
            numRays * sizeof(uint) +                    //primitive Id
            //gRESX * gRESY * NUMOCCLUSIONSAMPLES * sizeof(float3) + //light vector
            0u;

        MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);

        MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );

        dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );

        rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);
        //MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGrid", &aAccStruct, sizeof(UniformGrid)) );

        cudaEventCreate(&mTrace);

        trace<tPrimitive, t_PrimaryRayGenerator, t_RayBuffer, tPrimaryIntersector >
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>(
            aStorage,
            aRayGenerator,
            aAccStruct,
            rayBuffer,
            numRays,
            mGlobalMemoryPtr);
        
        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing primary rays failed!\n");
        cudaEventDestroy(mTrace);

    }

    HOST void shade(
        PrimitiveArray<tPrimitive>&                     aStorage,
        PrimitiveAttributeArray<tPrimitive, float3>     aNormalStorage,
        FrameBuffer&                                    aFrameBuffer,
        const int                                       aImageId
        )
    {
        const uint sharedMemoryShade =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +   //rayDir
            RENDERTHREADSY * 2 * sizeof(uint) +                 //Persistent threads    
            0u;

        dim3 threadBlockShade( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridShade  ( RENDERBLOCKSX, RENDERBLOCKSY );

        cudaEventCreate(&mShade);

        simpleShade<
            tPrimitive,
            t_RayBuffer>
            <<< blockGridShade, threadBlockShade, sharedMemoryShade>>>(
            aStorage,
            aNormalStorage,
            rayBuffer,
            aFrameBuffer,
            aImageId,
            mGlobalMemoryPtr);
        
        cudaEventRecord(mShade, 0);
        cudaEventSynchronize(mShade);
        MY_CUT_CHECK_ERROR("Simple shading kernel failed.\n");
        cudaEventDestroy(mShade);
    }

  
    HOST void integrate(
        PrimitiveArray<tPrimitive>&                     aStorage,
        PrimitiveAttributeArray<tPrimitive, float3>     aNormalStorage,
        t_AccelerationStructure&                        aAccStruct,
        t_PrimaryRayGenerator&                          aRayGenerator,
        FrameBuffer&                                    aFrameBuffer,
        const int                                       aImageId
        )
    {
        tracePrimary(aStorage, aAccStruct, aRayGenerator, aFrameBuffer);

        shade(aStorage, aNormalStorage, aFrameBuffer, aImageId);
    }

};


#endif // SIMPLEINTEGRATOR_H_DA829C75_BB8A_4034_B5FC_77B50B840948
