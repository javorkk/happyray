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

#ifndef AORAYEXPORTER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D
#define AORAYEXPORTER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D

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

#include "RT/Integrator/AOIntegrator.h"

#include "StdAfx.hpp"
#include <fstream>

//////////////////////////////////////////////////////////////////////////////////////////
//Ambient Occlusion Ray Dump Kernel
//////////////////////////////////////////////////////////////////////////////////////////

template<class tRayGenerator>
GLOBAL void dumpRays(
    tRayGenerator               aRayGenerator,
    float3*                     oBuffer,
    uint                        aNumRays
    )
{
    for (uint myRayIndex = globalThreadId1D(); myRayIndex < aNumRays; myRayIndex += numThreads())
    {

        float3 rayOrg;
        float3 rayDir;

        float rayT = aRayGenerator(rayOrg, rayDir, myRayIndex, aNumRays);

        oBuffer[2 * myRayIndex] = rayOrg;
        oBuffer[2 * myRayIndex + 1] = rayDir;
    }
}

template<
    class tPrimitive,
    class tAccelerationStructure,
        template <class, class, bool> class tTraverser,
    class tPrimaryIntersector,
    class tAlternativeIntersector>

class AORayExporter
{
    int* mGlobalMemoryPtr;
    size_t mGlobalMemorySize;
    int* mAORayGeneratorMemoryPtr;
    size_t mAORayGeneratorMemorySize;
    cudaEvent_t mStart, mTrace, mShade;
    float mAlpha;
    size_t mFrameCounter;

public:
    typedef RandomPrimaryRayGenerator< GaussianPixelSampler, true > t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                                         t_RayBuffer;
    typedef tPrimaryIntersector                                     t_Intersector;
    typedef tAccelerationStructure                                  t_AccelerationStructure;
    typedef AmbientOcclusionRayGenerator < tPrimitive, NUMAMBIENTOCCLUSIONSAMPLES >  t_AORayGenerator;
    typedef AOIlluminationBuffer                                    t_AOcclusionBuffer;

    t_RayBuffer             rayBuffer;


    AORayExporter(float aAlpha = 10.f) :rayBuffer(t_RayBuffer(NULL)),
        mGlobalMemorySize(0u), mAORayGeneratorMemorySize(0u), mAlpha(aAlpha),
        mFrameCounter(0u)
    {}

    ~AORayExporter()
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
            numPixels * sizeof(float3) +                //importanceBuffer : importance
            numPixels * NUMAMBIENTOCCLUSIONSAMPLES * (sizeof(float3)) + //ambient occlusion buffer: intensity               
            0u;

        MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);
        MemoryManager::allocateDeviceArray((void**)&mAORayGeneratorMemoryPtr, t_AORayGenerator::getParametersSize(), (void**)&mAORayGeneratorMemoryPtr, mAORayGeneratorMemorySize);

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //Ray Dumping
        size_t rayDumpMemorySize =
            (numPixels + NUMAMBIENTOCCLUSIONSAMPLES * numPixels) * sizeof(float3) + //ray origin
            (numPixels + NUMAMBIENTOCCLUSIONSAMPLES * numPixels) * sizeof(float3) + //ray direction
            0u;
        float3* rayDumpDevice = NULL;
        float3* rayDumpHost = NULL;
        size_t dummySize = 0u;
        MemoryManager::allocateHostDeviceArrayPair(
            (void**)&rayDumpDevice, (void**)&rayDumpHost, rayDumpMemorySize,
            (void**)&rayDumpDevice, (void**)&rayDumpHost, dummySize);
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //primary rays
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        MY_CUDA_SAFE_CALL(cudaMemset(mGlobalMemoryPtr, 0, sizeof(uint)));
        rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);

        MY_CUT_CHECK_ERROR("Setting memory for ray tracing failed!\n");

        dim3 threadBlockTrace(RENDERTHREADSX, RENDERTHREADSY);
        dim3 blockGridTrace(RENDERBLOCKSX, RENDERBLOCKSY);

        cudaEventCreate(&mStart);
        cudaEventCreate(&mTrace);

        cudaEventRecord(mStart, 0);

        trace<tPrimitive, tAccelerationStructure, t_PrimaryRayGenerator, t_RayBuffer, tTraverser, tPrimaryIntersector, false>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace >>>(
            aStorage,
            aRayGenerator,
            aAccStruct,
            rayBuffer,
            numPixels,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing primary rays failed!\n");

        float elapsedTimePrimary;
        cudaEventElapsedTime(&elapsedTimePrimary, mStart, mTrace);

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //Ray Dumping
        dumpRays<t_PrimaryRayGenerator>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace >>>(
            aRayGenerator,
            rayDumpDevice,
            numPixels
            );
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        
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


        MY_CUDA_SAFE_CALL(cudaMemset(mGlobalMemoryPtr, 0, sizeof(uint)));
        const uint numAORays = numPixels * NUMAMBIENTOCCLUSIONSAMPLES;

        cudaEventRecord(mStart, 0);

        trace<tPrimitive, tAccelerationStructure, t_AORayGenerator, t_AOcclusionBuffer, tTraverser, tPrimaryIntersector, true>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace >>>(
            aStorage,
            ambientOcclusionRayGenerator,
            aAccStruct,
            occlusionBuffer,
            numAORays,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing ambient occlusion rays failed!\n");

        float elapsedTimeAO;
        cudaEventElapsedTime(&elapsedTimeAO, mStart, mTrace);


        cudastd::logger::out << "AO ray length:   " << mAlpha << "\n";
        cudastd::logger::floatPrecision(4);
        cudastd::logger::out << "Trace time:      " << elapsedTimePrimary + elapsedTimeAO << "ms\n";
        cudastd::logger::out << "Primary Mrays/s:    " << (float)(numPixels) / (1000.f * elapsedTimePrimary) << "\n";
        cudastd::logger::out << "AO Mrays/s:         " << (float)(NUMAMBIENTOCCLUSIONSAMPLES * numPixels) / (1000.f * elapsedTimeAO) << "\n";

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //AO illumination
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        dim3 threadBlockDI(24 * NUMAMBIENTOCCLUSIONSAMPLES);
        dim3 blockGridDI(120);

        const uint sharedMemoryShade =
            threadBlockDI.x * sizeof(float3) +   //light vector   
            0u;

        computeAOIllumination < tPrimitive >
            <<< blockGridDI, threadBlockDI, sharedMemoryShade >>>(
            aStorage,
            aNormalStorage,
            aMaterialStorage,
            rayBuffer,
            occlusionBuffer,
            aFrameBuffer,
            numAORays,
            aImageId);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Computing ambient illumination failed!\n");


        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //Ray Dumping
        dumpRays<t_AORayGenerator>
            << < blockGridTrace, threadBlockTrace, sharedMemoryTrace >> >(
            ambientOcclusionRayGenerator,
            rayDumpDevice + numPixels * 2,
            numAORays
            );

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Dumping ambient illumination rays failed!\n");
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //Ray Dumping
        MY_CUDA_SAFE_CALL(cudaMemcpy(rayDumpHost, rayDumpDevice, rayDumpMemorySize, cudaMemcpyDeviceToHost));
        cudaEventSynchronize(mTrace);

        std::string filename("ray_dump_");
        filename.append(itoa(mFrameCounter++));
        filename.append("_t_");
        filename.append(ftoa(mAlpha));
        filename.append(".rays");


        std::ofstream rayFileStream(filename.c_str(), std::ios::binary | std::ios::out);

        if (!rayFileStream)
        {
            cudastd::logger::out << "Could not open file " << filename.c_str() << " for writing!\n";
            return;
        }

        cudastd::logger::out << "Writing ray dump in " << filename.c_str() << "...";
        rayFileStream.write(reinterpret_cast<char*>(rayDumpHost), rayDumpMemorySize);
        cudastd::logger::out << "done.\n";


        rayFileStream.close();

        cudaEventSynchronize(mTrace);
        MY_CUDA_SAFE_CALL(cudaFree(rayDumpDevice));
        MY_CUDA_SAFE_CALL(cudaFreeHost(rayDumpHost));
        //////////////////////////////////////////////////////////////////////////////////////////////////////
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


#endif // AORAYEXPORTER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D
