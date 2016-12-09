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

#ifndef SIMPLERAYTRAVERSER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D
#define SIMPLERAYTRAVERSER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D

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

#include "RT/Integrator/SimpleIntegrator.h"

#include "StdAfx.hpp"
#include <fstream>



template<
    class tPrimitive,
    class tPrimaryRayGenerator,
    class tAccelerationStructure,
        template <class, class, bool> class tTraverser,
    class tPrimaryIntersector,
    class tAlternativeIntersector>

class SimpleRayTraverser
{
    int* mGlobalMemoryPtr;
    int* mHostMemoryPtr;
    size_t mGlobalMemorySize;
    cudaEvent_t mStart, mTrace, mShade;
    std::string mInputRaysFileName;
public:
    typedef tPrimaryRayGenerator            t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                 t_RayBuffer;
    typedef tPrimaryIntersector             t_Intersector;
    typedef tAccelerationStructure          t_AccelerationStructure;

    t_RayBuffer rayBuffer;

    SimpleRayTraverser() :rayBuffer(t_RayBuffer(NULL)), mGlobalMemorySize(0u)
    {}

    ~SimpleRayTraverser()
    {
        cleanup();
    }

    HOST void tracePrimary(
        PrimitiveArray<tPrimitive>&     aStorage,
        t_AccelerationStructure&        aAccStruct,
        RayLoader<SimpleRayBuffer>&     aRayGenerator,
        FrameBuffer&                    aFrameBuffer
        )
    {

        const uint sharedMemoryTrace = SHARED_MEMORY_TRACE;

        const uint numRays = loadRaysFromFile();
        aRayGenerator.buffer = rayBuffer;

        MY_CUDA_SAFE_CALL(cudaMemset(mGlobalMemoryPtr, 0, sizeof(uint)));

        dim3 threadBlockTrace(RENDERTHREADSX, RENDERTHREADSY);
        dim3 blockGridTrace(RENDERBLOCKSX, RENDERBLOCKSY);

        //MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGrid", &aAccStruct, sizeof(UniformGrid)) );

        cudaEventCreate(&mStart);
        cudaEventCreate(&mTrace);

        cudaEventRecord(mStart, 0);

        trace<tPrimitive, tAccelerationStructure, t_PrimaryRayGenerator, t_RayBuffer, tTraverser, tPrimaryIntersector, false >
            << < blockGridTrace, threadBlockTrace, sharedMemoryTrace >> >(
            aStorage,
            aRayGenerator,
            aAccStruct,
            rayBuffer,
            numRays,
            mGlobalMemoryPtr);

        cudaEventRecord(mTrace, 0);
        cudaEventSynchronize(mTrace);
        MY_CUT_CHECK_ERROR("Tracing primary rays failed!\n");

        float elapsedTime;
        cudastd::logger::floatPrecision(4);
        cudaEventElapsedTime(&elapsedTime, mStart, mTrace);
        cudastd::logger::out << "Trace time:      " << elapsedTime << "ms\n";
        cudastd::logger::out << "Mrays/s:      " << (float)numRays / (1000.f * elapsedTime) << "ms\n";

        cudaEventDestroy(mStart);
        cudaEventDestroy(mTrace);

    }

    HOST void shade(
        PrimitiveArray<tPrimitive>&                     aStorage,
        VtxAttributeArray<tPrimitive, float3>           aNormalStorage,
        PrimitiveAttributeArray<PhongMaterial>          aMaterialStorage,
        FrameBuffer&                                    aFrameBuffer,
        const int                                       aImageId
        )
    {
        const uint sharedMemoryShade =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +   //rayDir
            RENDERTHREADSY * 2 * sizeof(uint) +                 //Persistent threads    
            0u;

        dim3 threadBlockShade(RENDERTHREADSX, RENDERTHREADSY);
        dim3 blockGridShade(RENDERBLOCKSX, RENDERBLOCKSY);

        cudaEventCreate(&mShade);

        simpleShade<tPrimitive>
            << < blockGridShade, threadBlockShade, sharedMemoryShade >> >(
            aStorage,
            aNormalStorage,
            aMaterialStorage,
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
        VtxAttributeArray<tPrimitive, float3>           aNormalStorage,
        PrimitiveAttributeArray<PhongMaterial>          aMaterialStorage,
        t_AccelerationStructure&                        aAccStruct,
        t_PrimaryRayGenerator&                          aRayGenerator,
        FrameBuffer&                                    aFrameBuffer,
        const int                                       aImageId
        )
    {
        tracePrimary(aStorage, aAccStruct, aRayGenerator, aFrameBuffer);

        shade(aStorage, aNormalStorage, aMaterialStorage, aFrameBuffer, aImageId);
    }
    
    HOST void setInputRaysFileName(const std::string& aInputRaysFileName)
    {
        mInputRaysFileName = aInputRaysFileName;
    }

    HOST void cleanup()
    {
        if (mGlobalMemorySize != 0u)
        {
            MY_CUDA_SAFE_CALL(cudaFreeHost(mHostMemoryPtr));
            mHostMemoryPtr = NULL;
            MY_CUDA_SAFE_CALL(cudaFree(mGlobalMemoryPtr));
            mGlobalMemoryPtr = NULL;
            mGlobalMemorySize = 0u;
        }
    }

    HOST uint loadRaysFromFile()
    {
        std::ifstream inputStream(mInputRaysFileName.c_str(), std::ios::in | std::ios::binary);
        std::string buf;

        if (inputStream.fail())
        {
            std::cerr << "Error opening .obj file\n";
            return 0;
        }

        inputStream.seekg(std::ios::beg, std::ios::end);
        uint size = (uint)inputStream.tellg();

        const uint numRays = size / (sizeof(float3) + sizeof(float3));
        const uint globalMemorySize = sizeof(uint) +    //Persistent threads
            numRays * sizeof(float3) +                   //rayOrg
            numRays * sizeof(float3) +                   //rayDir
            numRays * sizeof(float) +                   //rayT
            numRays * sizeof(uint) +                    //primitive Id
            //gRESX * gRESY * NUMOCCLUSIONSAMPLES * sizeof(float3) + //light vector
            0u;

        inputStream.seekg(0, std::ios::beg);
        float* buffer = new float[numRays * 6];
        inputStream.read((char*)buffer, size);

        MemoryManager::allocateHostDeviceArrayPair(
            (void**)&mGlobalMemoryPtr, (void**)&mHostMemoryPtr,
            globalMemorySize,
            (void**)&mGlobalMemoryPtr, (void**)&mHostMemoryPtr,
            mGlobalMemorySize);

        rayBuffer.setMemPtr(mHostMemoryPtr + 1);

        for (size_t rayId = 0; rayId < numRays; ++rayId)
        {
            float3 rayOrg = make_float3(0.f, 0.f, 0.f);
            float3 rayDir = make_float3(0.f, 0.f, 0.f);
            float  rayT = FLT_MAX;
            uint bestHit = uint(-1);

            rayOrg.x = buffer[rayId * 6 + 0];
            rayOrg.y = buffer[rayId * 6 + 1];
            rayOrg.z = buffer[rayId * 6 + 2];
            rayDir.x = buffer[rayId * 6 + 3];
            rayDir.y = buffer[rayId * 6 + 4];
            rayDir.z = buffer[rayId * 6 + 5];

            rayBuffer.storeInput(rayOrg, rayDir, rayT, bestHit, (uint)rayId, numRays);

        }

        MY_CUDA_SAFE_CALL(cudaMemcpy(mGlobalMemoryPtr, mHostMemoryPtr, globalMemorySize, cudaMemcpyHostToDevice));

        rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);
        
        delete buffer;
        inputStream.close();

        return numRays;
    }
};

#endif // SIMPLERAYTRAVERSER_H_INCLUDED_0DBBFC9C_D2F8_45B3_8700_23408B9A900D
