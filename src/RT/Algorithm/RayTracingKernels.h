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

#ifndef RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332
#define RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332

#include "RT/Algorithm/UGridTraverser.h"
#include "RT/Algorithm/TLGridTraverser.h"

#define RENDERTHREADSX  32
#define RENDERTHREADSY  4
#define RENDERBLOCKSX   60
#define RENDERBLOCKSY   1
#define BATCHSIZE       96

#define SHARED_MEMORY_TRACE                                                    \
    RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +                          \
    RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2 * sizeof(uint)              \
    /*End Macro*/

extern SHARED uint sharedMem[];

template<
    class tPrimitive,
    class tAccelerationStructure,
    class tRayGenerator,
    class tRayBuffer,
    template <class, class, bool> class tTraverser,
    class tIntersector, 
    bool  taIsShadowRay >

    GLOBAL void trace(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    tRayGenerator               aRayGenerator,
    tAccelerationStructure      dcGrid,
    tRayBuffer                  oBuffer,
    uint                        aNumRays,
    int*                        aGlobalMemoryPtr
    )
{
#if __CUDA_ARCH__ >= 110
    volatile uint*  nextRayArray = sharedMem;
    volatile uint*  rayCountArray = nextRayArray + RENDERTHREADSY;

    if (threadId1DInWarp32() == 0u)
    {
        rayCountArray[warpId32()] = 0u;
    }

    volatile uint& localPoolNextRay = nextRayArray[warpId32()];
    volatile uint& localPoolRayCount = rayCountArray[warpId32()];

    while (true)
    {
        if (localPoolRayCount==0 && threadId1DInWarp32() == 0)
        {
            localPoolNextRay = atomicAdd(&aGlobalMemoryPtr[0], BATCHSIZE);
            localPoolRayCount = BATCHSIZE;
        }
        // get rays from local pool
        uint myRayIndex = localPoolNextRay + threadId1DInWarp32();
        if (ALL(myRayIndex >= aNumRays))
        {
            return;
        }

        if (myRayIndex >= aNumRays) //keep whole warp active
        {
            myRayIndex = aNumRays - 1u;
        }

        if (threadId1DInWarp32() == 0)
        {
            localPoolNextRay += WARPSIZE;
            localPoolRayCount -= WARPSIZE;
        }
#else
    for(uint myRayIndex = globalThreadId1D(); myRayIndex < aNumRays;
        myRayIndex += numThreads())
    {
#endif
        //////////////////////////////////////////////////////////////////////////
        //Initialization
        uint* sharedMemNew = sharedMem + RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2;
        float3 rayOrg;
        float3& rayDirRCP = (((float3*)sharedMemNew)[threadId1D32()]);

        //DEBUG pt 1 of 2
        //uint myValidIndex = myRayIndex;
        //myRayIndex = 980;
        
        float rayT  = aRayGenerator(rayOrg, rayDirRCP, myRayIndex, aNumRays);
        rayDirRCP.x = 1.f / rayDirRCP.x;
        rayDirRCP.y = 1.f / rayDirRCP.y;
        rayDirRCP.z = 1.f / rayDirRCP.z;

        uint  bestHit = aPrimitiveArray.numPrimitives;
        bool traversalFlag = (rayT >= 0.f) && myRayIndex < aNumRays;
        //////////////////////////////////////////////////////////////////////////

        tTraverser<tPrimitive, tIntersector, taIsShadowRay> traverse;
        traverse(aPrimitiveArray, dcGrid, rayOrg, rayDirRCP, rayT, bestHit, traversalFlag, sharedMemNew);

        //////////////////////////////////////////////////////////////////////////
        //Output
        float3 rayDir;
        rayDir.x = 1.f / rayDirRCP.x;
        rayDir.y = 1.f / rayDirRCP.y;
        rayDir.z = 1.f / rayDirRCP.z;


        if(!taIsShadowRay)
        {
            if(rayT < FLT_MAX)
                bestHit = dcGrid.primitives[bestHit];

            if (traverse.getBestHitInstance() != (uint)-1)
                oBuffer.storeBestHitInstance(traverse.getBestHitInstance(), myRayIndex, aNumRays);
        }

        //DEBUG pt 2 of 2
        //myRayIndex = myValidIndex;

        oBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, aNumRays);        
        //////////////////////////////////////////////////////////////////////////
    }
}

#endif // RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332
