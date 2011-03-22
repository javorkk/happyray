#ifdef _MSC_VER
#pragma once
#endif

#ifndef RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332
#define RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

#include "RT/Structure/UniformGrid.h"
#include "RT/Structure/PrimitiveArray.h"

/////////////////////////////////////////////////////////////////
//Uniform Grid Traversal Kernel
/////////////////////////////////////////////////////////////////

#define RENDERTHREADSX  32
#define RENDERTHREADSY  4
#define RENDERBLOCKSX   120
#define RENDERBLOCKSY   1
#define BATCHSIZE       96

#define SHARED_MEMORY_TRACE                                                    \
    RENDERTHREADSX * RENDERTHREADSY * sizeof(float3) +                          \
    RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2 * sizeof(uint)              \
    /*End Macro*/

//DEVICE_NO_INLINE CONSTANT UniformGrid dcGrid;

template<class tPrimitive, class tRayGenerator, class tRayBuffer,
class tIntersector>
    GLOBAL void trace(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    tRayGenerator               aRayGenerator,
    UniformGrid                 dcGrid,
    tRayBuffer                  oBuffer,
    uint                        aNumRays,
    int*                        aGlobalMemoryPtr
    )
{
    tIntersector intersector;

    extern SHARED uint sharedMem[];
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
            myRayIndex = aNumRays - 1;
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

        float rayT  = aRayGenerator(rayOrg, rayDirRCP, myRayIndex, aNumRays);
        rayDirRCP.x = 1.f / rayDirRCP.x;
        rayDirRCP.y = 1.f / rayDirRCP.y;
        rayDirRCP.z = 1.f / rayDirRCP.z;

        uint  bestHit = aPrimitiveArray.numPrimitives;
        //////////////////////////////////////////////////////////////////////////
        //Traversal State
        bool traversalFlag = (rayT >= 0.f) && myRayIndex < aNumRays;
        float3 tMax;
        int cellId[3];
        //////////////////////////////////////////////////////////////////////////

        if (traversalFlag)
        {
            //////////////////////////////////////////////////////////////////////////
            //ray/box intersection test
            const float3 t1 = (dcGrid.vtx[0] - rayOrg) * rayDirRCP;
            float3 tFar = (dcGrid.vtx[1] - rayOrg) * rayDirRCP;

            const float3 tNear = min(t1, tFar);
            tFar = max(t1, tFar);

            const float tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);

            //BBox bounds = BBoxExtractor<UniformGrid>::get(dcGrid);
            //bounds.fastClip(rayOrg, rayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////

            const float3 entryPt = (tEntry >= 0.f) ?
                rayOrg + rep(tEntry + EPS) / rayDirRCP : rayOrg;

            float3 cellIdf = 
                (entryPt - dcGrid.vtx[0]) * dcGrid.getCellSizeRCP();

            cellIdf.x = floorf(cellIdf.x);
            cellIdf.y = floorf(cellIdf.y);
            cellIdf.z = floorf(cellIdf.z);

            float3 tmp;
            tmp.x = (rayDirRCP.x > 0.f) ? 1.f : 0.f;
            tmp.y = (rayDirRCP.y > 0.f) ? 1.f : 0.f;
            tmp.z = (rayDirRCP.z > 0.f) ? 1.f : 0.f;

            tMax = ((cellIdf + tmp) * dcGrid.getCellSize() + dcGrid.vtx[0] - rayOrg) * rayDirRCP;

            cellId[0] = static_cast<int>(cellIdf.x);
            cellId[1] = static_cast<int>(cellIdf.y);
            cellId[2] = static_cast<int>(cellIdf.z);

            traversalFlag = traversalFlag && (  
                    (cellId[0] != ((rayDirRCP.x > 0.f) ? dcGrid.res[0] : -1)) 
                &&  (cellId[1] != ((rayDirRCP.y > 0.f) ? dcGrid.res[1] : -1))
                &&  (cellId[2] != ((rayDirRCP.z > 0.f) ? dcGrid.res[2] : -1)) 
                );

            //if(traversalFlag) rayT = 0.1f;//(cellId.x + cellId.y + cellId.z) / (dcGrid.res[0] + dcGrid.res[1] + dcGrid.res[2]);
            //else rayT = 0.9f;
            //traversalFlag = false;
        }
        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                cellRange = //make_uint2(0u, 0u);
                    dcGrid.getCell(cellId[0], cellId[1], cellId[2]);
                //cellRange =  make_uint2(0u, 0u);
            }

            intersector(rayOrg, rayDirRCP, rayT, bestHit,
                cellRange, dcGrid.primitives, aPrimitiveArray, sharedMemNew);

            if (traversalFlag)
            {
                traversalFlag = traversalFlag && (
                    rayT >  tMax.x
                    ||  rayT >  tMax.y
                    ||  rayT >  tMax.z);


                /////////////////////////////////////////////////////////////////////////
                //Traverse to next cell
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

                const int tMinDimension =
                    MIN_DIMENSION(tMax.x, tMax.y, tMax.z);

#undef  MIN_DIMENSION
                //if(tMinDimension == 0)
                //{
                //    cellId.x += (rayDirRCP.x > 0.f) ? 1.f : -1.f;
                //    tMax.x += dcGrid.getCellSize().x * fabsf(rayDirRCP.x);
                //    traversalFlag = (count-- == 0u) && traversalFlag &&
                //        (fabsf(cellId.x  - (float)dcGrid.res[tMinDimension]) > 0.1f 
                //        && fabsf(cellId.x + 1.f) > 0.1f);
                //}
                //else if (tMinDimension == 1)
                //{
                //    cellId.y += (rayDirRCP.y > 0.f) ? 1.f : -1.f;
                //    tMax.y += dcGrid.getCellSize().y * fabsf(rayDirRCP.y);
                //    traversalFlag = (count-- == 0u) && traversalFlag &&
                //        (fabsf(cellId.y  - (float)dcGrid.res[tMinDimension]) > 0.1f 
                //        && fabsf(cellId.y + 1.f) > 0.1f);
                //}
                //else
                //{
                //    cellId.z += (rayDirRCP.z > 0.f) ? 1.f : -1.f;
                //    tMax.z += dcGrid.getCellSize().z * fabsf(rayDirRCP.z);
                //    traversalFlag = (count-- == 0u) && traversalFlag &&
                //        (fabsf(cellId.z  - (float)dcGrid.res[tMinDimension]) > 0.1f 
                //        && fabsf(cellId.z + 1.f) > 0.1f);
                //}

                cellId[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                toPtr(tMax)[tMinDimension] += toPtr(dcGrid.getCellSize())[tMinDimension] * 
                    fabsf(toPtr(rayDirRCP)[tMinDimension]);

                traversalFlag = traversalFlag &&
                    cellId[tMinDimension] != dcGrid.res[tMinDimension]
                    && cellId[tMinDimension] != -1;
                //////////////////////////////////////////////////////////////////////////
            }
        }
        //end traversal loop
        //////////////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////////////
        //Output
        float3 rayDir;
        rayDir.x = 1.f / rayDirRCP.x;
        rayDir.y = 1.f / rayDirRCP.y;
        rayDir.z = 1.f / rayDirRCP.z;

        if(rayT < FLT_MAX)
            bestHit = dcGrid.primitives[bestHit];

        oBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, aNumRays);
        //////////////////////////////////////////////////////////////////////////

    }
}

#endif // RAYTRACINGKERNELS_H_0927D9A2_F4E0_41DF_9EFF_944EF97FA332
