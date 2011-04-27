#ifdef _MSC_VER
#pragma once
#endif

#ifndef TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34
#define TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34


#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

#include "RT/Structure/TwoLevelGrid.h"
#include "RT/Structure/PrimitiveArray.h"

/////////////////////////////////////////////////////////////////
//Two-Level Grid Traversal Kernel
/////////////////////////////////////////////////////////////////

#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)


template<class tPrimitive, class tRayGenerator, class tRayBuffer,
class tIntersector>
    GLOBAL void trace(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    tRayGenerator               aRayGenerator,
    TwoLevelGrid                dcGrid,
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
            myRayIndex = aNumRays;
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
        float tEntry;
        //////////////////////////////////////////////////////////////////////////

        if (traversalFlag)
        {
            //////////////////////////////////////////////////////////////////////////
            //ray/box intersection test
            const float3 t1 = (dcGrid.vtx[0] - rayOrg) * rayDirRCP;
            float3 tFar = (dcGrid.vtx[1] - rayOrg) * rayDirRCP;

            const float3 tNear = min(t1, tFar);
            tFar = max(t1, tFar);

            tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);

            //BBox bounds = BBoxExtractor<TwoLevelGrid>::get(dcGrid);
            //bounds.fastClip(rayOrg, rayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////
            tEntry = fmaxf(0.f, tEntry);
            const float3 entryPt = rayOrg + rep(tEntry) / rayDirRCP;

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
        }
        //////////////////////////////////////////////////////////////////////////
        //Outer loop
        while (ANY(traversalFlag))
        {
            TwoLevelGrid::t_Cell cell;

            if (traversalFlag)
            {
                cell = dcGrid.getCell(cellId[0], cellId[1], cellId[2]);
            }

            bool secondLvlFlag = traversalFlag;// && cell.notEmpty();
            float3 tMaxLvl2; //TODO: float [3]
            int cellIdLvl2[3];
            float3 subCellSize;//TODO: remove


            if (secondLvlFlag)
            {

                const float3 entryPoint = rayOrg + 
                    fastDivide(rep(tEntry), rayDirRCP);

                float3 cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const float3 subCellSizeRCP = dcGrid.getCellSizeRCP() * cellRes; 
                float3 minBound;
                minBound.x = (float)cellId[0] * dcGrid.getCellSize().x
                    + dcGrid.vtx[0].x;
                minBound.y = (float)cellId[1] * dcGrid.getCellSize().y
                    + dcGrid.vtx[0].y;
                minBound.z = (float)cellId[2] * dcGrid.getCellSize().z
                    + dcGrid.vtx[0].z;

                float3 cellIdf;
                cellIdf.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                cellIdf.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                cellIdf.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subCellSize.x = fastDivide(1.f, subCellSizeRCP.x);
                subCellSize.y = fastDivide(1.f, subCellSizeRCP.y);
                subCellSize.z = fastDivide(1.f, subCellSizeRCP.z);

                cellIdf = min(cellRes - rep(1.f), 
                    max(cellIdf, rep(0.f)));

                tMaxLvl2.x = ((cellIdf.x + 
                    ((rayDirRCP.x > 0.f) ? 1.f : 0.f ))
                    * subCellSize.x + minBound.x
                    - rayOrg.x) * rayDirRCP.x;
                tMaxLvl2.y = ((cellIdf.y + 
                    ((rayDirRCP.y > 0.f) ? 1.f : 0.f ))
                    * subCellSize.y + minBound.y
                    - rayOrg.y) * rayDirRCP.y;
                tMaxLvl2.z = ((cellIdf.z +
                    ((rayDirRCP.z > 0.f) ? 1.f : 0.f ))
                    * subCellSize.z + minBound.z
                    - rayOrg.z) * rayDirRCP.z;

                cellIdLvl2[0] = static_cast<int>(cellIdf.x);
                cellIdLvl2[1] = static_cast<int>(cellIdf.y);
                cellIdLvl2[2] = static_cast<int>(cellIdf.z);

            }

            /////////////////////////////////////////////////////////////////////////
            //Inner loop
            while (ANY(secondLvlFlag))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (secondLvlFlag)
                {
                    cellRange = dcGrid.leaves[cell.getLeafRangeBegin()
                            + cellIdLvl2[0] + cell[0] * cellIdLvl2[1]
                            + cell[0] * cell[1] * cellIdLvl2[2]];
                }

                intersector(rayOrg, rayDirRCP, rayT, bestHit,
                    cellRange, dcGrid.primitives, aPrimitiveArray, sharedMemNew);

                if (secondLvlFlag)
                {
                    secondLvlFlag = secondLvlFlag && (
                        rayT >  tMaxLvl2.x
                    ||  rayT >  tMaxLvl2.y
                    ||  rayT >  tMaxLvl2.z); //replace with single check tMaxLvl2[tMinDimension]

                    /////////////////////////////////////////////////////////////////////////
                    //Traverse to next leaf
                    const int tMinDimension =
                        MIN_DIMENSION(tMaxLvl2.x, tMaxLvl2.y, tMaxLvl2.z);


                    cellIdLvl2[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                    toPtr(tMaxLvl2)[tMinDimension] +=
                        toPtr(subCellSize)[tMinDimension] * fabsf(toPtr(rayDirRCP)[tMinDimension]);
                    
                    secondLvlFlag = secondLvlFlag &&
                         cellIdLvl2[tMinDimension] != cell[tMinDimension]
                         && cellIdLvl2[tMinDimension] != -1;
                    //////////////////////////////////////////////////////////////////////////
                }
            }//end traversal inner loop

            if(traversalFlag)
            {
                traversalFlag = traversalFlag && (
                    rayT >  tMax.x
                    ||  rayT >  tMax.y
                    ||  rayT >  tMax.z); //replace with single check rayT > tEntry

                /////////////////////////////////////////////////////////////////////////
                //Traverse to next cell
                const int tMinDimension =
                    MIN_DIMENSION(tMax.x, tMax.y, tMax.z);

                tEntry = toPtr(tMax)[tMinDimension];
                cellId[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                toPtr(tMax)[tMinDimension] += toPtr(dcGrid.getCellSize())[tMinDimension] * 
                    fabsf(toPtr(rayDirRCP)[tMinDimension]);

                traversalFlag = traversalFlag &&
                    cellId[tMinDimension] != dcGrid.res[tMinDimension]
                && cellId[tMinDimension] != -1;
            }
        }
        //end traversal outer loop
        ////////////////////////////////////////////////////////////////////////////

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


#undef  MIN_DIMENSION

#endif // TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34

//struct __align__(16) TraversalState
//{
//    float3 tMax;
//    float3 cellId;
//    float tEntry;
//};
//
//struct __align__(16) SmallTraversalState
//{
//    float3 tMax;
//    float3 cellId;
//};
//
//
//DEVICE bool traverseInit(
//    const float3&                aRayOrg,
//    const float3&                aRayDirRCP,
//    const TwoLevelGrid&    aGridParams,
//    TraversalState&             oState)
//{
//    //////////////////////////////////////////////////////////////////////////
//    //ray/box intersection test
//    const float3 t1 = (aGridParams.vtx[0] - aRayOrg) * aRayDirRCP;
//    float3 tFar = (aGridParams.vtx[1] - aRayOrg) * aRayDirRCP;
//
//    const float3 tNear = min(t1, tFar);
//    tFar = max(t1, tFar);
//
//    oState.tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
//    const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);
//
//    if (tExit <= oState.tEntry || tExit < 0.f)
//    {
//        return false;
//    }
//    //end ray/box intersection test
//    //////////////////////////////////////////////////////////////////////////
//
//    oState.tEntry = fmaxf(0.f, oState.tEntry);
//    const float3 entryPt = aRayOrg +
//        fastDivide(rep(oState.tEntry + EPS) , aRayDirRCP);
//
//    oState.cellId = 
//        (entryPt - aGridParams.vtx[0]) * aGridParams.getCellSizeRCP();
//
//    oState.cellId.x = floorf(oState.cellId.x);
//    oState.cellId.y = floorf(oState.cellId.y);
//    oState.cellId.z = floorf(oState.cellId.z);
//
//    float3 tmp;
//    tmp.x = (aRayDirRCP.x > 0.f) ? 1.f : 0.f;
//    tmp.y = (aRayDirRCP.y > 0.f) ? 1.f : 0.f;
//    tmp.z = (aRayDirRCP.z > 0.f) ? 1.f : 0.f;
//
//    oState.tMax = ((oState.cellId + tmp) * aGridParams.getCellSize()
//        + aGridParams.vtx[0] - aRayOrg) * aRayDirRCP;
//
//    return (oState.cellId.x != ((aRayDirRCP.x > 0.f) ? aGridParams.res[0] : -1)) 
//        && (oState.cellId.y != ((aRayDirRCP.y > 0.f) ? aGridParams.res[1] : -1))
//        && (oState.cellId.z != ((aRayDirRCP.z > 0.f) ? aGridParams.res[2] : -1));
//}
//
//DEVICE bool traverseMacroCell(
//    const float3&                aRayDirRCP,
//    const TwoLevelGrid&    aGridParams,
//    TraversalState&             oState)
//{
//
//    const int tMinDimension =
//        MIN_DIMENSION(oState.tMax.x, oState.tMax.y, oState.tMax.z);
//
//
//
//    oState.tEntry = toPtr(oState.tMax)[tMinDimension];
//    toPtr(oState.cellId)[tMinDimension] += (toPtr(aRayDirRCP)[tMinDimension] > 0.f) ? 1.f : -1.f;
//    toPtr(oState.tMax)[tMinDimension] += toPtr(aGridParams.cellSize)[tMinDimension] *
//        fabsf(toPtr(aRayDirRCP)[tMinDimension]);
//
//    return
//        !(fabsf(toPtr(oState.cellId)[tMinDimension]  - (float)aGridParams.res[tMinDimension]) < 0.1f 
//        || fabsf(toPtr(oState.cellId)[tMinDimension] + 1.f) < 0.1f);
//}
//
//DEVICE bool traverseCell(
//    const float3&                aRayDirRCP,
//    const float3&                aSubCellSize,
//    const TwoLevelGrid::t_Cell& aCellParams,
//    SmallTraversalState&        oState)
//{
//
//
//    const int tMinDimension =
//        MIN_DIMENSION(oState.tMax.x, oState.tMax.y, oState.tMax.z);
//
//
//
//    toPtr(oState.cellId)[tMinDimension] += (toPtr(aRayDirRCP)[tMinDimension] > 0.f) ? 1.f : -1.f;
//    toPtr(oState.tMax)[tMinDimension] += toPtr(aSubCellSize)[tMinDimension] * fabsf(toPtr(aRayDirRCP)[tMinDimension]);
//
//    return
//        !(fabsf(toPtr(oState.cellId)[tMinDimension]  - (float)aCellParams[tMinDimension]) < 0.1f 
//        || fabsf(toPtr(oState.cellId)[tMinDimension] + 1.f) < 0.1f);
//}
//        TraversalState  aState;
//
//        if (traversalFlag)
//        {
//            traversalFlag &= 
//                traverseInit(rayOrg, rayDirRCP, dcGrid, aState);
//        }
//        //////////////////////////////////////////////////////////////////////////
//        //Traversal loop
//        while (ANY(traversalFlag))
//        {
//            TwoLevelGrid::t_Cell cell;
//
//            if (traversalFlag)
//            {
//                cell = dcGrid.getCell((uint) aState.cellId.x,
//                    (uint)aState.cellId.y, (uint)aState.cellId.z);
//            }
//
//
//            bool secondLvlFlag = traversalFlag && cell.notEmpty();
//
//            SmallTraversalState subState;
//            float3 subCellSize;
//
//            if (secondLvlFlag)
//            {
//
//                const float3 entryPoint = rayOrg + 
//                    fastDivide(rep(aState.tEntry), rayDirRCP);
//
//                float3 cellRes;
//                cellRes.x = (float)cell[0];
//                cellRes.y = (float)cell[1];
//                cellRes.z = (float)cell[2];
//
//                const float3 subCellSizeRCP = dcGrid.getCellSizeRCP() * cellRes; 
//                const float3 minBound = aState.cellId * dcGrid.getCellSize()
//                    + dcGrid.vtx[0];
//
//                subState.cellId.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
//                subState.cellId.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
//                subState.cellId.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);
//
//                subCellSize.x = fastDivide(1.f, subCellSizeRCP.x);
//                subCellSize.y = fastDivide(1.f, subCellSizeRCP.y);
//                subCellSize.z = fastDivide(1.f, subCellSizeRCP.z);
//
//                subState.cellId = min(cellRes - rep(1.f), 
//                    max(subState.cellId, rep(0.f)));
//
//                subState.tMax.x = ((subState.cellId.x + 
//                    ((rayDirRCP.x > 0.f) ? 1 : 0 ))
//                    * subCellSize.x + minBound.x
//                    - rayOrg.x) * rayDirRCP.x;
//                subState.tMax.y = ((subState.cellId.y + 
//                    ((rayDirRCP.y > 0.f) ? 1 : 0 ))
//                    * subCellSize.y + minBound.y
//                    - rayOrg.y) * rayDirRCP.y;
//                subState.tMax.z = ((subState.cellId.z +
//                    ((rayDirRCP.z > 0.f) ? 1 : 0 ))
//                    * subCellSize.z + minBound.z
//                    - rayOrg.z) * rayDirRCP.z;
//            }
//
//            while (ANY(secondLvlFlag))
//            {
//                uint2 cellRange = make_uint2(0u, 0u);
//
//                if (secondLvlFlag)
//                {
//                    //NOTE: Do not normalize coordinates!
//                    //cellRange = tex1Dfetch(texLeafCells, cell.getLeafRangeBegin()
//                    //    + (uint)(subState.cellId.x + (float)cell[0] * subState.cellId.y
//                    //    + (float)cell[0] * (float)cell[1] * subState.cellId.z));
//                    cellRange = dcGrid.leaves[cell.getLeafRangeBegin()
//                            + (uint)(subState.cellId.x + (float)cell[0] * subState.cellId.y
//                            + (float)cell[0] * (float)cell[1] * subState.cellId.z)];
//                }
//
//#ifdef GATHERSTATISTICS
//                oRadiance.z += (float)(cellRange.y - cellRange.x);
//#endif
//                intersector(rayOrg, rayDirRCP, rayT, bestHit,
//                    cellRange, dcGrid.primitives, aPrimitiveArray, sharedMemNew);
//
//                const bool keepRayActive =
//                    rayT > subState.tMax.x
//                    ||  rayT > subState.tMax.y
//                    ||  rayT > subState.tMax.z;
//
//                if (!keepRayActive)
//                    secondLvlFlag = false;
//
//                if (secondLvlFlag)
//                {
//                    secondLvlFlag &=
//                        traverseCell(rayDirRCP, subCellSize, cell, subState);
//
//                }
//
//            }
//
//            const bool keepRayActive =
//                rayT > aState.tMax.x
//                ||  rayT > aState.tMax.y
//                ||  rayT > aState.tMax.z;
//
//                if (!keepRayActive)
//                    traversalFlag = false;
//
//            if (traversalFlag)
//            {
//                traversalFlag &=
//                    traverseMacroCell(rayDirRCP, dcGrid, aState);
//            }
//
//        }
//        //end of traversal loop
//        //////////////////////////////////////////////////////////////////////////

