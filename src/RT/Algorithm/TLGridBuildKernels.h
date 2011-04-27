#ifdef _MSC_VER
#pragma once
#endif

#ifndef TLGRIDBUILDKERNELS_H_21EA95C7_8004_4EA0_A7A8_8E377C62A7AA
#define TLGRIDBUILDKERNELS_H_21EA95C7_8004_4EA0_A7A8_8E377C62A7AA

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Structure/TwoLevelGrid.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"

//Computes number of second-level-cells of a gird
//A cell=(x,y,z) is mapped to a thread like this:
//  x = threadIdx.x
//  y = blockIdx.x
//  z = blcokIdx.y
//The input cells are modified for rendering purposes
template<bool taExternLeafFlag>
GLOBAL void countLeafLevelCells(
    const float3    aCellSize,
    const float     aDensity,
    cudaPitchedPtr  oTopLevelCells,
    uint*           oCellCounts,
    char*           aLeafFlagInv = NULL
    )
{
    const float cellVolume      = aCellSize.x * aCellSize.y * aCellSize.z;
    const float lambda          = aDensity;

    uint cellId = globalThreadId1D();
    const uint2 cellRange = *((uint2*)((char*)oTopLevelCells.ptr 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize) + threadIdx.x);

    const uint numCellRefs = cellRange.y - cellRange.x;
    const float magicConstant  = 
        powf(lambda * static_cast<float>(numCellRefs) / cellVolume, 0.3333333f);

    //const float3 res = float3::min(float3::rep(255.f), 
    //    float3::max(float3::rep(1.f), aCellSize * magicConstant));
    const float3 res = aCellSize * magicConstant;

    bool isLeaf;
    if (taExternLeafFlag)
    {
        isLeaf = aLeafFlagInv[threadIdx.x + blockIdx.x * blockSize() +
            blockIdx.y * blockSize() * gridDim.x] == 0
            || numCellRefs <= 16u;
    }
    else
    {
        isLeaf = numCellRefs <= 16u;
    }

    const uint resX = (isLeaf) ? 1u : static_cast<uint>(res.x);
    const uint resY = (isLeaf) ? 1u : static_cast<uint>(res.y);
    const uint resZ = (isLeaf) ? 1u : static_cast<uint>(res.z);

    //number of cells
    oCellCounts[cellId] = resX * resY * resZ;
    //////////////////////////////////////////////////////////////////////////
    //DEBUG
    //if (cellRange.x > cellRange.y)
    //{
    //    printf("Thread %d cellRange.x: %d, cellRange.y: %d \n", globalThreadId1D(), cellRange.x, cellRange.y);
    //}
    //////////////////////////////////////////////////////////////////////////

    //prepare top level cell for rendering
    TwoLevelGridCell cell;
    cell.clear();

    cell.setNotLeaf();
    if (isLeaf)
    {
        cell.setLeaf();
    }

    cell.setNotEmpty();
    if (numCellRefs == 0)
    {
        cell.setEmpty();
    }


    cell.setX(resX);
    cell.setY(resY);
    cell.setZ(resZ);
    cell.setLeafRangeBegin(cellId);

    *((TwoLevelGridCell*)((char*)oTopLevelCells.ptr
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize)
        + threadIdx.x) = cell;

}

//Writes where the array of sub-cells starts for each top level cell
//A cell=(x,y,z) is mapped to a thread like this:
//  x = threadIdx.x
//  y = blockIdx.x
//  z = blcokIdx.y
//This completes the preparation of the top level cells for rendering
GLOBAL void prepareTopLevelCellRanges(             
    uint*             aCellCounts,
    cudaPitchedPtr    oTopLevelCells
    )
{
    uint cellId = globalThreadId1D();

    const uint cellCount = aCellCounts[cellId];

    TwoLevelGridCell* cell = ((TwoLevelGridCell*)((char*)oTopLevelCells.ptr
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize)
        + threadIdx.x);

    cell->setLeafRangeBegin(cellCount);

}

template<class tPrimitive, template<class> class tPrimitiveArray, int taBlockSize>
GLOBAL void countLeafLevelPairs(
    tPrimitiveArray<tPrimitive>   aPrimitiveArray,
    const uint        aNumTopLevelRefs,
    const uint2*      aTopLevelSortedPairs,
    cudaPitchedPtr    aTopLevelCells,
    //const float3       aGridRes,
    const uint        aGridResX,
    const uint        aGridResY,
    const uint        aGridResZ,
    const float3       aBoundsMin,
    const float3       aCellSize,
    uint*             oRefCounts
    //////////////////////////////////////////////////////////////////////////
    //DEBUG
    //, uint*             debugInfo
    //////////////////////////////////////////////////////////////////////////
    )
{
    extern SHARED uint shMem[];
    shMem[threadId1D()] = 0u;

    //uint* numTopLevelCells = shMem + blockSize() + threadId1D();
    uint numTopLevelCells = aGridResX * aGridResY * aGridResZ;

    for(uint refId = globalThreadId1D(); refId < aNumTopLevelRefs; refId += numThreads())
    {
        const uint2 indexPair = aTopLevelSortedPairs[refId];

        if (indexPair.x >= /***/numTopLevelCells)
        {
            break;
        }

        const tPrimitive prim = aPrimitiveArray[indexPair.y];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);

        //////////////////////////////////////////////////////////////////////////
        //correct, but serious precision issues
        //float tmp;
        //tmp = static_cast<float>( indexPair.x ) / aGridRes.x;
        //const int topLvlCellX = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.x );
        //tmp = static_cast<float>( indexPair.x ) / (aGridRes.x * aGridRes.y);
        //const int topLvlCellY = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.y );
        //const int topLvlCellZ = static_cast<uint>( truncf(tmp) );
        //////////////////////////////////////////////////////////////////////////

        const uint topLvlCellX = indexPair.x % aGridResX;
        const uint topLvlCellY = (indexPair.x %(aGridResX * aGridResY)) / aGridResX;
        const uint topLvlCellZ = indexPair.x / (aGridResX * aGridResY);

        const TwoLevelGridCell topLvlCell = *((TwoLevelGridCell*)
            ((char*)aTopLevelCells.ptr
            + topLvlCellY * aTopLevelCells.pitch
            + topLvlCellZ * aTopLevelCells.pitch * aTopLevelCells.ysize)
            + topLvlCellX);

        float3 topLvlCellRes;
        topLvlCellRes.x = static_cast<float>(topLvlCell[0]);
        topLvlCellRes.y = static_cast<float>(topLvlCell[1]);
        topLvlCellRes.z = static_cast<float>(topLvlCell[2]);

        float3 topLvlCellOrigin;
        topLvlCellOrigin.x = static_cast<float>(topLvlCellX) * aCellSize.x + aBoundsMin.x;
        topLvlCellOrigin.y = static_cast<float>(topLvlCellY) * aCellSize.y + aBoundsMin.y;
        topLvlCellOrigin.z = static_cast<float>(topLvlCellZ) * aCellSize.z + aBoundsMin.z;

        const float3 subCellSizeRCP = topLvlCellRes / aCellSize;

        //triangleBounds.tighten(topLvlCellOrigin, topLvlCellOrigin + aCellSize);

        //float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        //minCellIdf =
        const float3 minCellIdf =
            (bounds.vtx[0] - topLvlCellOrigin + rep(-10E-5f)) * subCellSizeRCP;
        const float3 maxCellIdPlus1f =
            (bounds.vtx[1] - topLvlCellOrigin + rep(10E-5f)) * subCellSizeRCP + rep(1.f);

        const int minCellIdX =  max(0, (int)(minCellIdf.x));
        const int minCellIdY =  max(0, (int)(minCellIdf.y));
        const int minCellIdZ =  max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)topLvlCell[0], (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)topLvlCell[1], (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)topLvlCell[2], (int)(maxCellIdPlus1f.z));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //if (numCells > 0 && numCells < 4096)
        //{
        //    shMem[threadId1D()] += numCells;
        //}
        //else
        //{
        //    uint infoId = atomicAdd(&debugInfo[0], 20);
        //    if(infoId < 20 * 4)
        //    {
        //        debugInfo           [infoId +  1]   = indexPair.x        ;
        //        debugInfo           [infoId +  2]   = topLvlCellX        ;
        //        debugInfo           [infoId +  3]   = topLvlCellY        ;
        //        debugInfo           [infoId +  4]   = topLvlCellZ        ;
        //        ((float*)debugInfo) [infoId +  5]   = topLvlCellRes.x    ;
        //        ((float*)debugInfo) [infoId +  6]   = topLvlCellRes.y    ;
        //        ((float*)debugInfo) [infoId +  7]   = topLvlCellRes.z    ;
        //        ((int*)debugInfo)   [infoId +  8]   = minCellIdX         ;
        //        ((int*)debugInfo)   [infoId +  9]   = minCellIdY         ;
        //        ((int*)debugInfo)   [infoId + 10]   = minCellIdZ         ;
        //        ((int*)debugInfo)   [infoId + 11]   = maxCellIdP1X       ;
        //        ((int*)debugInfo)   [infoId + 12]   = maxCellIdP1Y       ;
        //        ((int*)debugInfo)   [infoId + 13]   = maxCellIdP1Z       ;
        //        ((float*)debugInfo) [infoId + 14]   = minCellIdf.x       ;
        //        ((float*)debugInfo) [infoId + 15]   = minCellIdf.y       ;
        //        ((float*)debugInfo) [infoId + 16]   = minCellIdf.z       ;
        //        ((float*)debugInfo) [infoId + 17]   = maxCellIdPlus1f.x  ;
        //        ((float*)debugInfo) [infoId + 18]   = maxCellIdPlus1f.y  ;
        //        ((float*)debugInfo) [infoId + 19]   = maxCellIdPlus1f.z  ;
        //        debugInfo           [infoId + 20]   = numCells           ;
        //    }
        //}
        //////////////////////////////////////////////////////////////////////////

        shMem[threadId1D()] += numCells;
    }

    SYNCTHREADS;

#if HAPPYRAY__CUDA_ARCH__ >= 120

    //reduction
    if (taBlockSize >= 512) { if (threadId1D() < 256) { shMem[threadId1D()] += shMem[threadId1D() + 256]; } SYNCTHREADS;   }
    if (taBlockSize >= 256) { if (threadId1D() < 128) { shMem[threadId1D()] += shMem[threadId1D() + 128]; } SYNCTHREADS;   }
    if (taBlockSize >= 128) { if (threadId1D() <  64) { shMem[threadId1D()] += shMem[threadId1D() +  64]; } SYNCTHREADS;   }
    if (taBlockSize >=  64) { if (threadId1D() <  32) { shMem[threadId1D()] += shMem[threadId1D() +  32]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  32) { if (threadId1D() <  16) { shMem[threadId1D()] += shMem[threadId1D() +  16]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  16) { if (threadId1D() <   8) { shMem[threadId1D()] += shMem[threadId1D() +   8]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   8) { if (threadId1D() <   4) { shMem[threadId1D()] += shMem[threadId1D() +   4]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   4) { if (threadId1D() <   2) { shMem[threadId1D()] += shMem[threadId1D() +   2]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   2) { if (threadId1D() <   1) { shMem[threadId1D()] += shMem[threadId1D() +   1]; } EMUSYNCTHREADS;}

    // write out block sum 
    if (threadId1D() == 0) oRefCounts[blockId1D()] = shMem[0];

    //if (threadId1D() == 0) printf("Block sum %d :%d \n", blockId1D(), shMem[0]);
#else

    oRefCounts[globalThreadId1D()] = shMem[threadId1D()];

#endif
}

template<class tPrimitive, template<class> class tPrimitiveArray>
GLOBAL void writeLeafLevelPairs(
    tPrimitiveArray<tPrimitive> aPrimitiveArray,
    const uint        aNumTopLevelRefs,
    const uint2*      aTopLevelSortedPairs,
    cudaPitchedPtr    aTopLevelCells,
    const uint        aNumLeafLevelCells,
    uint*             aStartId,
    //const float3       aGridRes,
    const uint        aGridResX,
    const uint        aGridResY,
    const uint        aGridResZ,
    const float3       aBoundsMin,
    const float3       aCellSize,
    uint*             oPairs
    )
{
    extern SHARED uint shMem[];


#if HAPPYRAY__CUDA_ARCH__ >= 120

    if (threadId1D() == 0)
    {
        shMem[0] = aStartId[blockId1D()];
    }

    SYNCTHREADS;

#else

    uint startPosition = aStartId[globalThreadId1D()];

#endif

    uint numTopLevelCells = aGridResX * aGridResY * aGridResZ;

    for(uint refId = globalThreadId1D(); refId < aNumTopLevelRefs; refId += numThreads())
    {
        const uint2 indexPair = aTopLevelSortedPairs[refId];

        if (indexPair.x >= numTopLevelCells)
        {
            break;
        }

        const tPrimitive prim = aPrimitiveArray[indexPair.y];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);


        //////////////////////////////////////////////////////////////////////////
        //correct, but serious precision issues
        //float tmp;
        //tmp = static_cast<float>( indexPair.x ) / aGridRes.x;
        //const int topLvlCellX = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.x );
        //tmp = static_cast<float>( indexPair.x ) / (aGridRes.x * aGridRes.y);
        //const int topLvlCellY = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.y );
        //const int topLvlCellZ = static_cast<uint>( truncf(tmp) );
        //////////////////////////////////////////////////////////////////////////

        const uint topLvlCellX = indexPair.x % aGridResX;
        const uint topLvlCellY = (indexPair.x %(aGridResX * aGridResY)) / aGridResX;
        const uint topLvlCellZ = indexPair.x / (aGridResX * aGridResY);

        const TwoLevelGridCell topLvlCell = *((TwoLevelGridCell*)
            ((char*)aTopLevelCells.ptr
            + topLvlCellY * aTopLevelCells.pitch
            + topLvlCellZ * aTopLevelCells.pitch * aTopLevelCells.ysize)
            + topLvlCellX);

        float3 topLvlCellRes;
        topLvlCellRes.x = static_cast<float>(topLvlCell[0]);
        topLvlCellRes.y = static_cast<float>(topLvlCell[1]);
        topLvlCellRes.z = static_cast<float>(topLvlCell[2]);

        float3 topLvlCellOrigin;
        topLvlCellOrigin.x = static_cast<float>(topLvlCellX) * aCellSize.x + aBoundsMin.x;
        topLvlCellOrigin.y = static_cast<float>(topLvlCellY) * aCellSize.y + aBoundsMin.y;
        topLvlCellOrigin.z = static_cast<float>(topLvlCellZ) * aCellSize.z + aBoundsMin.z;

        const float3 subCellSizeRCP = topLvlCellRes / aCellSize;

        const float3 minCellIdf =
            (bounds.vtx[0] - topLvlCellOrigin + rep(-10E-5f)) * subCellSizeRCP;
        const float3 maxCellIdPlus1f =
            (bounds.vtx[1] - topLvlCellOrigin + rep(10E-5f)) * subCellSizeRCP + rep(1.f);

        const int minCellIdX =  max(0, (int)(minCellIdf.x));
        const int minCellIdY =  max(0, (int)(minCellIdf.y));
        const int minCellIdZ =  max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)topLvlCell[0], (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)topLvlCell[1], (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)topLvlCell[2], (int)(maxCellIdPlus1f.z));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

#if HAPPYRAY__CUDA_ARCH__ >= 120
        uint nextSlot  = atomicAdd(&shMem[0], numCells);
#else
        uint nextSlot = startPosition;
        startPosition += numCells;
#endif

        for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
        {
            for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
            {
                for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                {
                    oPairs[2 * nextSlot] = x + y * topLvlCell[0] +
                        z * topLvlCell[0] * topLvlCell[1] +
                        topLvlCell.getLeafRangeBegin();
                    oPairs[2 * nextSlot + 1] = indexPair.y;
                }//end for z
            }//end for y
        }//end for x


    }//end  for(uint refId = globalThreadId1D(); ...

}


GLOBAL void prepareLeafCellRanges(
    uint*             oPrimitiveIndices,
    uint2*            aSortedPairs,
    const uint        aNumPairs,
    uint2*            oGridCells
    )
{
    extern SHARED uint shMem[];

    //padding
    if (threadId1D() == 0)
    {
        shMem[0] = 0u;
        shMem[blockSize()] = 0u;
    }

    uint *padShMem = shMem + 1;
    padShMem[threadId1D()] = 0u;

    SYNCTHREADS;


    for(int instanceId = globalThreadId1D();
        instanceId < aNumPairs + blockSize() - 1;
        instanceId += numThreads())
    {
        //load blockSize() + 2 input elements in shared memory

        SYNCTHREADS;


        if (threadId1D() == 0 && instanceId > 0u)
        {
            //padding left
            shMem[0] = aSortedPairs[instanceId - 1].x;
        }
        if (threadId1D() == 0 && instanceId + blockSize() < aNumPairs)
        {
            //padding right
            padShMem[blockSize()] = aSortedPairs[instanceId + blockSize()].x;
        }
        if (instanceId < aNumPairs)
        {
            padShMem[threadId1D()] = aSortedPairs[instanceId].x;
        }

        SYNCTHREADS;

        //Check if the two neighboring cell indices are different
        //which means that at this point there is an end of and a begin of a range

        //compare left neighbor
        if (instanceId > 0 && instanceId < aNumPairs && padShMem[threadId1D()] != shMem[threadId1D()])
        {
            //begin of range
            oGridCells[padShMem[threadId1D()]].x = instanceId;
        }

        //compare right neighbor
        if (instanceId < aNumPairs && padShMem[threadId1D()] != padShMem[threadId1D() + 1])
        {
            //end of range
            oGridCells[padShMem[threadId1D()]].y = instanceId + 1;
        }

    }//end for(uint startId = blockId1D() * blockSize()...

    SYNCTHREADS;

    //compact triangle indices from aInIndices to oFaceSoup.indices
    for(int instanceId = globalThreadId1D();
        instanceId < aNumPairs;
        instanceId += numThreads())
    {
        oPrimitiveIndices[instanceId] = aSortedPairs[instanceId].y;
    }

}

template<int taDummy>
GLOBAL void computeLeafLevelTraversalCost(
    const float3     aCellSize,
    const float     aDensity,
    cudaPitchedPtr  oTopLevelCells,
    uint*           oCellCounts
    )
{
    const float cellVolume      = aCellSize.x * aCellSize.y * aCellSize.z;
    const float lambda          = aDensity;

    uint cellId = globalThreadId1D();
    const uint2 cellRange = *((uint2*)((char*)oTopLevelCells.ptr + 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize) + threadIdx.x);

    const uint numCellRefs = cellRange.y - cellRange.x;
    const float magicConstant  = 
        powf(lambda * static_cast<float>(numCellRefs) / cellVolume, 0.3333333f);

    //const float3 res = float3::min(float3::rep(255.f), 
    //    float3::max(float3::rep(1.f), aCellSize * magicConstant));
    const float3 res = aCellSize * magicConstant;

    bool isLeaf = numCellRefs <= 16u;

    const uint resX = (isLeaf) ? 1u : static_cast<uint>(res.x);
    const uint resY = (isLeaf) ? 1u : static_cast<uint>(res.y);
    const uint resZ = (isLeaf) ? 1u : static_cast<uint>(res.z);

    //number of cells
    oCellCounts[cellId] = resX + resY + resZ;

    //prepare top level cell for rendering
    TwoLevelGridCell cell;
    cell.clear();

    cell.setNotLeaf();
    if (isLeaf)
    {
        cell.setLeaf();
    }

    cell.setNotEmpty();
    if (numCellRefs == 0)
    {
        cell.setEmpty();
    }


    cell.setX(resX);
    cell.setY(resY);
    cell.setZ(resZ);
    cell.setLeafRangeBegin(cellId);

    *((TwoLevelGridCell*)((char*)oTopLevelCells.ptr + 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize)
        + threadIdx.x) = cell;

}

template<class tPrimitive, template<class> class tPrimitiveArray, int taBlockSize>
GLOBAL void computeLeafLevelIntersectionCost(
    tPrimitiveArray<tPrimitive> aPrimitiveArray,
    const uint        aNumTopLevelRefs,
    const uint2*      aTopLevelSortedPairs,
    cudaPitchedPtr    aTopLevelCells,
    const uint        aGridResX,
    const uint        aGridResY,
    const uint        aGridResZ,
    const float3       aBoundsMin,
    const float3       aCellSize,
    float*             oCost
    )
{
    extern SHARED uint shMem[];
    float* shMemF = (float*) shMem;
    shMemF[threadId1D()] = 0u;

    uint numTopLevelCells = aGridResX * aGridResY * aGridResZ;

    for(uint refId = globalThreadId1D(); refId < aNumTopLevelRefs; refId += numThreads())
    {
        const uint2 indexPair = aTopLevelSortedPairs[refId];

        if (indexPair.x >= /***/numTopLevelCells)
        {
            break;
        }

        const tPrimitive prim = aPrimitiveArray[indexPair.y];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);

        const uint topLvlCellX = indexPair.x % aGridResX;
        const uint topLvlCellY = (indexPair.x %(aGridResX * aGridResY)) / aGridResX;
        const uint topLvlCellZ = indexPair.x / (aGridResX * aGridResY);

        const TwoLevelGridCell topLvlCell = *((TwoLevelGridCell*)
            ((char*)aTopLevelCells.ptr + 
            + topLvlCellY * aTopLevelCells.pitch
            + topLvlCellZ * aTopLevelCells.pitch * aTopLevelCells.ysize)
            + topLvlCellX);

        float3 topLvlCellRes;
        topLvlCellRes.x = static_cast<float>(topLvlCell[0]);
        topLvlCellRes.y = static_cast<float>(topLvlCell[1]);
        topLvlCellRes.z = static_cast<float>(topLvlCell[2]);

        float3 topLvlCellOrigin;
        topLvlCellOrigin.x = static_cast<float>(topLvlCellX) * aCellSize.x + aBoundsMin.x;
        topLvlCellOrigin.y = static_cast<float>(topLvlCellY) * aCellSize.y + aBoundsMin.y;
        topLvlCellOrigin.z = static_cast<float>(topLvlCellZ) * aCellSize.z + aBoundsMin.z;

        const float3 subCellSizeRCP = topLvlCellRes / aCellSize;

        //triangleBounds.tighten(topLvlCellOrigin, topLvlCellOrigin + aCellSize);

        float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        minCellIdf =
            //const float3 minCellIdf =
            (bounds.vtx[0] - topLvlCellOrigin + rep(10E-5f)) * subCellSizeRCP;
        const float3 maxCellIdPlus1f =
            (bounds.vtx[1] - topLvlCellOrigin + rep(10E-5f)) * subCellSizeRCP + rep(1.f);

        const int minCellIdX =  max(0, (int)(minCellIdf.x));
        const int minCellIdY =  max(0, (int)(minCellIdf.y));
        const int minCellIdZ =  max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)topLvlCellRes.x, (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)topLvlCellRes.y, (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)topLvlCellRes.z, (int)(maxCellIdPlus1f.z));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

        const float cellSurfaceArea = 2.f * ( 
            1.f / (subCellSizeRCP.x * subCellSizeRCP.y) + 
            1.f / (subCellSizeRCP.x * subCellSizeRCP.z) +
            1.f / (subCellSizeRCP.y * subCellSizeRCP.z));

        const float topCellSurfaceArea = 2.f * (
            aCellSize.x * aCellSize.y + 
            aCellSize.x * aCellSize.z +
            aCellSize.y * aCellSize.z
            );

        shMemF[threadId1D()] += numCells * cellSurfaceArea / topCellSurfaceArea;
    }

    SYNCTHREADS;

#if HAPPYRAY__CUDA_ARCH__ >= 120

    //reduction
    if (taBlockSize >= 512) { if (threadId1D() < 256) { shMemF[threadId1D()] += shMemF[threadId1D() + 256]; } SYNCTHREADS;   }
    if (taBlockSize >= 256) { if (threadId1D() < 128) { shMemF[threadId1D()] += shMemF[threadId1D() + 128]; } SYNCTHREADS;   }
    if (taBlockSize >= 128) { if (threadId1D() <  64) { shMemF[threadId1D()] += shMemF[threadId1D() +  64]; } SYNCTHREADS;   }
    if (taBlockSize >=  64) { if (threadId1D() <  32) { shMemF[threadId1D()] += shMemF[threadId1D() +  32]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  32) { if (threadId1D() <  16) { shMemF[threadId1D()] += shMemF[threadId1D() +  16]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  16) { if (threadId1D() <   8) { shMemF[threadId1D()] += shMemF[threadId1D() +   8]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   8) { if (threadId1D() <   4) { shMemF[threadId1D()] += shMemF[threadId1D() +   4]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   4) { if (threadId1D() <   2) { shMemF[threadId1D()] += shMemF[threadId1D() +   2]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   2) { if (threadId1D() <   1) { shMemF[threadId1D()] += shMemF[threadId1D() +   1]; } EMUSYNCTHREADS;}

    // write out block sum 
    if (threadId1D() == 0) oCost[blockId1D()] = shMemF[0];

#else

    oCost[globalThreadId1D()] = shMemF[threadId1D()];

#endif
}


#endif // TLGRIDBUILDKERNELS_H_21EA95C7_8004_4EA0_A7A8_8E377C62A7AA
