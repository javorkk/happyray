/****************************************************************************/
/* Copyright (c) 2013, Javor Kalojanov
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

#ifndef TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448
#define TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/TwoLevelGridHierarchy.h"
#include "RT/Structure/TLGridHierarchyMemoryManager.h"
#include "RT/Algorithm/UniformGridBuildKernels.h"
#include "RT/Algorithm/TLGridBuildKernels.h"

#include "Utils/Scan.h"
#include "Utils/Sort.h"

extern SHARED uint shMem[];

//Computes the resolution and number cells for each input item
GLOBAL void countGridCellsMultiUniformGrid(
    const uint      aNumItems,
    const float     aDensity,
    UniformGrid*    oGirds,
    bool            aSetResolution,
    uint*           oCellCounts //stores the number of primitives
    )
{
    for(uint gridId = globalThreadId1D(); gridId < aNumItems; gridId += numThreads())
    {
        if(aSetResolution)
        {
            float3 diagonal = oGirds[gridId].vtx[1] - oGirds[gridId].vtx[0];
            const float volume = diagonal.x * diagonal.y * diagonal.z;
            const float lambda = aDensity;
            const float magicConstant =
                powf(lambda * static_cast<float>(oCellCounts[gridId]) / volume, 0.3333333f);

            diagonal *= magicConstant;
            int resX = static_cast<int>(diagonal.x);
            int resY = static_cast<int>(diagonal.y);
            int resZ = static_cast<int>(diagonal.z);
            oGirds[gridId].res[0] = resX > 0 ? resX : 1;
            oGirds[gridId].res[1] = resY > 0 ? resY : 1;
            oGirds[gridId].res[2] = resZ > 0 ? resZ : 1;
            float3 resolution = make_float3(oGirds[gridId].res[0], oGirds[gridId].res[1], oGirds[gridId].res[2]);
            oGirds[gridId].cellSize = diagonal / resolution;
            oGirds[gridId].cellSizeRCP = resolution / diagonal;

            oCellCounts[gridId] = resX * resY * resZ;

        }
        else
        {
            oCellCounts[gridId] = oGirds[gridId].res[0] * oGirds[gridId].res[1] * oGirds[gridId].res[2];
        }
    }
    //Should not be necessary ( escan should overwrite the last item )
    //if(globalThreadId1D() == 0)
    //{
    //    oCellCounts[aNumItems] = 0;
    //}
}

GLOBAL void prepareLeavesPointersMultiUniformGrid(
    const uint                              aNumItems,
    TLGridHierarchyMemoryManager::t_Leaf*   aLeavesBasePtr,
    uint*                                   aCellCounts,
    UniformGrid*                            oGirds    
    )
{
    for(uint gridId = globalThreadId1D(); gridId < aNumItems; gridId += numThreads())
    {
        int xsize = oGirds[gridId].res[0];
        int ysize = oGirds[gridId].res[1];
        int pitch = xsize * sizeof(uint2);
        void* ptr = (void*)(aLeavesBasePtr + aCellCounts[gridId]);
        oGirds[gridId].cells.xsize = xsize;
        oGirds[gridId].cells.ysize = ysize;
        oGirds[gridId].cells.pitch = pitch;
        oGirds[gridId].cells.ptr = ptr;
    }
}
GLOBAL void preparePrimitivePointersMultiUniformGrid(
    const uint                              aNumItems,
    uint*                                   aPrimitiveIndices,
    UniformGrid*                            oGirds   
    )
{
    for(uint gridId = globalThreadId1D(); gridId < aNumItems; gridId += numThreads())
    {
        oGirds[gridId].primitives = aPrimitiveIndices;
    }
}

template<class tPrimitive, int taBlockSize>
GLOBAL void countPairsMultiUniformGrid(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    UniformGrid*                aGirds,
    const uint                  aNumGrids,
    uint*                       aScannedPrimitivesPerGrid,
    uint*                       oRefCounts
    )
{
    shMem[threadId1D()] = 0u;    

    for(uint primId = globalThreadId1D(); primId < aPrimitiveArray.numPrimitives; primId += numThreads())
    {
        
        const tPrimitive prim = aPrimitiveArray[primId];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);
        uint gridId = 0;
        for (; primId >= aScannedPrimitivesPerGrid[gridId]; ++gridId) ;

        float3 topLvlCellOrigin  = aGirds[gridId].vtx[0];
        const float3 subCellSizeRCP = aGirds[gridId].getCellSizeRCP();
        const float3 minCellIdf =
            (bounds.vtx[0] - topLvlCellOrigin ) * subCellSizeRCP;
        const float3 maxCellIdPlus1f =
            (bounds.vtx[1] - topLvlCellOrigin ) * subCellSizeRCP + rep(1.f);

        const int minCellIdX =  min(aGirds[gridId].res[0]-1, max(0, (int)(minCellIdf.x)));
        const int minCellIdY =  min(aGirds[gridId].res[1]-1, max(0, (int)(minCellIdf.y)));
        const int minCellIdZ =  min(aGirds[gridId].res[2]-1, max(0, (int)(minCellIdf.z)));

        const int maxCellIdP1X =  max(1, min(aGirds[gridId].res[0], (int)(maxCellIdPlus1f.x)));
        const int maxCellIdP1Y =  max(1, min(aGirds[gridId].res[1], (int)(maxCellIdPlus1f.y)));
        const int maxCellIdP1Z =  max(1, min(aGirds[gridId].res[2], (int)(maxCellIdPlus1f.z)));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

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

template<class tPrimitive>
GLOBAL void writePairsMultiUniformGrid(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    UniformGrid*                aGirds,
    const uint                  aNumGrids,
    uint*                       aScannedPrimitivesPerGrid,
    uint*                       aStartId,
    uint*                       oPairs
    )
{

#if HAPPYRAY__CUDA_ARCH__ >= 120

    if (threadId1D() == 0)
    {
        shMem[0] = aStartId[blockId1D()];
    }

    SYNCTHREADS;

#else

    uint startPosition = aStartId[globalThreadId1D()];

#endif

    uint2* basePtr = (uint2*)(aGirds[0].cells.ptr);

    for(uint primId = globalThreadId1D(); primId < aPrimitiveArray.numPrimitives; primId += numThreads())
    {

        const tPrimitive prim = aPrimitiveArray[primId];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);
        uint gridId = 0;
        for (; primId >= aScannedPrimitivesPerGrid[gridId]; ++gridId) ;

        float3 topLvlCellOrigin  = aGirds[gridId].vtx[0];
        const float3 subCellSizeRCP = aGirds[gridId].getCellSizeRCP();
        const float3 minCellIdf =
            (bounds.vtx[0] - topLvlCellOrigin ) * subCellSizeRCP;
        const float3 maxCellIdPlus1f =
            (bounds.vtx[1] - topLvlCellOrigin ) * subCellSizeRCP + rep(1.f);

        const int minCellIdX =  min(aGirds[gridId].res[0]-1, max(0, (int)(minCellIdf.x)));
        const int minCellIdY =  min(aGirds[gridId].res[1]-1, max(0, (int)(minCellIdf.y)));
        const int minCellIdZ =  min(aGirds[gridId].res[2]-1, max(0, (int)(minCellIdf.z)));

        const int maxCellIdP1X =  max(1, min(aGirds[gridId].res[0], (int)(maxCellIdPlus1f.x)));
        const int maxCellIdP1Y =  max(1, min(aGirds[gridId].res[1], (int)(maxCellIdPlus1f.y)));
        const int maxCellIdP1Z =  max(1, min(aGirds[gridId].res[2], (int)(maxCellIdPlus1f.z)));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

#if HAPPYRAY__CUDA_ARCH__ >= 120
        int nextSlot  = atomicAdd(shMem, numCells);
#else
        int nextSlot = startPosition;
        startPosition += numCells;
#endif
        int cellOffset = (uint2*)(aGirds[gridId].cells.ptr) - basePtr;
        for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
        {
            for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
            {
                for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                {
                    oPairs[2 * nextSlot] = x + y * aGirds[gridId].res[0] +
                        z * aGirds[gridId].res[0] * aGirds[gridId].res[1] +
                        cellOffset;
                    oPairs[2 * nextSlot + 1] = primId;
                }//end for z
            }//end for y
        }//end for x


    }//end  for(uint refId = globalThreadId1D(); ...

}


template<class tPrimitive>
class TLGridHierarchySortBuilder
{
    static const uint   sNUM_COUNTER_THREADS    = 128u;
    static const uint   sNUM_COUNTER_BLOCKS     =  90u;
    //NOTE: WRITE and COUNTER threads and blocks have to be exactly the same
    static const uint   sNUM_WRITE_THREADS      = sNUM_COUNTER_THREADS;
    static const uint   sNUM_WRITE_BLOCKS       = sNUM_COUNTER_BLOCKS;
    static const uint   sNUM_CELL_SETUP_THREADS = 256u;
    static const uint   sNUM_CELL_SETUP_BLOCKS  = 240u;

    cudaEvent_t mStart, mDataUpload, mScan;
    cudaEvent_t mTopLevel, mLeafCellCount, mLeafRefsCount,
        mLeafRefsWrite, mSortLeafPairs, mEnd;

public:
    HOST void init(
        TLGridHierarchyMemoryManager&   aMemoryManager,
        const uint                      aNumInstances,
        const float                     aTopLevelDensity = 0.0625f,
        const float                     aLeafLevelDensity = 1.2f
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        cudaEventSynchronize(mStart);
        //////////////////////////////////////////////////////////////////////////

        float3 diagonal = aMemoryManager.bounds.diagonal();

        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = aTopLevelDensity;
        const float magicConstant =
            powf(lambda * 16.f /*static_cast<float>(aNumInstances)*/ / volume, 0.3333333f);

        diagonal *= magicConstant;

        aMemoryManager.resX = static_cast<int>(diagonal.x);
        aMemoryManager.resY = static_cast<int>(diagonal.y);
        aMemoryManager.resZ = static_cast<int>(diagonal.z);

        aMemoryManager.topLevelDensity = aTopLevelDensity;
        aMemoryManager.leafLevelDensity = aLeafLevelDensity;

        aMemoryManager.allocateDeviceCells();
        aMemoryManager.setDeviceCellsToZero();
        
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        PrimitiveArray<tPrimitive>&         aPrimitiveArray)
    {

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        uint* primitiveCounts;
        MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&primitiveCounts, 4 * sizeof(uint) ) );
        BBox* bounds;
        MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&bounds, 4 * sizeof(BBox) ) );
        for (int i = 0; i < 4; ++i)
        {
            primitiveCounts[i] = (uint)aPrimitiveArray.numPrimitives / 4u + 1u;
            bounds[i] = aMemoryManager.bounds;
        }

        aMemoryManager.allocateGrids(4);
        for(size_t gridId = 0; gridId < 4; ++gridId)
        {
            aMemoryManager.gridsHost[gridId].vtx[0] = bounds[gridId].vtx[0];
            aMemoryManager.gridsHost[gridId].vtx[1] = bounds[gridId].vtx[1];
        }
        aMemoryManager.copyGridsHostToDevice();

        GeometryInstance* hostInstances = aMemoryManager.allocateHostInstances(16);
        GeometryInstance* deviceInstances = aMemoryManager.allocateDeviceInstances(16);

        for (int i = 0; i < 16; ++i)
        {
            hostInstances[i].vtx[0] = aMemoryManager.bounds.vtx[0] + aMemoryManager.bounds.diagonal() * 0.25f * floorf(i*0.25f);
            hostInstances[i].vtx[1] = hostInstances[i].vtx[0] + aMemoryManager.bounds.diagonal() * 0.25f;
            hostInstances[i].index = i % 4;
            hostInstances[i].rotation0   = make_float3(1.f, 0.f, 0.f);
            hostInstances[i].rotation1   = make_float3(0.f, 1.f, 0.f);
            hostInstances[i].rotation2   = make_float3(0.f, 0.f, 1.f);
            hostInstances[i].translation = make_float3(0.f, 0.f, 0.f); //aMemoryManager.bounds.diagonal() * (float) (i / 4);
            hostInstances[i].irotation0   = make_float3(1.f, 0.f, 0.f);
            hostInstances[i].irotation1   = make_float3(0.f, 1.f, 0.f);
            hostInstances[i].irotation2   = make_float3(0.f, 0.f, 1.f);
            hostInstances[i].itranslation =  make_float3(0.f, 0.f, 0.f); //aMemoryManager.bounds.diagonal() * (float) (-i / 4);
        }
        aMemoryManager.copyInstancesHostToDevice();
        //END DEBUG
        //////////////////////////////////////////////////////////////////////////

        build(aMemoryManager, 4, primitiveCounts, aPrimitiveArray);

        MY_CUDA_SAFE_CALL( cudaFreeHost( primitiveCounts ) );
        MY_CUDA_SAFE_CALL( cudaFreeHost( bounds ) );
    }

    HOST void build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint                                aNumUniqueInstances,
        uint*                               aPrimitiveCounts,
        PrimitiveArray<tPrimitive>&         aPrimitiveArray)
    {
        //////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&mScan);
        cudaEventCreate(&mTopLevel);
        cudaEventCreate(&mLeafCellCount);
        cudaEventCreate(&mLeafRefsCount);
        cudaEventCreate(&mLeafRefsWrite);
        cudaEventCreate(&mSortLeafPairs);
        cudaEventCreate(&mEnd);
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //TOP LEVEL GRID CONSTRUCTION
        //////////////////////////////////////////////////////////////////////////

        dim3 blockTotalSize(sNUM_COUNTER_THREADS);
        dim3 gridTotalSize (sNUM_COUNTER_BLOCKS);

#if HAPPYRAY__CUDA_ARCH__ >= 120
        const int numCounters = gridTotalSize.x;
#else
        const int numCounters = gridTotalSize.x * blockTotalSize.x;
#endif
        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);

        const uint numInstances = (uint)(aMemoryManager.instancesSize / sizeof(GeometryInstance));
        countPairs<GeometryInstance, GeometryInstance*, sNUM_COUNTER_THREADS >
            <<< gridTotalSize, blockTotalSize,
            blockTotalSize.x * (sizeof(uint) + sizeof(float3))>>>(
            aMemoryManager.instancesDevice,
            numInstances,
            aMemoryManager.getResolution(), 
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP(),
            aMemoryManager.refCountsBuffer);

        /////////////////////////////////////////////////////////////////////////
        //DEBUG
        //cudaThreadSynchronize();
        //cudastd::logger::out << "Initial counts:";
        //for(size_t i = 0; i <= numCounters; ++i)
        //{
        //    cudastd::logger::out << " " <<  aMemoryManager.refCountsBufferHost[i];
        //}
        //cudastd::logger::out << "\n ----------------------\n";
        /////////////////////////////////////////////////////////////////////////

        ExclusiveScan scan;
        scan(aMemoryManager.refCountsBuffer, numCounters + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        /////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mScan, 0);
        cudaEventSynchronize(mScan);
        MY_CUT_CHECK_ERROR("Counting top level primitive-cell pairs failed.\n");
        /////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////
        //DEBUG
        //cudastd::logger::out << "Scanned counts:";
        //for(size_t i = 0; i <= numCounters; ++i)
        //{
        //    cudastd::logger::out << " " <<  aMemoryManager.refCountsBufferHost[i];
        //}
        //cudastd::logger::out << "\n ----------------------\n";
        /////////////////////////////////////////////////////////////////////////


        const uint numTopLevelPairs = aMemoryManager.refCountsBufferHost[numCounters];
        aMemoryManager.allocateTopLevelPairsBufferPair(numTopLevelPairs);


        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

        writePairs<GeometryInstance, GeometryInstance*, false>
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aMemoryManager.instancesDevice,
            aMemoryManager.topLevelPairsBuffer,
            numInstances,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        MY_CUT_CHECK_ERROR("Writing primitive-cell pairs failed.\n");


        const uint numCellsPlus1 = aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ;
        uint numBits = 9u;
        while (numCellsPlus1 >> numBits != 0u){numBits += 1u;}
        numBits = cudastd::min(32u, numBits + 1u);

        Sort radixSort;
        radixSort(aMemoryManager.topLevelPairsBuffer, aMemoryManager.topLevelPairsPingBufferKeys, numTopLevelPairs, numBits);

        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");

        aMemoryManager.allocateInstanceIndices(numTopLevelPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.instanceIndicesDevice,
            (uint2*)aMemoryManager.topLevelPairsBuffer,
            numTopLevelPairs,
            aMemoryManager.cellsPtrDevice,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ)
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mTopLevel, 0);
        cudaEventSynchronize(mTopLevel);
        MY_CUT_CHECK_ERROR("Setting up top level cells failed.\n");
        //////////////////////////////////////////////////////////////////////////
        //END OF TOP LEVEL GRID CONSTRUCTION
        //////////////////////////////////////////////////////////////////////////


        buildLevelTwo(aMemoryManager, aPrimitiveCounts, aNumUniqueInstances, aPrimitiveArray);
    }

    //dummy to use instead of init
    HOST void initEvents(
        TLGridHierarchyMemoryManager&   aMemoryManager,
        const uint                      aNumInstances,
        const float                     aTopLevelDensity = 0.0625f,
        const float                     aLeafLevelDensity = 1.2f
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        cudaEventSynchronize(mStart);
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void buildLevelTwo(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint*                               aPrimitiveCounts,
        const uint                          aNumGrids,
        PrimitiveArray<tPrimitive>&         aPrimitiveArray)
    {


        aMemoryManager.allocateCellCountsBuffer(aNumGrids + 1);
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBuffer, aPrimitiveCounts , aNumGrids * sizeof(uint), cudaMemcpyHostToDevice) );
        
        dim3 blockCellCount(sNUM_COUNTER_THREADS);
        dim3 gridCellCount(sNUM_COUNTER_BLOCKS);

        countGridCellsMultiUniformGrid<<< gridCellCount, blockCellCount >>>(
            aNumGrids,
            aMemoryManager.leafLevelDensity,
            aMemoryManager.gridsDevice,
            true,
            aMemoryManager.cellCountsBuffer
            );


        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafCellCount, 0);
        cudaEventSynchronize(mLeafCellCount);
        MY_CUT_CHECK_ERROR("Counting leaf level cells failed.\n");
        //////////////////////////////////////////////////////////////////////////

        ExclusiveScan escan;
        escan(aMemoryManager.cellCountsBuffer, aNumGrids + 1);


#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBufferHost + aNumGrids, (aMemoryManager.refCountsBuffer + aNumGrids), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif

        const uint numLeafCells = *(aMemoryManager.cellCountsBufferHost + aNumGrids);

        aMemoryManager.allocateDeviceLeaves(numLeafCells);
        aMemoryManager.setDeviceLeavesToZero();
        prepareLeavesPointersMultiUniformGrid<<< gridCellCount, blockCellCount >>>(
            aNumGrids,
            aMemoryManager.leavesDevice,
            aMemoryManager.cellCountsBuffer,
            aMemoryManager.gridsDevice
            );

        //re-use cell counts buffer to store the scanned number of primitives per grid
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBuffer, aPrimitiveCounts , aNumGrids * sizeof(uint), cudaMemcpyHostToDevice) );
        InclusiveScan iscan;
        iscan(aMemoryManager.cellCountsBuffer, aNumGrids); 

        dim3 blockRefCount = sNUM_COUNTER_THREADS;
        dim3 gridRefCount  = sNUM_COUNTER_BLOCKS;

#if HAPPYRAY__CUDA_ARCH__ >= 120
        const int numCounters = gridRefCount.x;
#else
        const int numCounters = gridRefCount.x * blockRefCount.x;
#endif
        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);


        countPairsMultiUniformGrid<tPrimitive, sNUM_COUNTER_THREADS > 
            <<< gridRefCount, blockRefCount,  blockRefCount.x * (sizeof(uint) /*+ sizeof(float3)*/) >>>(
            aPrimitiveArray,
            aMemoryManager.gridsDevice,
            aNumGrids,
            aMemoryManager.cellCountsBuffer,
            aMemoryManager.refCountsBuffer
            );

        escan(aMemoryManager.refCountsBuffer, numCounters + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsCount, 0);
        cudaEventSynchronize(mLeafRefsCount);
        MY_CUT_CHECK_ERROR("Counting leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

        const uint numLeafLevelPairs = aMemoryManager.refCountsBufferHost[numCounters];

        aMemoryManager.allocateLeafLevelPairsBufferPair(numLeafLevelPairs);

        dim3 blockRefWrite = sNUM_WRITE_THREADS;
        dim3 gridRefWrite  = sNUM_WRITE_BLOCKS;

        writePairsMultiUniformGrid<tPrimitive>
            <<< gridRefWrite, blockRefWrite,  sizeof(uint)>>>(
            aPrimitiveArray,            
            aMemoryManager.gridsDevice,
            aNumGrids,
            aMemoryManager.cellCountsBuffer,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.leafLevelPairsBuffer
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        cudaEventSynchronize(mLeafRefsWrite);
        MY_CUT_CHECK_ERROR("Writing the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////


        uint numBits = 7u;
        while (numLeafCells >> numBits != 0u){numBits += 1u;}
        numBits = cudastd::min(32u, numBits + 1u);

        Sort radixSort;
        radixSort(aMemoryManager.leafLevelPairsBuffer, aMemoryManager.leafLevelPairsPingBufferKeys, numLeafLevelPairs, numBits);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        cudaEventSynchronize(mSortLeafPairs);
        MY_CUT_CHECK_ERROR("Sorting the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

        aMemoryManager.allocatePrimitiveIndicesBuffer(numLeafLevelPairs);

        dim3 blockPrepLeafRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepLeafRng (sNUM_CELL_SETUP_BLOCKS );

        prepareLeafCellRanges
            <<< gridPrepLeafRng, blockPrepLeafRng,
            (2 + blockPrepLeafRng.x) * sizeof(uint) >>>(
            aMemoryManager.primitiveIndices,
            (uint2*)aMemoryManager.leafLevelPairsBuffer,
            numLeafLevelPairs,
            (uint2*)aMemoryManager.leavesDevice
            );

        preparePrimitivePointersMultiUniformGrid<<< gridCellCount, blockCellCount >>>(
            aNumGrids,
            aMemoryManager.primitiveIndices,
            aMemoryManager.gridsDevice
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        MY_CUT_CHECK_ERROR("Setting up leaf cells and primitive array failed.\n");
        //////////////////////////////////////////////////////////////////////////
        
        //outputStats();
        cleanup();

    }

    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mTopLevel);
        cudaEventDestroy(mLeafCellCount);
        cudaEventDestroy(mLeafRefsCount);
        cudaEventDestroy(mLeafRefsWrite);
        cudaEventDestroy(mSortLeafPairs);
        cudaEventDestroy(mEnd);
    }
    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::logger::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::logger::out << "Data upload:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mTopLevel);
        cudastd::logger::out << "Top Level:        " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mEnd);
        cudastd::logger::out << "Leaf Level:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mLeafCellCount);
        cudastd::logger::out << "Leaf Cells Count: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafCellCount, mLeafRefsCount);
        cudastd::logger::out << "Leaf Refs Count:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsCount, mLeafRefsWrite);
        cudastd::logger::out << "Leaf Refs Write:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsWrite, mSortLeafPairs);
        cudastd::logger::out << "Leaf Refs Sort:   " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime,mSortLeafPairs, mEnd);
        cudastd::logger::out << "Prep. Leaf cells: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::logger::out << "Total:            " << elapsedTime << "ms\n";
        //////////////////////////////////////////////////////////////////////////
    }
};

#endif // TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448
