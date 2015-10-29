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

#ifndef TLGRIDSORTBUILDER_H_085A14AE_437D_424C_BBA8_417665305A48
#define TLGRIDSORTBUILDER_H_085A14AE_437D_424C_BBA8_417665305A48

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/TwoLevelGrid.h"
#include "RT/Structure/TLGridMemoryManager.h"
#include "RT/Algorithm/UniformGridBuildKernels.h"
#include "RT/Algorithm/TLGridBuildKernels.h"

#include "Utils/Scan.h"
//#define CHAG_SORT
#ifdef CHAG_SORT
#include "Utils/Sort.h"
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

template<class tPrimitive>
class TLGridSortBuilder
{
    static const uint   sNUM_COUNTER_THREADS    = 128u;
    static const uint   sNUM_COUNTER_BLOCKS     =  90u;
    //NOTE: WRITE and COUNTER threads and blocks have to be exactly the same
    static const uint   sNUM_WRITE_THREADS      = sNUM_COUNTER_THREADS;
    static const uint   sNUM_WRITE_BLOCKS       = sNUM_COUNTER_BLOCKS;
    static const uint   sNUM_CELL_SETUP_THREADS = 256u;
    static const uint   sNUM_CELL_SETUP_BLOCKS  = 240u;

    uint mNumPrimitives;
    cudaEvent_t mStart, mDataUpload, mScan;
    cudaEvent_t mTopLevel, mLeafCellCount, mLeafRefsCount,
        mLeafRefsWrite, mSortLeafPairs, mEnd;
public:
    HOST void init(
        TLGridMemoryManager&            aMemoryManager,
        const uint                      aNumPrimitives,
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
            powf(lambda * static_cast<float>(aNumPrimitives) / volume, 0.3333333f);

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

    //dummy to use instead of init
    HOST void initEvents(
        TLGridMemoryManager&            aMemoryManager,
        const uint                      aNumPrimitives,
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

    HOST void build(
        TLGridMemoryManager&                aMemoryManager,
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


        countPairs<tPrimitive, PrimitiveArray<tPrimitive>, sNUM_COUNTER_THREADS >
            <<< gridTotalSize, blockTotalSize,
            blockTotalSize.x * (sizeof(uint) + sizeof(float3))>>>(
            aPrimitiveArray,
            (uint)aPrimitiveArray.numPrimitives,
            aMemoryManager.getResolution(), 
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP(),
            aMemoryManager.refCountsBuffer);

        /////////////////////////////////////////////////////////////////////////
        //DEBUG
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
        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

#ifdef CHAG_SORT  
        aMemoryManager.allocateTopLevelPairsBufferPair(numTopLevelPairs);

        writePairs<tPrimitive, PrimitiveArray<tPrimitive>, false>
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aPrimitiveArray,
            aMemoryManager.topLevelPairsBuffer,
            (uint)aPrimitiveArray.numPrimitives,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        MY_CUT_CHECK_ERROR("Writing primitive-cell pairs failed.\n");

        const uint numCellsPlus1 = aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ;
        uint numBits = 9u;
        while (numCellsPlus1 >> numBits != 0u){ numBits += 1u; }
        numBits = cudastd::min(32u, numBits + 1u);

        Sort radixSort;
        radixSort(aMemoryManager.topLevelPairsBuffer, aMemoryManager.topLevelPairsPingBufferKeys, numTopLevelPairs, numBits);

        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");

#else
        aMemoryManager.allocateTopLevelKeyValueBuffers(numTopLevelPairs);
        writeKeysAndValues<tPrimitive, PrimitiveArray<tPrimitive>, false>
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aPrimitiveArray,
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
            (uint)aPrimitiveArray.numPrimitives,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        MY_CUT_CHECK_ERROR("Writing primitive-cell pairs failed.\n");

        thrust::device_ptr<unsigned int> dev_keys(aMemoryManager.topLevelPairsPingBufferKeys);
        thrust::device_ptr<unsigned int> dev_values(aMemoryManager.topLevelPairsPingBufferValues);
        thrust::sort_by_key(dev_keys, (dev_keys + numTopLevelPairs), dev_values);

        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");
#endif //CHAG_SORT


        aMemoryManager.allocatePrimitiveIndicesBuffer(numTopLevelPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.primitiveIndices,
#ifdef CHAG_SORT
            (uint2*)aMemoryManager.topLevelPairsBuffer,
#else
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
#endif
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

        aMemoryManager.allocateCellCountsBuffer(aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ + 1);

        dim3 blockCellCount(aMemoryManager.resX);
        dim3 gridCellCount(aMemoryManager.resY, aMemoryManager.resZ);

        countLeafLevelCells<false> 
            <<< gridCellCount, blockCellCount >>>(
            aMemoryManager.getCellSize(),
            aMemoryManager.leafLevelDensity,
            aMemoryManager.cellsPtrDevice,
            aMemoryManager.cellCountsBuffer
            );

        scan(aMemoryManager.cellCountsBuffer, aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ + 1);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafCellCount, 0);
        cudaEventSynchronize(mLeafCellCount);
        MY_CUT_CHECK_ERROR("Counting leaf level cells failed.\n");
        //////////////////////////////////////////////////////////////////////////
        
        prepareTopLevelCellRanges<0><<< gridCellCount, blockCellCount >>>(
            aMemoryManager.cellCountsBuffer,
            aMemoryManager.cellsPtrDevice
            );

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBufferHost + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ, (aMemoryManager.refCountsBuffer + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif

        const uint numLeafCells = *(aMemoryManager.cellCountsBufferHost + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ);

        dim3 blockRefCount = sNUM_COUNTER_THREADS;
        dim3 gridRefCount  = sNUM_COUNTER_BLOCKS;

        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);

        countLeafLevelPairs<tPrimitive, PrimitiveArray, sNUM_COUNTER_THREADS > 
            <<< gridRefCount, blockRefCount,  blockRefCount.x * (sizeof(uint) /*+ sizeof(float3)*/) >>>(
            aPrimitiveArray,
            numTopLevelPairs,
#ifdef CHAG_SORT
            (uint2*)aMemoryManager.topLevelPairsBuffer,
#else
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
#endif
            aMemoryManager.cellsPtrDevice,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.refCountsBuffer
            //////////////////////////////////////////////////////////////////////////
            //DEBUG
            //, debugInfo
            //////////////////////////////////////////////////////////////////////////
            );

        scan(aMemoryManager.refCountsBuffer, numCounters + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsCount, 0);
        cudaEventSynchronize(mLeafRefsCount);
        MY_CUT_CHECK_ERROR("Counting leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////
        
        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //cudastd::logger::out << "Scanned leaf level reference counts: ";
        ////#if HAPPYRAY__CUDA_ARCH__ < 120
        ////MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost, aMemoryManager.refCountsBuffer, (numCounters + 1) * sizeof(uint), cudaMemcpyDeviceToHost ));
        ////#endif
        //for(int it = 0; it < numCounters + 1; ++it)
        //{
        //    cudastd::logger::out << aMemoryManager.refCountsBufferHost[it] << " ";
        //}
        //cudastd::logger::out << "\n";
        //////////////////////////////////////////////////////////////////////////


        const uint numLeafLevelPairs = aMemoryManager.refCountsBufferHost[numCounters];
        dim3 blockRefWrite = sNUM_WRITE_THREADS;
        dim3 gridRefWrite  = sNUM_WRITE_BLOCKS;


#ifdef CHAG_SORT
        aMemoryManager.allocateLeafLevelPairsBufferPair(numLeafLevelPairs);

        writeLeafLevelPairs<tPrimitive, PrimitiveArray>
            <<< gridRefWrite, blockRefWrite,  sizeof(uint)>>>(
            aPrimitiveArray,
            numTopLevelPairs,
            (uint2*)aMemoryManager.topLevelPairsBuffer,
            aMemoryManager.cellsPtrDevice,
            numLeafCells,
            aMemoryManager.refCountsBuffer,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.leafLevelPairsBuffer
            );
       
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        cudaEventSynchronize(mLeafRefsWrite);
        MY_CUT_CHECK_ERROR("Writing the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////
        

        numBits = 7u;
        while (numLeafCells >> numBits != 0u){numBits += 1u;}
        numBits = cudastd::min(32u, numBits + 1u);

        radixSort(aMemoryManager.leafLevelPairsBuffer, aMemoryManager.leafLevelPairsPingBufferKeys, numLeafLevelPairs, numBits);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        cudaEventSynchronize(mSortLeafPairs);
        MY_CUT_CHECK_ERROR("Sorting the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint2* hostPairs;
        //MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&hostPairs, numLeafLevelPairs * sizeof(uint2)) );
        //MY_CUDA_SAFE_CALL( cudaMemcpy(hostPairs, aMemoryManager.leafLevelPairsBuffer, numLeafLevelPairs * sizeof(uint2), cudaMemcpyDeviceToHost) );
        //uint numRealPairs = 0u;
        //for(uint it = 0; it < numLeafLevelPairs - 1; ++it)
        //{
        //    if (hostPairs[it].x < numLeafCells)
        //    {
        //        ++numRealPairs;
        //    }
        //    if(hostPairs[it].x > hostPairs[it + 1].x)
        //    {
        //        cudastd::logger::out << "Unsorted pairs ( " << hostPairs[it].x << " | " << hostPairs[it].y  << " ) ";
        //        cudastd::logger::out << " ( " << hostPairs[it+1].x << " | " << hostPairs[it+1].y  << " ) ";
        //    }
        //}
        //////////////////////////////////////////////////////////////////////////

#else
        aMemoryManager.allocateLeafLevelKeyValueBuffers(numLeafLevelPairs);

        writeLeafLevelKeysAndValues<tPrimitive, PrimitiveArray>
            << < gridRefWrite, blockRefWrite, sizeof(uint) >> >(
            aPrimitiveArray,
            numTopLevelPairs,
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
            aMemoryManager.cellsPtrDevice,
            numLeafCells,
            aMemoryManager.refCountsBuffer,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.leafLevelPairsPingBufferKeys,
            aMemoryManager.leafLevelPairsPingBufferValues
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        cudaEventSynchronize(mLeafRefsWrite);
        MY_CUT_CHECK_ERROR("Writing the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

        thrust::device_ptr<unsigned int> dev_keys_leaf(aMemoryManager.leafLevelPairsPingBufferKeys);
        thrust::device_ptr<unsigned int> dev_values_leaf(aMemoryManager.leafLevelPairsPingBufferValues);
        thrust::sort_by_key(dev_keys_leaf, (dev_keys_leaf + numLeafLevelPairs), dev_values_leaf);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        cudaEventSynchronize(mSortLeafPairs);
        MY_CUT_CHECK_ERROR("Sorting the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

#endif //CHAG_SORT
        
        aMemoryManager.allocatePrimitiveIndicesBuffer(numLeafLevelPairs);
        aMemoryManager.allocateDeviceLeaves(numLeafCells);
        aMemoryManager.setDeviceLeavesToZero();

        dim3 blockPrepLeafRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepLeafRng (sNUM_CELL_SETUP_BLOCKS );

        prepareLeafCellRanges<sNUM_CELL_SETUP_THREADS>
            <<< gridPrepLeafRng, blockPrepLeafRng,
            (2 + blockPrepLeafRng.x) * sizeof(uint) >>>(
            aMemoryManager.primitiveIndices,
#ifdef CHAG_SORT
            (uint2*)aMemoryManager.leafLevelPairsBuffer,
#else
            aMemoryManager.leafLevelPairsPingBufferKeys,
            aMemoryManager.leafLevelPairsPingBufferValues,
#endif
            numLeafLevelPairs,
            (uint2*)aMemoryManager.leavesDevice
            );
         
        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //aMemoryManager.allocateHostLeaves(numLeafCells);
        //aMemoryManager.copyLeavesDeviceToHost();
        //for(uint i = 0; i < numLeafCells; ++i)
        //{
        //    if(aMemoryManager.leavesHost[i].x > aMemoryManager.leavesHost[i].y)
        //    {
        //        cudastd::logger::out << "Bad leaf at position " << i <<" -> ( " << 
        //            aMemoryManager.leavesHost[i].x << " | " << aMemoryManager.leavesHost[i].y << " )\n";
        //    }
        //    else if(aMemoryManager.leavesHost[i].y - aMemoryManager.leavesHost[i].x > 32u )
        //    {
        //        //cudastd::logger::out << "Big leaf at position " << i <<" -> ( " << 
        //        //    aMemoryManager.leavesHost[i].x << " | " << aMemoryManager.leavesHost[i].y << " )\n";
        //    }
        //}
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        MY_CUT_CHECK_ERROR("Setting up leaf cells and primitive array failed.\n");
        //////////////////////////////////////////////////////////////////////////

        //cudastd::logger::out << "Top  level cells:     " << aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ << "\n";
        //cudastd::logger::out << "Top  level refs:      " << numTopLevelPairs << "\n";
        //cudastd::logger::out << "Leaf level cells:     " << numLeafCells << "\n";
        //cudastd::logger::out << "Leaf level refs:      " << numLeafLevelPairs << "\n";
        //cudastd::logger::out << "Allocated memory:     " << (float)aMemoryManager.getMemorySize() / 1048576.f << " MB\n";
        //const float memCells = (float)(aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ + aMemoryManager.leavesSize) / 1048576.f;
        //cudastd::logger::out << "Memory for cells:     " << memCells << " MB\n";
        //const float memRefs = (float)(numLeafLevelPairs * sizeof(uint)) / 1048576.f;
        //cudastd::logger::out << "Memory for refs:      " << memRefs << " MB\n";
        //cudastd::logger::out << "Memory total:         " << memCells + memRefs << " MB\n";


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

#endif // TLGRIDSORTBUILDER_H_085A14AE_437D_424C_BBA8_417665305A48
