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

#ifndef UGRIDSORTBUILDER_H_INCLUDED_BAE12F51_83A9_47AC_9A00_C1EBFC7062BC
#define UGRIDSORTBUILDER_H_INCLUDED_BAE12F51_83A9_47AC_9A00_C1EBFC7062BC

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/UniformGrid.h"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UniformGridBuildKernels.h"

#include "Utils/Scan.h"
#define CUB_SORT
#ifdef CUB_SORT
#include "Utils/Sort.h"
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif 

template<class tPrimitive, bool taExactTriangleInsertion = false>
class UGridSortBuilder
{
    static const uint   sNUM_COUNTER_THREADS    = 128u;
    static const uint   sNUM_COUNTER_BLOCKS     =  90u;
    //NOTE: WRITE and COUNTER threads and blocks have to be exactly the same
    static const uint   sNUM_WRITE_THREADS      = sNUM_COUNTER_THREADS;
    static const uint   sNUM_WRITE_BLOCKS       = sNUM_COUNTER_BLOCKS;
    static const uint   sNUM_CELL_SETUP_THREADS = 256u;
    static const uint   sNUM_CELL_SETUP_BLOCKS  =  90u;

    uint mNumPrimitives;
    cudaEvent_t mStart, mDataUpload;
    cudaEvent_t mRefCount, mScan, mWritePairs, mSort, mEnd;
public:
    HOST void init(
        UGridMemoryManager&             aMemoryManager,
        const uint                      aNumPrimitives,
        const float                     aDensity = 5.f,
        const float                     aDummy = 1.2f
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        //////////////////////////////////////////////////////////////////////////

        float3 diagonal = aMemoryManager.bounds.diagonal();

        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = aDensity;
        const float magicConstant =
            powf(lambda * static_cast<float>(aNumPrimitives) / volume, 0.3333333f);

        diagonal *= magicConstant;

        aMemoryManager.resX = static_cast<int>(diagonal.x);
        aMemoryManager.resY = static_cast<int>(diagonal.y);
        aMemoryManager.resZ = static_cast<int>(diagonal.z);

        aMemoryManager.allocateDeviceCells();
        aMemoryManager.setDeviceCellsToZero();

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }
    //dummy to use instead of init
    HOST void initEvents()
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }


    HOST void build(
        UGridMemoryManager&                 aMemoryManager,
        PrimitiveArray<tPrimitive>&         aPrimitiveArray)
    {
        //////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&mRefCount);
        cudaEventCreate(&mScan);
        cudaEventCreate(&mWritePairs);
        cudaEventCreate(&mSort);
        cudaEventCreate(&mEnd);
        //////////////////////////////////////////////////////////////////////////
        
        dim3 blockTotalSize(sNUM_COUNTER_THREADS);
        dim3 gridTotalSize (sNUM_COUNTER_BLOCKS);

#if HAPPYRAY__CUDA_ARCH__ >= 120  && !defined STABLE_PRIMITIVE_ORDERING
        const int numCounters = gridTotalSize.x;
#else
        const int numCounters = gridTotalSize.x * blockTotalSize.x;
#endif
        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);
        
        countPairs<tPrimitive, PrimitiveArray<tPrimitive>, sNUM_COUNTER_THREADS >
            <<< gridTotalSize, blockTotalSize, blockTotalSize.x * (sizeof(uint) + sizeof(float3)) >>>(
            aPrimitiveArray,
            (uint)aPrimitiveArray.numPrimitives,
            aMemoryManager.getResolution(), 
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP(),
            aMemoryManager.refCountsBuffer);

        cudaEventRecord(mRefCount, 0);
        //cudaEventSynchronize(mRefCount);
        //MY_CUT_CHECK_ERROR("Counting primitive-cell pairs failed.\n");
        
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

#if HAPPYRAY__CUDA_ARCH__ < 120 || defined STABLE_PRIMITIVE_ORDERING
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        cudaEventRecord(mScan, 0);
        cudaEventSynchronize(mScan);

        /////////////////////////////////////////////////////////////////////////
        //DEBUG
        //cudastd::logger::out << "Scanned counts:";
        //for(size_t i = 0; i <= numCounters; ++i)
        //{
        //    cudastd::logger::out << " " <<  aMemoryManager.refCountsBufferHost[i];
        //}
        //cudastd::logger::out << "\n ----------------------\n";
        /////////////////////////////////////////////////////////////////////////
        const uint& numPairs = aMemoryManager.refCountsBufferHost[numCounters];

        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

        aMemoryManager.allocateKeyValueBuffers(numPairs);
        writeKeysAndValues<tPrimitive, PrimitiveArray<tPrimitive>, taExactTriangleInsertion>
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aPrimitiveArray,
            aMemoryManager.pairsPingBufferKeys,
            aMemoryManager.pairsPingBufferValues,
            (uint)aPrimitiveArray.numPrimitives,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        cudaEventRecord(mWritePairs, 0);
        cudaEventSynchronize(mWritePairs);
        MY_CUT_CHECK_ERROR("Writing primitive-cell pairs failed.\n");

        ////////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* keysBufferHost0;
        //MY_CUDA_SAFE_CALL(cudaMallocHost((void**)&keysBufferHost0, numPairs * sizeof(uint)));

        //MY_CUDA_SAFE_CALL(cudaMemcpy(keysBufferHost0, aMemoryManager.pairsPingBufferKeys, numPairs * sizeof(uint), cudaMemcpyDeviceToHost));

        //uint* valuesBufferHost0;
        //MY_CUDA_SAFE_CALL(cudaMallocHost((void**)&valuesBufferHost0, numPairs * sizeof(uint)));

        //MY_CUDA_SAFE_CALL(cudaMemcpy(valuesBufferHost0, aMemoryManager.pairsPingBufferValues, numPairs * sizeof(uint), cudaMemcpyDeviceToHost));

        //for (uint i = 0; i < numPairs - 1; ++i)
        //{
        //    //Check if primitives inside a cell are ordered correctly
        //    if (valuesBufferHost0[i] > valuesBufferHost0[i + 1])
        //    {
        //        cudastd::logger::out << "Unsorted primitive ids (before sort) ( " << keysBufferHost0[i] << "," <<
        //            valuesBufferHost0[i] << " ) ";
        //        cudastd::logger::out << " and ( " << keysBufferHost0[i + 1] << "," <<
        //            valuesBufferHost0[i + 1] << " ) ";
        //        cudastd::logger::out << " at postion " << i << " and " << i + 1 << " out of " << numPairs;
        //        cudastd::logger::out << "\n";
        //        break;
        //    }
        //}
        ////////////////////////////////////////////////////////////////////////////

#ifdef CUB_SORT 
        aMemoryManager.allocateKeyValuePongBuffers(numPairs);
        RadixSort radixSort;
        //radixSort(aMemoryManager.pairsPingBufferValues, aMemoryManager.pairsPongBufferValues, 
        //    aMemoryManager.pairsPingBufferKeys, aMemoryManager.pairsPongBufferKeys,            
        //    numPairs);

        radixSort(aMemoryManager.pairsPingBufferKeys, aMemoryManager.pairsPongBufferKeys,
            aMemoryManager.pairsPingBufferValues, aMemoryManager.pairsPongBufferValues,
            numPairs);
#else
        thrust::device_ptr<unsigned int> dev_keys(aMemoryManager.pairsPingBufferKeys);
        thrust::device_ptr<unsigned int> dev_values(aMemoryManager.pairsPingBufferValues);
        thrust::sort_by_key(dev_keys, (dev_keys + numPairs), dev_values);
#endif
        cudaEventRecord(mSort, 0);
        cudaEventSynchronize(mSort);
        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");


        ////////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* keysBufferHost;
        //MY_CUDA_SAFE_CALL(cudaMallocHost((void**)&keysBufferHost, numPairs * sizeof(uint)));

        //MY_CUDA_SAFE_CALL(cudaMemcpy(keysBufferHost, aMemoryManager.pairsPingBufferKeys, numPairs * sizeof(uint), cudaMemcpyDeviceToHost));

        //uint* valuesBufferHost;
        //MY_CUDA_SAFE_CALL(cudaMallocHost((void**)&valuesBufferHost, numPairs * sizeof(uint)));

        //MY_CUDA_SAFE_CALL(cudaMemcpy(valuesBufferHost, aMemoryManager.pairsPingBufferValues, numPairs * sizeof(uint), cudaMemcpyDeviceToHost));

        //for(uint i = 0; i < numPairs - 1; ++i)
        //{
        //    //Check if properly sorted
        //    if (keysBufferHost[i] > keysBufferHost[i + 1])
        //    {
        //        cudastd::logger::out << "Unsorted pairs ( " << keysBufferHost[i] << "," <<
        //            valuesBufferHost[i] << " ) ";
        //        cudastd::logger::out << " and ( " << keysBufferHost[i + 1] << "," <<
        //            valuesBufferHost[i + 1] << " ) ";
        //        cudastd::logger::out << " at postion " << i << " and " << i + 1 << " out of " << numPairs;
        //        cudastd::logger::out << "\n";

        //    }
        //    //Check for cell and primitive indices that make sense
        //    if (keysBufferHost[i] > (uint)aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ ||
        //        valuesBufferHost[i] > aPrimitiveArray.numPrimitives)
        //    {
        //        cudastd::logger::out << "( " << keysBufferHost[i] << "," <<
        //            valuesBufferHost[i] << " ) ";
        //        cudastd::logger::out << "\n";
        //    }

        //    //Check if primitives inside a cell are ordered correctly
        //    if (keysBufferHost[i] == keysBufferHost[i + 1] &&
        //        valuesBufferHost[i] > valuesBufferHost[i + 1])
        //    {
        //        cudastd::logger::out << "Unsorted primitive ids ( " << keysBufferHost[i] << "," <<
        //            valuesBufferHost[i] << " ) ";
        //        cudastd::logger::out << " and ( " << keysBufferHost[i + 1] << "," <<
        //            valuesBufferHost[i + 1] << " ) ";
        //        cudastd::logger::out << " at postion " << i << " and " << i + 1 << " out of " << numPairs;
        //        cudastd::logger::out << "\n";
        //        break;
        //    }

        //}

        //uint2* cellsOnHost;
        //MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&cellsOnHost, (uint)aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ * sizeof(uint2)));
        //for(uint k = 0; k < (uint)aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ; ++k)
        //{
        //    cellsOnHost[k] = make_uint2(0u, 0u);
        //}

        ////Build the grid cells on the host
        //for(uint bla = 0; bla < numPairs; bla += 1)
        //{
        //    if (bla < numPairs - 1  && pairsBufferHost[2 * bla] != pairsBufferHost[2 * bla + 2])
        //    {
        //        cellsOnHost[pairsBufferHost[2 * bla]].y = bla + 1;
        //    }

        //    if (bla > 0 && pairsBufferHost[2 * bla - 2] != pairsBufferHost[2 * bla])
        //    {
        //        cellsOnHost[pairsBufferHost[2 * bla]].x = bla;
        //    }
        //}
        ////set second elem of last pair
        //cellsOnHost[pairsBufferHost[2 * numPairs - 2]].y = numPairs;
        ////////////////////////////////////////////////////////////////////////////


        aMemoryManager.allocatePrimitiveIndicesBuffer(numPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.primitiveIndices,
            aMemoryManager.pairsPingBufferKeys,
            aMemoryManager.pairsPingBufferValues,
            numPairs,
            aMemoryManager.cellsPtrDevice,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ)
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        MY_CUT_CHECK_ERROR("Setting up grid cells failed.\n");
        //////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //DEBUG
        //dim3 blockChkCells(aMemoryManager.resX);
        //dim3 gridChkCells(aMemoryManager.resY, aMemoryManager.resZ);

        //cudastd::logger::out << aMemoryManager.resX << " " << aMemoryManager.resY << " " << aMemoryManager.resZ << "\n";

        //checkGridCells<<< gridPrepRng, blockChkCells >>>
        //    (aPrimitiveArray,
        //    aMemoryManager.primitiveIndices,
        //    aMemoryManager.cellsPtrDevice,
        //    aMemoryManager.getResolution());

        //MY_CUT_CHECK_ERROR("Checking grid cells failed.\n");

        //aMemoryManager.allocateHostCells();
        //aMemoryManager.copyCellsDeviceToHost();

        ////Compare the cells built on the host to those built on the device
        //for(uint k = 0; k < (uint)aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ; ++k)
        //{
        //    if (cellsOnHost[k].x != ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].x ||
        //        cellsOnHost[k].y != ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].y ||
        //        ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].x > numPairs ||
        //        ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].y > numPairs ||
        //        ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].x > ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].y)
        //    {
        //        cudastd::logger::out << "index : " << k << "\n";
        //        cudastd::logger::out << "h( " << cellsOnHost[k].x << "," << cellsOnHost[k].y << " ) ";
        //        cudastd::logger::out << "d( " << ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].x << "," << 
        //            ((uint2*)aMemoryManager.cellsPtrHost.ptr)[k].y << " ) ";
        //        cudastd::logger::out << "\n";
        //    }           
        //}
        //uint* primitiveIndicesHost;
        //MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&primitiveIndicesHost, numPairs * sizeof(uint)));
        //MY_CUDA_SAFE_CALL( cudaMemcpy( primitiveIndicesHost, aMemoryManager.primitiveIndices, aMemoryManager.primitiveIndicesSize,  cudaMemcpyDeviceToHost));

        ////Check for primitive indices that make sense
        //for(uint i = 0; i < numPairs; ++i)
        //{
        //    if(primitiveIndicesHost[i] > aPrimitiveArray.numPrimitives)
        //    {
        //        cudastd::logger::out << "Invalid primitive indirection at position " << i;
        //        cudastd::logger::out << " primitive id is " << primitiveIndicesHost[i];
        //        cudastd::logger::out << " number of primitives is " << aPrimitiveArray.numPrimitives;
        //        cudastd::logger::out << " comming from pair (" << pairsBufferHost[2 * i] << ", " << pairsBufferHost[2 * i + 1] << ")";
        //        cudastd::logger::out << "\n";
        //    }
        //}
        //aMemoryManager.bindDeviceDataToTexture();
        ////////////////////////////////////////////////////////////////////////////

        //cudastd::logger::out << "Number of pairs:" << numPairs << "\n";
        //outputStats();

        cleanup();
    }
    
    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mRefCount);
        cudaEventDestroy(mScan);
        cudaEventDestroy(mWritePairs);
        cudaEventDestroy(mSort);
        cudaEventDestroy(mEnd);
    }

    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::logger::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::logger::out << "Data upload:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mRefCount);
        cudastd::logger::out << "Reference Count:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mRefCount, mScan);
        cudastd::logger::out << "Scan:             " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mScan, mWritePairs);
        cudastd::logger::out << "Write pairs:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mWritePairs, mSort);
        cudastd::logger::out << "Sort pairs:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mSort, mEnd);
        cudastd::logger::out << "Setup grid cells: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::logger::out << "Total:            " << elapsedTime << "ms\n";
        //////////////////////////////////////////////////////////////////////////
    }
};

#ifdef CUB_SORT
#undef CUB_SORT
#endif

#endif // UGRIDSORTBUILDER_H_INCLUDED_BAE12F51_83A9_47AC_9A00_C1EBFC7062BC
