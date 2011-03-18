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
#include "Utils/Sort.h"

template<class tPrimitive>
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
        UniformGridMemoryManager&               aMemoryManager,
        const uint                              aNumPrimitives,
        const float&                            aDensity = 5.f
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

    HOST void build(
        UniformGridMemoryManager&                   aMemoryManager,
        PrimitiveArray<tPrimitive>&                 aPrimitiveArray)
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

#if HAPPYRAY__CUDA_ARCH__ >= 120
        const int numCounters = gridTotalSize.x;
#else
        const int numCounters = gridTotalSize.x * blockTotalSize.x;
#endif
        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);
        
        countPairs<tPrimitive, PrimitiveArray, sNUM_COUNTER_THREADS >
            <<< gridTotalSize, blockTotalSize, blockTotalSize.x * (sizeof(uint) + sizeof(float3)) >>>(
            aPrimitiveArray,
            aPrimitiveArray.numPrimitives,
            aMemoryManager.getResolution(), 
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP(),
            aMemoryManager.refCountsBuffer);

        MY_CUT_CHECK_ERROR("Kernel Execution failed.\n");

        cudaEventRecord(mRefCount, 0);
        cudaEventSynchronize(mRefCount);
        
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
        aMemoryManager.allocatePairsBufferPair(numPairs);



        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

        writePairs<tPrimitive, PrimitiveArray >
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aPrimitiveArray,
            aMemoryManager.pairsBuffer,
            aPrimitiveArray.numPrimitives,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        MY_CUT_CHECK_ERROR("Kernel Execution failed.\n");

        cudaEventRecord(mWritePairs, 0);
        cudaEventSynchronize(mWritePairs);

        const uint numCellsPlus1 = aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ;
        uint numBits = 9u;
        while (numCellsPlus1 >> numBits != 0u){numBits += 1u;}
        numBits = cudastd::min(32u, numBits + 1u);

        Sort radixSort;
        radixSort((uint2*)aMemoryManager.pairsBuffer, (uint2*)aMemoryManager.pairsPingBuffer, numPairs, numBits);

        cudaEventRecord(mSort, 0);
        cudaEventSynchronize(mSort);

        aMemoryManager.allocatePrimitiveIndicesBuffer(numPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.primitiveIndices,
            (uint2*)aMemoryManager.pairsBuffer,
            numPairs,
            aMemoryManager.cellsPtrDevice,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ)
            );

        MY_CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        //////////////////////////////////////////////////////////////////////////

        cudastd::logger::out << "Number of pairs:" << numPairs << "\n";
        outputStats();

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

#endif // UGRIDSORTBUILDER_H_INCLUDED_BAE12F51_83A9_47AC_9A00_C1EBFC7062BC
