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
#include "Utils/Sort.h"

template<class tPrimitive>
class TLGridSortBuilder
{
    static const uint   sNUM_COUNTER_THREADS    = 128u;
    static const uint   sNUM_COUNTER_BLOCKS     =  90u;
    //NOTE: WRITE and COUNTER threads and blocks have to be exactly the same
    static const uint   sNUM_WRITE_THREADS      = sNUM_COUNTER_THREADS;
    static const uint   sNUM_WRITE_BLOCKS       = sNUM_COUNTER_BLOCKS;
    static const uint   sNUM_CELL_SETUP_THREADS = 256u;
    static const uint   sNUM_CELL_SETUP_BLOCKS  =  90u;

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


        countPairs<tPrimitive, PrimitiveArray, sNUM_COUNTER_THREADS >
            <<< gridTotalSize, blockTotalSize,
            blockTotalSize.x * (sizeof(uint) + sizeof(float3))>>>(
            aPrimitiveArray,
            aPrimitiveArray.numPrimitives,
            aMemoryManager.getResolution(), 
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP(),
            aMemoryManager.refCountsBuffer);

        MY_CUT_CHECK_ERROR("Counting top level primitive-cell pairs failed.\n");

        ExclusiveScan scan;
        scan(aMemoryManager.refCountsBuffer, numCounters + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        cudaEventRecord(mScan, 0);
        cudaEventSynchronize(mScan);

        const uint& numTopLevelPairs = aMemoryManager.refCountsBufferHost[numCounters];
        aMemoryManager.allocateTopLevelPairsBufferPair(numTopLevelPairs);


        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

        writePairs<tPrimitive, PrimitiveArray >
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aPrimitiveArray,
            aMemoryManager.topLevelPairsBuffer,
            aPrimitiveArray.numPrimitives,
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
        radixSort((uint2*)aMemoryManager.topLevelPairsBuffer, (uint2*)aMemoryManager.topLevelPairsPingBuffer, numTopLevelPairs, numBits);

        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");

        aMemoryManager.allocatePrimitiveIndicesBuffer(numTopLevelPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.primitiveIndices,
            (uint2*)aMemoryManager.topLevelPairsBuffer,
            numTopLevelPairs,
            aMemoryManager.cellsPtrDevice,
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ)
            );

        MY_CUT_CHECK_ERROR("Setting up top level cells failed.\n");

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mTopLevel, 0);
        cudaEventSynchronize(mTopLevel);
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

        MY_CUT_CHECK_ERROR("Counting leaf level cells failed.\n");

        ExclusiveScan escan;
        escan(aMemoryManager.cellCountsBuffer, aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBufferHost + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ, (aMemoryManager.refCountsBuffer + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafCellCount, 0);
        cudaEventSynchronize(mLeafCellCount);
        //////////////////////////////////////////////////////////////////////////
        const uint numLeafCells = *(aMemoryManager.cellCountsBufferHost + aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ);

        dim3 blockRefCount = sNUM_COUNTER_THREADS;
        dim3 gridRefCount  = sNUM_COUNTER_BLOCKS;

        aMemoryManager.allocateRefCountsBuffer(numCounters + 1);

        countLeafLevelPairs<tPrimitive, PrimitiveArray, sNUM_COUNTER_THREADS > 
            <<< gridRefCount, blockRefCount,  blockRefCount.x * (sizeof(uint) + sizeof(float3)) >>>(
            aPrimitiveArray,
            numTopLevelPairs,
            (uint2*)aMemoryManager.topLevelPairsBuffer,
            aMemoryManager.cellsPtrDevice,
            //oGrid.getResolution(),
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

        MY_CUT_CHECK_ERROR("Counting leaf level pairs failed.\n");

        scan(aMemoryManager.refCountsBuffer, numCounters + 1);

#if HAPPYRAY__CUDA_ARCH__ < 120
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.refCountsBufferHost + numCounters, (aMemoryManager.refCountsBuffer + numCounters), sizeof(uint), cudaMemcpyDeviceToHost) );
#endif
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsCount, 0);
        cudaEventSynchronize(mLeafRefsCount);
        //////////////////////////////////////////////////////////////////////////
        const uint numLeafLevelPairs = aMemoryManager.refCountsBufferHost[numCounters];

        aMemoryManager.allocateLeafLevelPairsBufferPair(numLeafLevelPairs);

        dim3 blockRefWrite = sNUM_WRITE_THREADS;
        dim3 gridRefWrite  = sNUM_WRITE_BLOCKS;

        writeLeafLevelPairs<tPrimitive, PrimitiveArray>
            <<< gridRefWrite, blockRefWrite,  sizeof(uint) + sizeof(float3) * blockRefWrite.x >>>(
            aPrimitiveArray,
            numTopLevelPairs,
            (uint2*)aMemoryManager.topLevelPairsBuffer,
            aMemoryManager.cellsPtrDevice,
            numLeafCells,
            aMemoryManager.refCountsBuffer,
            //oGrid.getResolution(),
            static_cast<uint>(aMemoryManager.resX),
            static_cast<uint>(aMemoryManager.resY),
            static_cast<uint>(aMemoryManager.resZ),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.leafLevelPairsBuffer
            );

        MY_CUT_CHECK_ERROR("Writing leaf level pairs failed.\n");
        
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        cudaEventSynchronize(mLeafRefsWrite);
        //////////////////////////////////////////////////////////////////////////

        numBits = 7u;
        while (numLeafCells >> numBits != 0u){numBits += 1u;}
        numBits = cudastd::min(32u, numBits + 1u);

        radixSort((uint2*)aMemoryManager.leafLevelPairsBuffer, (uint2*)aMemoryManager.leafLevelPairsPingBuffer, numLeafLevelPairs, numBits);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        cudaEventSynchronize(mSortLeafPairs);
        //////////////////////////////////////////////////////////////////////////

        aMemoryManager.allocateDeviceLeaves(numLeafCells);
        aMemoryManager.setDeviceLeavesToZero();

        dim3 blockPrepLeafRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepLeafRng (sNUM_CELL_SETUP_BLOCKS );

        prepareLeafCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepLeafRng, blockPrepLeafRng,
            (2 + blockPrepRng.x) * sizeof(uint) >>>(
            aMemoryManager.primitiveIndices,
            (uint2*)aMemoryManager.leafLevelPairsBuffer,
            numLeafLevelPairs,
            (uint2*)aMemoryManager.leavesDevice
            );

        MY_CUT_CHECK_ERROR("Setting up leaf cells and primitive array failed.\n");

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        //////////////////////////////////////////////////////////////////////////

        cudastd::logger::out << "Top  level cells:     " << aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ << "\n";
        cudastd::logger::out << "Top  level refs:      " << numTopLevelPairs << "\n";
        cudastd::logger::out << "Leaf level cells:     " << numLeafCells << "\n";
        cudastd::logger::out << "Leaf level refs:      " << numLeafLevelPairs << "\n";
        cudastd::logger::out << "Allocated memory:     " << (float)aMemoryManager.getMemorySize() / 1048576.f << " MB\n";
        const float memCells = (float)(aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ + aMemoryManager.leavesSize) / 1048576.f;
        cudastd::logger::out << "Memory for cells:     " << memCells << " MB\n";
        const float memRefs = (float)(numLeafLevelPairs * sizeof(uint)) / 1048576.f;
        cudastd::logger::out << "Memory for refs:      " << memRefs << " MB\n";
        cudastd::logger::out << "Memory total:         " << memCells + memRefs << " MB\n";


        outputStats();
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