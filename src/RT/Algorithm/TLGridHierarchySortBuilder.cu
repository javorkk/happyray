#include "CUDAStdAfx.h"
#include "RT/Algorithm/TLGridHierarchySortBuilder.h"

#include "RT/Structure/TwoLevelGridHierarchy.h"
#include "RT/Algorithm/UniformGridBuildKernels.h"
#include "RT/Algorithm/TLGridBuildKernels.h"

#include "Utils/Scan.h"
#include "Utils/Sort.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

            float3 resolution = diagonal * magicConstant;
            int resX = static_cast<int>(resolution.x);
            int resY = static_cast<int>(resolution.y);
            int resZ = static_cast<int>(resolution.z);
            oGirds[gridId].res[0] = resX > 0 ? resX : 1;
            oGirds[gridId].res[1] = resY > 0 ? resY : 1;
            oGirds[gridId].res[2] = resZ > 0 ? resZ : 1;
            resolution = make_float3(oGirds[gridId].res[0], oGirds[gridId].res[1], oGirds[gridId].res[2]);
            oGirds[gridId].setCellSize(diagonal / resolution);
            oGirds[gridId].setCellSizeRCP(resolution / diagonal);

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
GLOBAL void writeKeysAndValuesMultiUniformGrid(
    PrimitiveArray<tPrimitive>  aPrimitiveArray,
    UniformGrid*                aGirds,
    const uint                  aNumGrids,
    uint*                       aScannedPrimitivesPerGrid,
    uint*                       aStartId,
    uint*                       oKeys,
    uint*                       oValues
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
                    oKeys[nextSlot] = x + y * aGirds[gridId].res[0] +
                        z * aGirds[gridId].res[0] * aGirds[gridId].res[1] +
                        cellOffset;
                    oValues[nextSlot] = primId;
                }//end for z
            }//end for y
        }//end for x


    }//end  for(uint refId = globalThreadId1D(); ...

}

    HOST void TLGridHierarchySortBuilder::init(
        TLGridHierarchyMemoryManager&   aMemoryManager,
        const uint                      aNumInstances,
        const float                     aTopLevelDensity,
        const float                     aLeafLevelDensity,
        const bool                      aResetGridResolution
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        cudaEventSynchronize(mStart);
        //////////////////////////////////////////////////////////////////////////

        if(aResetGridResolution)
        {
            float3 diagonal = aMemoryManager.bounds.diagonal();

            const float volume = diagonal.x * diagonal.y * diagonal.z;
            const float lambda = aTopLevelDensity;
            const float magicConstant =
                powf(lambda * static_cast<float>(aNumInstances) / volume, 0.3333333f);

            diagonal *= magicConstant;

            aMemoryManager.resX = cudastd::max(1, static_cast<int>(diagonal.x));
            aMemoryManager.resY = cudastd::max(1, static_cast<int>(diagonal.y));
            aMemoryManager.resZ = cudastd::max(1, static_cast<int>(diagonal.z));

            mSetResolution = true;
        }
        else
        {
            mSetResolution = false;
        }

        aMemoryManager.topLevelDensity = aTopLevelDensity;
        aMemoryManager.leafLevelDensity = aLeafLevelDensity;

        aMemoryManager.allocateDeviceCells();
        aMemoryManager.setDeviceCellsToZero();
        
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void TLGridHierarchySortBuilder::build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        PrimitiveArray<Triangle>&         aPrimitiveArray)
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
            //hostInstances[i].rotation0   = make_float3(1.f, 0.f, 0.f);
            //hostInstances[i].rotation1   = make_float3(0.f, 1.f, 0.f);
            //hostInstances[i].rotation2   = make_float3(0.f, 0.f, 1.f);
            //hostInstances[i].translation = make_float3(0.f, 0.f, 0.f); //aMemoryManager.bounds.diagonal() * (float) (i / 4);
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

    HOST void TLGridHierarchySortBuilder::build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint                                aNumUniqueInstances,
        uint*                               aPrimitiveCounts,
        PrimitiveArray<Triangle>&         aPrimitiveArray)
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
        //aMemoryManager.allocateTopLevelPairsBufferPair(numTopLevelPairs);
        
        aMemoryManager.allocateTopLevelKeyValueBuffers(numTopLevelPairs);


        dim3 blockUnsortedGrid(sNUM_WRITE_THREADS);
        dim3 gridUnsortedGrid (sNUM_WRITE_BLOCKS);

        writeKeysAndValues<GeometryInstance, GeometryInstance*, false>
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint)/* + sizeof(float3) * blockUnsortedGrid.x*/ >>>(
            aMemoryManager.instancesDevice,
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
            numInstances,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.getResolution(),
            aMemoryManager.bounds.vtx[0],
            aMemoryManager.getCellSize(),
            aMemoryManager.getCellSizeRCP());

        MY_CUT_CHECK_ERROR("Writing primitive-cell pairs failed.\n");


        //const uint numCellsPlus1 = aMemoryManager.resX * aMemoryManager.resY * aMemoryManager.resZ;
        //uint numBits = 9u;
        //while (numCellsPlus1 >> numBits != 0u){numBits += 1u;}
        //numBits = cudastd::min(32u, numBits + 1u);

        //Sort radixSort;
        //radixSort(aMemoryManager.topLevelPairsBuffer, aMemoryManager.topLevelPairsPingBufferKeys, numTopLevelPairs, numBits);

        thrust::device_ptr<unsigned int> dev_keys(aMemoryManager.topLevelPairsPingBufferKeys);
        thrust::device_ptr<unsigned int> dev_values(aMemoryManager.topLevelPairsPingBufferValues);
        thrust::sort_by_key(dev_keys, (dev_keys + numTopLevelPairs), dev_values);

        MY_CUT_CHECK_ERROR("Sorting primitive-cell pairs failed.\n");

        aMemoryManager.allocateInstanceIndices(numTopLevelPairs);

        dim3 blockPrepRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepRng (sNUM_CELL_SETUP_BLOCKS);

        prepareCellRanges< sNUM_CELL_SETUP_THREADS >
            <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            aMemoryManager.instanceIndicesDevice,
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
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
        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //test(aMemoryManager, aNumUniqueInstances, aPrimitiveCounts, aPrimitiveArray);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void TLGridHierarchySortBuilder::buildLevelTwo(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint*                               aPrimitiveCounts,
        const uint                          aNumGrids,
        PrimitiveArray<Triangle>&         aPrimitiveArray)
    {


        aMemoryManager.allocateCellCountsBuffer(aNumGrids + 1);
        MY_CUDA_SAFE_CALL( cudaMemcpy(aMemoryManager.cellCountsBuffer, aPrimitiveCounts , aNumGrids * sizeof(uint), cudaMemcpyHostToDevice) );
        
        dim3 blockCellCount(sNUM_COUNTER_THREADS);
        dim3 gridCellCount(sNUM_COUNTER_BLOCKS);

        countGridCellsMultiUniformGrid<<< gridCellCount, blockCellCount >>>(
            aNumGrids,
            aMemoryManager.leafLevelDensity,
            aMemoryManager.gridsDevice,
            mSetResolution,
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


        countPairsMultiUniformGrid<Triangle, sNUM_COUNTER_THREADS > 
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

        //aMemoryManager.allocateLeafLevelPairsBufferPair(numLeafLevelPairs);
        aMemoryManager.allocateLeafLevelKeyValueBuffers(numLeafLevelPairs);

        dim3 blockRefWrite = sNUM_WRITE_THREADS;
        dim3 gridRefWrite  = sNUM_WRITE_BLOCKS;

        writeKeysAndValuesMultiUniformGrid<Triangle>
            <<< gridRefWrite, blockRefWrite,  sizeof(uint)>>>(
            aPrimitiveArray,            
            aMemoryManager.gridsDevice,
            aNumGrids,
            aMemoryManager.cellCountsBuffer,
            aMemoryManager.refCountsBuffer,
            aMemoryManager.leafLevelPairsPingBufferKeys,
            aMemoryManager.leafLevelPairsPingBufferValues
            );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        cudaEventSynchronize(mLeafRefsWrite);
        MY_CUT_CHECK_ERROR("Writing the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////


        //uint numBits = 7u;
        //while (numLeafCells >> numBits != 0u){numBits += 1u;}
        //numBits = cudastd::min(32u, numBits + 1u);

        //Sort radixSort;
        //radixSort(aMemoryManager.leafLevelPairsBuffer, aMemoryManager.leafLevelPairsPingBufferKeys, numLeafLevelPairs, numBits);

        thrust::device_ptr<unsigned int> dev_keys(aMemoryManager.leafLevelPairsPingBufferKeys);
        thrust::device_ptr<unsigned int> dev_values(aMemoryManager.leafLevelPairsPingBufferValues);
        thrust::sort_by_key(dev_keys, (dev_keys + numLeafLevelPairs), dev_values);


        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        cudaEventSynchronize(mSortLeafPairs);
        MY_CUT_CHECK_ERROR("Sorting the leaf level pairs failed.\n");
        //////////////////////////////////////////////////////////////////////////

        aMemoryManager.allocatePrimitiveIndicesBuffer(numLeafLevelPairs);

        dim3 blockPrepLeafRng(sNUM_CELL_SETUP_THREADS);
        dim3 gridPrepLeafRng (sNUM_CELL_SETUP_BLOCKS );

        prepareLeafCellRanges<sNUM_CELL_SETUP_BLOCKS>
            <<< gridPrepLeafRng, blockPrepLeafRng,
            (2 + blockPrepLeafRng.x) * sizeof(uint) >>>(
            aMemoryManager.primitiveIndices,
            aMemoryManager.topLevelPairsPingBufferKeys,
            aMemoryManager.topLevelPairsPingBufferValues,
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

    HOST void TLGridHierarchySortBuilder::test(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint                                aNumUniqueInstances,
        uint*                               aPrimitiveCounts,
        PrimitiveArray<Triangle>&         aPrimitiveArray)

    {
        //////////////////////////////////////////////////////////////////////////
        //download data
        //////////////////////////////////////////////////////////////////////////
        aMemoryManager.allocateHostCells();
        aMemoryManager.copyCellsDeviceToHost();
        aMemoryManager.allocateHostInstances(aMemoryManager.instancesSize / sizeof(GeometryInstance));
        aMemoryManager.copyInstancesDeviceToHost();
        aMemoryManager.copyInstanceIndicesDeviceToHost();
        aMemoryManager.allocateHostLeaves(aMemoryManager.leavesSize / sizeof(TLGridHierarchyMemoryManager::t_Leaf));
        aMemoryManager.copyLeavesDeviceToHost();
        aMemoryManager.allocatePrimitiveIndicesBufferHost(aMemoryManager.primitiveIndicesSize / sizeof(uint));
        aMemoryManager.copyPrimitiveIndicesDeviceToHost();
        aMemoryManager.copyGridsDeviceToHost();
        
        MY_CUDA_SAFE_CALL(cudaMemcpy(aPrimitiveArray.indicesBufferHostPtr, aPrimitiveArray.indicesBufferDevicePtr, aPrimitiveArray.indicesBufferSize, cudaMemcpyDeviceToHost));
        MY_CUDA_SAFE_CALL(cudaMemcpy(aPrimitiveArray.vertexBufferHostPtr, aPrimitiveArray.vertexBufferDevicePtr, aPrimitiveArray.vertexBufferSize, cudaMemcpyDeviceToHost));
        //////////////////////////////////////////////////////////////////////////
        //top level test
        //////////////////////////////////////////////////////////////////////////
        const uint numInstanceIndices = (uint)(aMemoryManager.instanceIndicesSize / sizeof(uint));

        TwoLevelGridHierarchy hierarchy = aMemoryManager.getParametersHost();

        for (uint z = 0; z < hierarchy.res[2]; ++z)
        {
            for (uint y = 0; y < hierarchy.res[1]; ++y)
            {
                for (uint x = 0; x < hierarchy.res[0]; ++x)
                {
                    uint2 cell = hierarchy.getCell(x,y,z);
                    for (uint k = cell.x; k < cell.y; ++k)
                    {
                        if (k >= numInstanceIndices)
                        {
                            cudastd::logger::out << "Invalid cell range [" << cell.x << ", " << cell.y << ") at cell " << x << " " << y << " " << z << " !\n";
                            break;
                        }
                        GeometryInstance instance = hierarchy.getInstances()[hierarchy.getInstanceIndices()[k]];
                        BBox bounds = BBoxExtractor<GeometryInstance>::get(instance);

                        const int3 minCellId = hierarchy.getCellIdAt(bounds.vtx[0]);
                        int3 maxCellIdPlus1 = hierarchy.getCellIdAt(bounds.vtx[1]);
                        maxCellIdPlus1.x += 1;
                        maxCellIdPlus1.y += 1;
                        maxCellIdPlus1.z += 1;

                        const int minCellIdX =  min(hierarchy.res[0]-1, max(0, minCellId.x));
                        const int minCellIdY =  min(hierarchy.res[1]-1, max(0, minCellId.y));
                        const int minCellIdZ =  min(hierarchy.res[2]-1, max(0, minCellId.z));

                        const int maxCellIdP1X =  max(1, min(hierarchy.res[0], maxCellIdPlus1.x));
                        const int maxCellIdP1Y =  max(1, min(hierarchy.res[1], maxCellIdPlus1.y));
                        const int maxCellIdP1Z =  max(1, min(hierarchy.res[2], maxCellIdPlus1.z));

                        if (minCellIdX > x || maxCellIdP1X <= x ||
                            minCellIdY > y || maxCellIdP1Y <= y ||
                            minCellIdZ > z || maxCellIdP1Z <= z)
                        {
                            cudastd::logger::out << "Instance " << hierarchy.getInstanceIndices()[k]
                                << " inserted in WRONG CELL " << x << " " << y << " " << z << " !\n";
                            cudastd::logger::out << "Instance " << hierarchy.getInstanceIndices()[k]
                                << " belongs to cells ["
                                << minCellIdX << ", " << maxCellIdP1X << ") [" 
                                << minCellIdY << ", " << maxCellIdP1Y << ") ["
                                << minCellIdZ << ", " << maxCellIdP1Z << ")\n";

                        }
                    }
                }
            }
        }

        for (uint it = 0; it < hierarchy.getNumInstances(); ++it)
        {
            GeometryInstance instance = aMemoryManager.instancesHost[it];
            BBox bounds = BBoxExtractor<GeometryInstance>::get(instance);

            const int3 minCellId = hierarchy.getCellIdAt(bounds.vtx[0]);
            int3 maxCellIdPlus1 = hierarchy.getCellIdAt(bounds.vtx[1]);
            maxCellIdPlus1.x += 1;
            maxCellIdPlus1.y += 1;
            maxCellIdPlus1.z += 1;

            const int minCellIdX =  min(hierarchy.res[0]-1, max(0, minCellId.x));
            const int minCellIdY =  min(hierarchy.res[1]-1, max(0, minCellId.y));
            const int minCellIdZ =  min(hierarchy.res[2]-1, max(0, minCellId.z));

            const int maxCellIdP1X =  max(1, min(hierarchy.res[0], maxCellIdPlus1.x));
            const int maxCellIdP1Y =  max(1, min(hierarchy.res[1], maxCellIdPlus1.y));
            const int maxCellIdP1Z =  max(1, min(hierarchy.res[2], maxCellIdPlus1.z));
            for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
            {
                for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
                {
                    for (uint x = minCellIdX; x < maxCellIdP1X; ++x)
                    {
                        uint2 cell = hierarchy.getCell(x,y,z);
                        bool inserted = false;
                        for (uint k = cell.x; k < cell.y; ++k)
                        {
                            if (k < numInstanceIndices && hierarchy.getInstanceIndices()[k] == it)
                            {
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted)
                        {
                            cudastd::logger::out << "Instance " << it << " NOT INSERTED in cell " << x
                                << " " << y << " " << z << " !\n";
                        }
                    }//end for z
                }//end for y
            }//end for x

        }        

        //////////////////////////////////////////////////////////////////////////
        //leaf level test
        //////////////////////////////////////////////////////////////////////////
        uint* scannedPrimitiveCounts = (uint*)malloc((aNumUniqueInstances + 1)* sizeof(uint));
        memcpy_s(scannedPrimitiveCounts, (aNumUniqueInstances + 1)* sizeof(uint), aPrimitiveCounts, aNumUniqueInstances * sizeof(uint));
        uint sum = 0;
        for (int i = 0; i < aNumUniqueInstances; ++i)
        {
            uint oldsum = sum;
            sum += scannedPrimitiveCounts[i];
            scannedPrimitiveCounts[i] = oldsum;
        }

        scannedPrimitiveCounts[aNumUniqueInstances] = sum;
        size_t cellCounts = 0;
        for (int gridId = 0; gridId < aNumUniqueInstances; ++gridId)
        {
            UniformGrid grid = hierarchy.getGrids()[gridId];
            void* ptr = (void*)(hierarchy.getLeaves() + cellCounts);
            grid.cells.ptr = ptr;
            grid.primitives = aMemoryManager.primitiveIndicesHost;
            cellCounts += grid.res[0] * grid.res[1] * grid.res[2];

            if (grid.vtx[0].x > grid.vtx[1].x ||
                grid.vtx[0].y > grid.vtx[1].y ||
                grid.vtx[0].z > grid.vtx[1].z ||
                grid.vtx[0].x < hierarchy.vtx[0].x ||
                grid.vtx[0].y < hierarchy.vtx[0].y ||
                grid.vtx[0].z < hierarchy.vtx[0].z ||
                grid.vtx[1].x > hierarchy.vtx[1].x ||
                grid.vtx[1].y > hierarchy.vtx[1].y ||
                grid.vtx[1].z > hierarchy.vtx[1].z
                )
            {
                cudastd::logger::out << "Invalid bounds of grid " << gridId
                    << " min ("
                    << grid.vtx[0].x << ", "
                    << grid.vtx[0].y << ", "
                    << grid.vtx[0].z << ") "
                    << ", max ("
                    << grid.vtx[1].x << ", "
                    << grid.vtx[1].y << ", "
                    << grid.vtx[1].z << ") "
                    << ", MIN ("
                    << hierarchy.vtx[0].x << ", "
                    << hierarchy.vtx[0].y << ", "
                    << hierarchy.vtx[0].z << ") "
                    << ", MAX ("
                    << hierarchy.vtx[1].x << ", "
                    << hierarchy.vtx[1].y << ", "
                    << hierarchy.vtx[1].z << ")\n";
            }
            
            float3 cellSize = (grid.vtx[1] - grid.vtx[0]) / grid.getResolution(); 
            float3 cellSizeRCP = grid.getResolution() / (grid.vtx[1] - grid.vtx[0]);

            if (fabsf(grid.getCellSize().x - cellSize.x) > EPS ||
                fabsf(grid.getCellSize().y - cellSize.y) > EPS ||
                fabsf(grid.getCellSize().z - cellSize.z) > EPS ||
                fabsf(grid.getCellSizeRCP().x - cellSizeRCP.x) > EPS ||
                fabsf(grid.getCellSizeRCP().y - cellSizeRCP.y) > EPS ||
                fabsf(grid.getCellSizeRCP().z - cellSizeRCP.z) > EPS 
                )
            {
                cudastd::logger::out << "INVALID CELL SIZE of grid " << gridId
                    << " cell size: ("
                    << cellSize.x << ", "
                    << cellSize.y << ", "
                    << cellSize.z << ") grid cell size: ("
                    << grid.getCellSize().x << ", "
                    << grid.getCellSize().y << ", "
                    << grid.getCellSize().z << ")\n";

            }
            if (grid.res[0] < 1 ||
                grid.res[1] < 1 ||
                grid.res[2] < 1
                )
            {
                cudastd::logger::out << "INVALID RESOLUTION of grid " << gridId
                    << " resolution: ("
                    << grid.res[0] << ", "
                    << grid.res[1] << ", "
                    << grid.res[2] << ")\n";
            }

            for (uint z = 0; z < grid.res[2]; ++z)
            {
                for (uint y = 0; y < grid.res[1]; ++y)
                {
                    for (uint x = 0; x < grid.res[0]; ++x)
                    {
                         uint2 cell = grid.getCell(x,y,z);
                         for (uint k = cell.x; k < cell.y; ++k)
                         {
                             if (k >= aMemoryManager.primitiveIndicesSize / sizeof(uint))
                             {
                                 cudastd::logger::out << "Invalid cell range [" << cell.x << ", " << cell.y << ") at leaf level cell " << x << " " << y << " " << z
                                     << " grid id " << gridId << " !\n";
                                 break;
                             }
                             Triangle prim = aPrimitiveArray[grid.primitives[k]];
                             BBox bounds = BBoxExtractor<Triangle>::get(prim);

                             const int3 minCellId = grid.getCellIdAt(bounds.vtx[0]);
                             int3 maxCellIdPlus1 = grid.getCellIdAt(bounds.vtx[1]);
                             maxCellIdPlus1.x += 1;
                             maxCellIdPlus1.y += 1;
                             maxCellIdPlus1.z += 1;

                             const int minCellIdX =  min(grid.res[0]-1, max(0, minCellId.x));
                             const int minCellIdY =  min(grid.res[1]-1, max(0, minCellId.y));
                             const int minCellIdZ =  min(grid.res[2]-1, max(0, minCellId.z));

                             const int maxCellIdP1X =  max(1, min(grid.res[0], maxCellIdPlus1.x));
                             const int maxCellIdP1Y =  max(1, min(grid.res[1], maxCellIdPlus1.y));
                             const int maxCellIdP1Z =  max(1, min(grid.res[2], maxCellIdPlus1.z));

                             if (minCellIdX > x || maxCellIdP1X <= x ||
                                 minCellIdY > y || maxCellIdP1Y <= y ||
                                 minCellIdZ > z || maxCellIdP1Z <= z)
                             {
                                 cudastd::logger::out << "Primitive " << k
                                 << " inserted in WRONG LEAF CELL " << x << " " << y << " " << z 
                                 << " grid id " << gridId <<" !\n";
                                 cudastd::logger::out << "Primitive " << k
                                 << " belongs to cells ["
                                     << minCellIdX << ", " << maxCellIdP1X << ") [" 
                                     << minCellIdY << ", " << maxCellIdP1Y << ") ["
                                     << minCellIdZ << ", " << maxCellIdP1Z << ")\n";

                             }
                         }
                    }
                }
            }

            for (int primId = scannedPrimitiveCounts[gridId]; primId < scannedPrimitiveCounts[gridId + 1]; ++primId)
            {
                Triangle prim = aPrimitiveArray[primId];
                BBox bounds = BBoxExtractor<Triangle>::get(prim);

                const int3 minCellId = grid.getCellIdAt(bounds.vtx[0]);
                int3 maxCellIdPlus1 = grid.getCellIdAt(bounds.vtx[1]);
                maxCellIdPlus1.x += 1;
                maxCellIdPlus1.y += 1;
                maxCellIdPlus1.z += 1;

                const int minCellIdX =  min(grid.res[0]-1, max(0, minCellId.x));
                const int minCellIdY =  min(grid.res[1]-1, max(0, minCellId.y));
                const int minCellIdZ =  min(grid.res[2]-1, max(0, minCellId.z));

                const int maxCellIdP1X =  max(1, min(grid.res[0], maxCellIdPlus1.x));
                const int maxCellIdP1Y =  max(1, min(grid.res[1], maxCellIdPlus1.y));
                const int maxCellIdP1Z =  max(1, min(grid.res[2], maxCellIdPlus1.z));
                for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
                {
                    for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
                    {
                        for (uint x = minCellIdX; x < maxCellIdP1X; ++x)
                        {
                            uint2 cell = grid.getCell(x,y,z);
                            bool inserted = false;
                            for (uint k = cell.x; k < cell.y; ++k)
                            {
                                if (k < aMemoryManager.primitiveIndicesSize / sizeof(uint) && grid.primitives[k] == primId)
                                {
                                    inserted = true;
                                    break;
                                }
                            }
                            if (!inserted)
                            {
                                cudastd::logger::out << "Primitive " << primId << " NOT INSERTED in leaf cell " << x
                                    << " " << y << " " << z << " grid id " << gridId <<" !\n";
                            }
                        }//end for z
                    }//end for y
                }//end for x
            }
        }
        
        free(scannedPrimitiveCounts);
    }