#include "StdAfx.hpp"
#include "CUDAStdAfx.h"
#include "RT/Structure/TLGridMemoryManager.h"
#include "RT/Structure/MemoryManager.h"

//////////////////////////////////////////////////////////////////////////
//data transfer related
//////////////////////////////////////////////////////////////////////////

HOST void TLGridMemoryManager::copyCellsDeviceToHost()
{
    cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
    cpyParamsDownloadPtr.srcPtr  = cellsPtrDevice;
    cpyParamsDownloadPtr.dstPtr  = cellsPtrHost;
    cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
    cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
}

HOST void TLGridMemoryManager::copyCellsHostToDevice()
{
    cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
    cpyParamsUploadPtr.srcPtr  = cellsPtrHost;
    cpyParamsUploadPtr.dstPtr  = cellsPtrDevice;
    cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
    cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
}


//////////////////////////////////////////////////////////////////////////
//memory allocation
//////////////////////////////////////////////////////////////////////////
HOST cudaPitchedPtr TLGridMemoryManager::allocateHostCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ)
    {
        return cellsPtrHost;
    }

    freeCellMemoryHost();

    t_Cell* cpuCells = NULL;
    MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
        resX * resY * resZ * sizeof(t_Cell)));

    cellsPtrHost = 
        make_cudaPitchedPtr(cpuCells, resX * sizeof(t_Cell), resX, resY);

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;


    return cellsPtrHost;
}

HOST cudaPitchedPtr TLGridMemoryManager::allocateDeviceCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ)
    {
        return cellsPtrDevice;
    }

    freeCellMemoryDevice();

    t_Cell* gpuCells = NULL;
    cellsPtrDevice =
        make_cudaPitchedPtr(gpuCells, resX * sizeof(t_Cell), resX, resY);

    cudaExtent cellDataExtent = 
        make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);

    MY_CUDA_SAFE_CALL( cudaMalloc3D(&cellsPtrDevice, cellDataExtent) );

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;


    return cellsPtrDevice;
}

HOST void TLGridMemoryManager::setDeviceCellsToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(cellsPtrDevice.ptr, 0 ,
        cellsPtrDevice.pitch * resY * resZ ) );

    //does not work!
    //cudaExtent cellDataExtent = 
    //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
    //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
}

TLGridMemoryManager::t_Leaf* TLGridMemoryManager::allocateHostLeaves(const size_t aNumLeaves)
{
    
    leavesSize = aNumLeaves * sizeof(t_Leaf);
    MY_CUDA_SAFE_CALL( cudaFreeHost(leavesHost) );
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&leavesHost,
        leavesSize, cudaHostAllocDefault));

    return leavesHost;
}

HOST void TLGridMemoryManager::setDeviceLeavesToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(leavesDevice, 0 ,leavesSize) );

    //does not work!
    //cudaExtent cellDataExtent = 
    //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
    //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
}

TLGridMemoryManager::t_Leaf* TLGridMemoryManager::allocateDeviceLeaves(const size_t aNumLeaves)
{
     MemoryManager::allocateDeviceArray((void**)&leavesDevice, aNumLeaves * sizeof(t_Leaf),
        (void**)&leavesDevice, leavesSize);

    return leavesDevice;
}

HOST uint* TLGridMemoryManager::allocatePrimitiveIndicesBuffer(const size_t aNumIndices)
{
    MemoryManager::allocateDeviceArray((void**)&primitiveIndices, aNumIndices * sizeof(uint),
        (void**)&primitiveIndices, primitiveIndicesSize);

    return primitiveIndices;
}

HOST void TLGridMemoryManager::allocateRefCountsBuffer(const size_t aNumSlots)
{
    MemoryManager::allocateMappedDeviceArray(
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, aNumSlots * sizeof(uint),
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, refCountsBufferSize);

    MY_CUDA_SAFE_CALL( cudaMemset(refCountsBuffer + aNumSlots - 1, 0, sizeof(uint)) );
}

HOST void TLGridMemoryManager::allocateCellCountsBuffer(const size_t aNumCells)
{
    MemoryManager::allocateMappedDeviceArray(
        (void**)&cellCountsBuffer, (void**)&cellCountsBufferHost, aNumCells * sizeof(uint),
        (void**)&cellCountsBuffer, (void**)&cellCountsBufferHost, cellCountsBufferSize);

    MY_CUDA_SAFE_CALL( cudaMemset(cellCountsBuffer + aNumCells - 1, 0, sizeof(uint)) );
}


HOST void TLGridMemoryManager::allocateTopLevelPairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBuffer, aNumPairs * sizeof(uint2),
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBuffer, topLevelPairsBufferSize);
}

HOST void TLGridMemoryManager::allocateLeafLevelPairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBuffer, aNumPairs * sizeof(uint2),
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBuffer, leafLevelPairsBufferSize);
}


//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void TLGridMemoryManager::freeCellMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)cellsPtrDevice.ptr) );
}

HOST void TLGridMemoryManager::freeCellMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost((char*)cellsPtrHost.ptr) );
}

HOST void TLGridMemoryManager::freeLeafMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree(leavesDevice) );
}

HOST void TLGridMemoryManager::freeLeafMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost(leavesHost) );
}

HOST void TLGridMemoryManager::freePrimitiveIndicesBuffer()
{
    if(primitiveIndicesSize != 0u)
    {
        primitiveIndicesSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(primitiveIndices) );
    }
}

HOST void TLGridMemoryManager::freeRefCountsBuffer()
{
    if(refCountsBufferSize != 0u)
    {
        refCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(refCountsBufferHost, refCountsBuffer);
    }
}

HOST void TLGridMemoryManager::freeCellCountsBuffer()
{
    if(cellCountsBufferSize != 0u)
    {
        cellCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(cellCountsBufferHost, cellCountsBuffer);
    }
}

HOST void TLGridMemoryManager::freeTopLevelPairsBufferPair()
{
    if(topLevelPairsBufferSize != 0u)
    {
        topLevelPairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(topLevelPairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(topLevelPairsPingBuffer) );
    }
}

HOST void TLGridMemoryManager::freeLeafLevelPairsBufferPair()
{
    if(leafLevelPairsBufferSize != 0u)
    {
        topLevelPairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(leafLevelPairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(leafLevelPairsPingBuffer) );
    }
}

HOST void TLGridMemoryManager::cleanup()
{
    freeCellMemoryDevice();
    freeCellMemoryHost();
    freeLeafMemoryDevice();
    freeLeafMemoryHost();
    freePrimitiveIndicesBuffer();
    freeRefCountsBuffer();
    freeCellCountsBuffer();
    freeTopLevelPairsBufferPair();
    freeLeafLevelPairsBufferPair();

}
//////////////////////////////////////////////////////////////////////////
//debug related
//////////////////////////////////////////////////////////////////////////
HOST void TLGridMemoryManager::checkResolution()
{
    if (resX <= 0 || resY <= 0 || resZ <= 0)
    {
        cudastd::logger::out << "Invalid grid resolution!" 
            << " Setting grid resolution to 32 x 32 x 32\n";
        resX = resY = resZ = 32;
    }
}
