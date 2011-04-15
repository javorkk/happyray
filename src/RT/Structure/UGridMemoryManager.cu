#include "CUDAStdAfx.h"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Structure/MemoryManager.h"

#include "Core/Algebra.hpp"
#include "Textures.h"


//////////////////////////////////////////////////////////////////////////
//data transfer related
//////////////////////////////////////////////////////////////////////////

HOST void UGridMemoryManager::copyCellsDeviceToHost()
{
    cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
    cpyParamsDownloadPtr.srcPtr  = cellsPtrDevice;
    cpyParamsDownloadPtr.dstPtr  = cellsPtrHost;
    cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
    cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
}

HOST void UGridMemoryManager::copyCellsHostToDevice()
{
    cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
    cpyParamsUploadPtr.srcPtr  = cellsPtrHost;
    cpyParamsUploadPtr.dstPtr  = cellsPtrDevice;
    cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
    cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
}


//////////////////////////////////////////////////////////////////////////
//memory allocation
//////////////////////////////////////////////////////////////////////////
HOST cudaPitchedPtr UGridMemoryManager::allocateHostCells()
{
    checkResolution();

    MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
        resX * resY * resZ * sizeof(Cell)));

    cellsPtrHost = 
        make_cudaPitchedPtr(cpuCells, resX * sizeof(Cell), resX, resY);

    return cellsPtrHost;
}

HOST cudaPitchedPtr UGridMemoryManager::allocateDeviceCells()
{
    checkResolution();

    cellsPtrDevice =
        make_cudaPitchedPtr(gpuCells, resX * sizeof(Cell), resX, resY);

    cudaExtent cellDataExtent = 
        make_cudaExtent(resX * sizeof(Cell), resY, resZ);

    MY_CUDA_SAFE_CALL( cudaMalloc3D(&cellsPtrDevice, cellDataExtent) );

    return cellsPtrDevice;
}

HOST void UGridMemoryManager::setDeviceCellsToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(cellsPtrDevice.ptr, 0 ,
        cellsPtrDevice.pitch * resY * resZ ) );

    //does not work!
    //cudaExtent cellDataExtent = 
    //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
    //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
}

HOST uint* UGridMemoryManager::allocatePrimitiveIndicesBuffer(const size_t aNumIndices)
{
    MemoryManager::allocateDeviceArray((void**)&primitiveIndices, aNumIndices * sizeof(uint),
        (void**)&primitiveIndices, primitiveIndicesSize);
    
    return primitiveIndices;
}

HOST void UGridMemoryManager::allocateRefCountsBuffer(const size_t aNumSlots)
{
    MemoryManager::allocateMappedDeviceArray(
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, aNumSlots * sizeof(uint),
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, refCountsBufferSize);

    MY_CUDA_SAFE_CALL( cudaMemset(refCountsBuffer + aNumSlots - 1, 0, sizeof(uint)) );
}

HOST void UGridMemoryManager::allocatePairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&pairsBuffer, (void**)&pairsPingBuffer, aNumPairs * sizeof(uint2),
        (void**)&pairsBuffer, (void**)&pairsPingBuffer, pairsBufferSize);
}

//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void UGridMemoryManager::freeCellMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)cellsPtrDevice.ptr) );
}

HOST void UGridMemoryManager::freeCellMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost((char*)cellsPtrHost.ptr) );
}

HOST void UGridMemoryManager::freePrimitiveIndicesBuffer()
{
    if(primitiveIndicesSize != 0u)
    {
        primitiveIndicesSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(primitiveIndices) );
    }
}

HOST void UGridMemoryManager::freeRefCountsBuffer()
{
    if(refCountsBufferSize != 0u)
    {
        refCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(refCountsBufferHost, refCountsBuffer);
    }
}

HOST void UGridMemoryManager::freePairsBufferPair()
{
    if(pairsBufferSize != 0u)
    {
        pairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(pairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(pairsPingBuffer) );
    }
}


HOST void UGridMemoryManager::cleanup()
{
    if(cellArray != NULL)
        MY_CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
}
//////////////////////////////////////////////////////////////////////////
//debug related
//////////////////////////////////////////////////////////////////////////
HOST void UGridMemoryManager::checkResolution()
{
    if (resX <= 0 || resY <= 0 || resZ <= 0)
    {
        cudastd::logger::out << "Invalid grid resolution!" 
            << " Setting grid resolution to 32 x 32 x 32\n";
        resX = resY = resZ = 32;
    }
}