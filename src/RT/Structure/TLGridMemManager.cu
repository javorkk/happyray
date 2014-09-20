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

HOST void TLGridMemoryManager::copyLeavesHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(leavesDevice, leavesHost, leavesSize, cudaMemcpyHostToDevice));
}

HOST void TLGridMemoryManager::copyLeavesDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(leavesHost, leavesDevice, leavesSize, cudaMemcpyDeviceToHost));
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
        make_cudaPitchedPtr(cpuCells, resX * sizeof(t_Cell), resX * sizeof(t_Cell), resY);

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

    //t_Cell* gpuCells = NULL;

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
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBufferKeys, aNumPairs * sizeof(uint2),
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBufferKeys, topLevelPairsBufferSize);
}

HOST void TLGridMemoryManager::allocateLeafLevelPairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBufferKeys, aNumPairs * sizeof(uint2),
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBufferKeys, leafLevelPairsBufferSize);
}


//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void TLGridMemoryManager::freeCellMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)cellsPtrDevice.ptr) );
    cellsPtrDevice.ptr = NULL;
}

HOST void TLGridMemoryManager::freeCellMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost((char*)cellsPtrHost.ptr) );
    cellsPtrHost.ptr  = NULL;
}

HOST void TLGridMemoryManager::freeLeafMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree(leavesDevice) );
    leavesDevice = NULL;
    leavesSize = 0u;
}

HOST void TLGridMemoryManager::freeLeafMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost(leavesHost) );
    leavesHost = NULL;
    leavesSize = 0u;
}

HOST void TLGridMemoryManager::freePrimitiveIndicesBuffer()
{
    if(primitiveIndicesSize != 0u)
    {
        primitiveIndicesSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(primitiveIndices) );
        primitiveIndices = NULL;
    }
}

HOST void TLGridMemoryManager::freeRefCountsBuffer()
{
    if(refCountsBufferSize != 0u)
    {
        refCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(refCountsBufferHost, refCountsBuffer);
        refCountsBufferHost = NULL;
        refCountsBuffer = NULL;
    }
}

HOST void TLGridMemoryManager::freeCellCountsBuffer()
{
    if(cellCountsBufferSize != 0u)
    {
        cellCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(cellCountsBufferHost, cellCountsBuffer);
        cellCountsBufferHost = NULL;
        cellCountsBuffer = NULL;
    }
}

HOST void TLGridMemoryManager::freeTopLevelPairsBufferPair()
{
    if(topLevelPairsBufferSize != 0u)
    {
        topLevelPairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(topLevelPairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(topLevelPairsPingBufferKeys) );
        topLevelPairsBuffer = NULL;
        topLevelPairsPingBufferKeys= NULL;
    }
}

HOST void TLGridMemoryManager::freeLeafLevelPairsBufferPair()
{
    if(leafLevelPairsBufferSize != 0u)
    {
        leafLevelPairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(leafLevelPairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(leafLevelPairsPingBufferKeys) );
        leafLevelPairsBuffer = NULL;
        leafLevelPairsPingBufferKeys = NULL;
    }
}

HOST void TLGridMemoryManager::cleanup()
{
    oldResX = 0;
    oldResY = 0;
    oldResZ = 0;
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
