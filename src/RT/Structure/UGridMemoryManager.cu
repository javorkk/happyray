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


HOST void UGridMemoryManager::copyPrimitiveIndicesDeviceToHost()
{
    MY_CUDA_SAFE_CALL( cudaMemcpy(primitiveIndicesHost, primitiveIndices, primitiveIndicesSize, cudaMemcpyDeviceToHost) );
}

HOST void UGridMemoryManager::copyPrimitiveIndicesHostToDevice()
{
    MY_CUDA_SAFE_CALL( cudaMemcpy(primitiveIndices, primitiveIndicesHost, primitiveIndicesSize, cudaMemcpyHostToDevice) );
}

HOST void UGridMemoryManager::bindDeviceDataToTexture()
{
    cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
    cudaExtent res = make_cudaExtent(resX, resY, resZ);
    MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

    cudaMemcpy3DParms cpyParams = { 0 };
    cpyParams.srcPtr    = cellsPtrDevice;
    cpyParams.dstArray  = cellArray;
    cpyParams.extent    = res;
    cpyParams.kind      = cudaMemcpyDeviceToDevice;


    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

    MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
}

HOST void UGridMemoryManager::reBindDeviceDataToTexture( cudaStream_t& aStream )
{
    MY_CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
    MY_CUDA_SAFE_CALL( cudaUnbindTexture(texGridCells) );

    cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
    cudaExtent res = make_cudaExtent(resX, resY, resZ);
    MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

    cudaMemcpy3DParms cpyParams = { 0 };
    cpyParams.srcPtr    = cellsPtrDevice;
    cpyParams.dstArray  = cellArray;
    cpyParams.extent    = res;
    cpyParams.kind      = cudaMemcpyDeviceToDevice;


    MY_CUDA_SAFE_CALL( cudaMemcpy3DAsync(&cpyParams, aStream) );

    MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
}

HOST void UGridMemoryManager::bindHostDataToTexture()
{
    cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
    cudaExtent res = make_cudaExtent(resX, resY, resZ);
    MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

    cudaMemcpy3DParms cpyParams = { 0 };
    cpyParams.srcPtr    = cellsPtrHost;
    cpyParams.dstArray  = cellArray;
    cpyParams.extent    = res;
    cpyParams.kind      = cudaMemcpyHostToDevice;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

    MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
}
//////////////////////////////////////////////////////////////////////////
//memory allocation
//////////////////////////////////////////////////////////////////////////
HOST cudaPitchedPtr UGridMemoryManager::allocateHostCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ && cellsPtrHost.ptr != NULL)
    {
        return cellsPtrHost;
    }

    freeCellMemoryHost();


    MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
        resX * resY * resZ * sizeof(Cell)));

    cellsPtrHost = 
        make_cudaPitchedPtr(cpuCells, resX * sizeof(Cell), resX * sizeof(Cell), resY);

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;

    return cellsPtrHost;
}

HOST cudaPitchedPtr UGridMemoryManager::allocateDeviceCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ && cellsPtrDevice.ptr != NULL)
    {
        return cellsPtrDevice;
    }

    freeCellMemoryDevice();

    cudaExtent cellDataExtent = 
        make_cudaExtent(resX * sizeof(Cell), resY, resZ);

    MY_CUDA_SAFE_CALL( cudaMalloc3D(&cellsPtrDevice, cellDataExtent) );

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;

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
    MemoryManager::allocateHostDeviceArrayPair(
        (void**)&primitiveIndices, (void**)&primitiveIndicesHost, aNumIndices * sizeof(uint),
        (void**)&primitiveIndices, (void**)&primitiveIndicesHost, primitiveIndicesSize);
    
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
//#if HAPPYRAY__CUDA_ARCH__ < 200
    //use double buffers and store pairs (chag::pp radix sort)
    MemoryManager::allocateDeviceArrayPair(
        (void**)&pairsBuffer, (void**)&pairsPingBufferKeys, aNumPairs * sizeof(uint2),
        (void**)&pairsBuffer, (void**)&pairsPingBufferKeys, pairsBufferSize);
//#else
//    //allocate extra buffers to split keys and values (thrust radix sort)
//    MemoryManager::allocateDeviceArrayTriple(
//        (void**)&pairsBuffer, (void**)&pairsPingBufferKeys, (void**)&pairsPingBufferValues,
//        aNumPairs * sizeof(uint2), aNumPairs * sizeof(uint), aNumPairs * sizeof(uint),
//        (void**)&pairsBuffer, (void**)&pairsPingBufferKeys, (void**)&pairsPingBufferValues,
//        pairsBufferSize, pairsPingBufferKeysSize, pairsPingBufferValuesSize );
//
//#endif
}

//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void UGridMemoryManager::freeCellMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)cellsPtrDevice.ptr) );
    cellsPtrDevice.ptr = NULL;
}

HOST void UGridMemoryManager::freeCellMemoryHost()
{
    if(cellsPtrHost.ptr != NULL)
    {
        MY_CUDA_SAFE_CALL( cudaFreeHost((char*)cellsPtrHost.ptr) );
        cellsPtrHost.ptr = NULL;
    }
}

HOST void UGridMemoryManager::freePrimitiveIndicesBuffer()
{
    if(primitiveIndicesSize != 0u)
    {
        primitiveIndicesSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(primitiveIndices) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(primitiveIndicesHost) );
        primitiveIndices = NULL;
        primitiveIndicesHost = NULL;
    }
}

HOST void UGridMemoryManager::freeRefCountsBuffer()
{
    if(refCountsBufferSize != 0u)
    {
        refCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(refCountsBufferHost, refCountsBuffer);
        refCountsBufferHost = NULL;
        refCountsBuffer = NULL;
    }
}

HOST void UGridMemoryManager::freePairsBufferPair()
{
    if(pairsBufferSize != 0u)
    {
        pairsBufferSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(pairsBuffer) );
        MY_CUDA_SAFE_CALL( cudaFree(pairsPingBufferKeys) );
        pairsBuffer = NULL;
        pairsPingBufferKeys = NULL;

    }
}


HOST void UGridMemoryManager::cleanup()
{
    oldResX = 0;
    oldResY = 0;
    oldResZ = 0;
    freeCellMemoryDevice();

    if(cellArray != NULL)
        MY_CUDA_SAFE_CALL( cudaFreeArray(cellArray) );

    freeCellMemoryDevice();
    freeCellMemoryHost();
    freePrimitiveIndicesBuffer();
    freeRefCountsBuffer();
    freePairsBufferPair();
    freeKeyValueBuffers();
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

HOST void UGridMemoryManager::allocateKeyValueBuffers( const size_t aNumKeys )
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&pairsPingBufferKeys, (void**)&pairsPingBufferValues, aNumKeys * sizeof(uint),
        (void**)&pairsPingBufferKeys, (void**)&pairsPingBufferValues, pairsPingBufferKeysSize);
    pairsPingBufferValuesSize = pairsPingBufferKeysSize;
}

HOST void UGridMemoryManager::freeKeyValueBuffers()
{
    if(pairsPingBufferKeysSize != 0u)
    {
        pairsPingBufferKeysSize = 0u;
        pairsPingBufferValuesSize =  0u;
        MY_CUDA_SAFE_CALL( cudaFree(pairsPingBufferKeys) );
        MY_CUDA_SAFE_CALL( cudaFree(pairsPingBufferValues) );
        pairsPingBufferKeys = NULL;
        pairsPingBufferValues = NULL;
    }
}


