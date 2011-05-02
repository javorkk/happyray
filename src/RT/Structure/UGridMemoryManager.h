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

#ifndef UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922
#define UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922

#include "CUDAStdAfx.h"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/UniformGrid.h"

class UGridMemoryManager
{
public:
    typedef uint2 Cell;

    int resX, resY, resZ;
    int oldResX, oldResY, oldResZ;
    BBox bounds;
    Cell* cpuCells;
    Cell* gpuCells;

    cudaPitchedPtr cellsPtrDevice;
    cudaPitchedPtr cellsPtrHost;
    cudaArray* cellArray;

    uint* primitiveIndices;
    size_t primitiveIndicesSize;

    //////////////////////////////////////////////////////////////////////////
    //construction buffers
    //////////////////////////////////////////////////////////////////////////
    uint* refCountsBuffer;
    uint* refCountsBufferHost;
    size_t refCountsBufferSize;

    uint* pairsBuffer;
    uint* pairsPingBuffer;
    size_t pairsBufferSize;


    UGridMemoryManager()
        :resX(0), resY(0), resZ(0), oldResX(0), oldResY(0), oldResZ(0), bounds(BBox::empty()),
        cpuCells(NULL), gpuCells(NULL), cellArray(NULL), primitiveIndices(NULL),
        primitiveIndicesSize(0u), refCountsBuffer(NULL), refCountsBufferHost(NULL),
        refCountsBufferSize(0u),pairsBuffer(NULL), pairsPingBuffer(NULL),
        pairsBufferSize(0u)
    {}

    //////////////////////////////////////////////////////////////////////////
    //construction related
    //////////////////////////////////////////////////////////////////////////
    Cell& getCell(uint aId)
    {
        return cpuCells[aId];
    }

    const float3 getResolution() const
    {
        float3 retval;
        retval.x = static_cast<float>(resX);
        retval.y = static_cast<float>(resY);
        retval.z = static_cast<float>(resZ);
        return retval;
    }

    float3 getCellSize() const
    {
        return bounds.diagonal() / getResolution();
    }

    float3 getCellSizeRCP() const
    {
        return getResolution() / bounds.diagonal();
    }

    //////////////////////////////////////////////////////////////////////////
    //data transfer related
    //////////////////////////////////////////////////////////////////////////
    UniformGrid getParameters() const
    {
        UniformGrid retval;
        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.cellSize = getCellSize();
        retval.cellSizeRCP = getCellSizeRCP();
        retval.cells = cellsPtrDevice;
        retval.primitives = primitiveIndices;
        return retval;
    }

    HOST void copyCellsDeviceToHost();

    HOST void copyCellsHostToDevice();

    HOST void bindDeviceDataToTexture()
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

    HOST void reBindDeviceDataToTexture(cudaStream_t& aStream)
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

    HOST void bindHostDataToTexture()
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
    HOST cudaPitchedPtr allocateHostCells();

    HOST cudaPitchedPtr allocateDeviceCells();

    HOST void setDeviceCellsToZero();

    HOST uint* allocatePrimitiveIndicesBuffer(const size_t aNumPrimitives);

    HOST void allocateRefCountsBuffer(const size_t aNumSlots);

    HOST void allocatePairsBufferPair(const size_t aNumPairs);

    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    HOST void freeCellMemoryDevice();

    HOST void freeCellMemoryHost();

    HOST void freePrimitiveIndicesBuffer();

    HOST void freeRefCountsBuffer();

    HOST void freePairsBufferPair();

    HOST void cleanup();
    //////////////////////////////////////////////////////////////////////////
    //debug related
    //////////////////////////////////////////////////////////////////////////
    HOST void checkResolution();
};

#endif // UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922
