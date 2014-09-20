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
    uint* primitiveIndicesHost;

    size_t primitiveIndicesSize;

    //////////////////////////////////////////////////////////////////////////
    //construction buffers
    //////////////////////////////////////////////////////////////////////////
    uint* refCountsBuffer;
    uint* refCountsBufferHost;
    size_t refCountsBufferSize;

    uint* pairsBuffer;
    uint* pairsPingBufferKeys;
    uint* pairsPingBufferValues;
    size_t pairsBufferSize;
    size_t pairsPingBufferKeysSize;
    size_t pairsPingBufferValuesSize;


    UGridMemoryManager()
        :resX(0), resY(0), resZ(0), oldResX(0), oldResY(0), oldResZ(0), bounds(BBox::empty()),
        cpuCells(NULL), gpuCells(NULL),
        cellArray(NULL), primitiveIndices(NULL),
        primitiveIndicesHost(NULL),primitiveIndicesSize(0u), refCountsBuffer(NULL), refCountsBufferHost(NULL),
        refCountsBufferSize(0u),pairsBuffer(NULL), pairsPingBufferKeys(NULL),
        pairsPingBufferValues(NULL), pairsBufferSize(0u), pairsPingBufferKeysSize(0u),
        pairsPingBufferValuesSize(0u)

    {
         cellsPtrDevice.ptr = NULL;
         cellsPtrHost.ptr = NULL;
    }

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
    HOST UniformGrid getParameters() const
    {
        UniformGrid retval;
        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.setCellSize(getCellSize());
        retval.setCellSizeRCP(getCellSizeRCP());
        retval.cells = cellsPtrDevice;
        retval.primitives = primitiveIndices;
        //retval.numPrimitiveReferences = primitiveIndicesSize / sizeof(uint);
        return retval;
    }

    HOST UniformGrid getParametersHost()
    {
        UniformGrid retval;
        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.setCellSize(getCellSize());
        retval.setCellSizeRCP(getCellSizeRCP());
        allocateHostCells();
        copyCellsDeviceToHost();
        copyPrimitiveIndicesDeviceToHost();
        retval.cells = cellsPtrHost;
        retval.primitives = primitiveIndicesHost;
        //retval.numPrimitiveReferences = primitiveIndicesSize / sizeof(uint);
        return retval;
    }

    HOST void copyCellsDeviceToHost();

    HOST void copyCellsHostToDevice();

    HOST void bindDeviceDataToTexture();

    HOST void reBindDeviceDataToTexture(cudaStream_t& aStream);

    HOST void bindHostDataToTexture();

    HOST void copyPrimitiveIndicesDeviceToHost();
    HOST void copyPrimitiveIndicesHostToDevice();

    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    HOST cudaPitchedPtr allocateHostCells();

    HOST cudaPitchedPtr allocateDeviceCells();

    HOST void setDeviceCellsToZero();

    HOST uint* allocatePrimitiveIndicesBuffer(const size_t aNumPrimitives);

    HOST void allocateRefCountsBuffer(const size_t aNumSlots);

    HOST void allocatePairsBufferPair(const size_t aNumPairs);

    HOST void allocateKeyValueBuffers(const size_t aNumKeys);


    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    HOST void freeCellMemoryDevice();

    HOST void freeCellMemoryHost();

    HOST void freePrimitiveIndicesBuffer();

    HOST void freeRefCountsBuffer();

    HOST void freePairsBufferPair();

    HOST void freeKeyValueBuffers();

    HOST void cleanup();
    //////////////////////////////////////////////////////////////////////////
    //debug related
    //////////////////////////////////////////////////////////////////////////
    HOST void checkResolution();
};

#endif // UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922
