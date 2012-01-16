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

#ifndef TLGRIDMEMORYMANAGER_H_INCLUDED_79124A76_4D54_49D8_A49F_680A78886D0F
#define TLGRIDMEMORYMANAGER_H_INCLUDED_79124A76_4D54_49D8_A49F_680A78886D0F

#include "CUDAStdAfx.h"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/TwoLevelGrid.h"

class TLGridMemoryManager
{
public:
    typedef TwoLevelGrid::t_Cell    t_Cell;
    typedef uint2                   t_Leaf;

    int resX, resY, resZ;
    int oldResX, oldResY, oldResZ;
    BBox bounds;
    float topLevelDensity;
    float leafLevelDensity;

    cudaPitchedPtr  cellsPtrDevice;
    cudaPitchedPtr  cellsPtrHost;

    t_Leaf*         leavesHost;
    t_Leaf*         leavesDevice;
    size_t          leavesSize;

    uint* primitiveIndices;
    size_t primitiveIndicesSize;

    //////////////////////////////////////////////////////////////////////////
    //construction buffers
    //////////////////////////////////////////////////////////////////////////
    uint* refCountsBuffer;
    uint* refCountsBufferHost;
    size_t refCountsBufferSize;
    uint* cellCountsBuffer;
    uint* cellCountsBufferHost;
    size_t cellCountsBufferSize;
    uint* topLevelPairsBuffer;
    uint* topLevelPairsPingBufferKeys;
    uint* topLevelPairsPingBufferValues;
    size_t topLevelPairsBufferSize;
    size_t topLevelPairsPingBufferKeysSize;
    size_t topLevelPairsPingBufferValuesSize;
    uint* leafLevelPairsBuffer;
    uint* leafLevelPairsPingBufferKeys;
    uint* leafLevelPairsPingBufferValues;
    size_t leafLevelPairsBufferSize;
    size_t leafLevelPairsPingBufferKeysSize;
    size_t leafLevelPairsPingBufferValuesSize;


    TLGridMemoryManager()
        :resX(0), resY(0), resZ(0), oldResX(0), oldResY(0), oldResZ(0), bounds(BBox::empty()),leavesHost(NULL),
        leavesDevice(NULL), leavesSize(0), primitiveIndices(NULL),
        primitiveIndicesSize(0u), refCountsBuffer(NULL), refCountsBufferHost(NULL),
        refCountsBufferSize(0u),cellCountsBuffer(NULL),cellCountsBufferHost(NULL),
        cellCountsBufferSize(0u),
        topLevelPairsBuffer(NULL), topLevelPairsPingBufferKeys(NULL), topLevelPairsPingBufferValues(NULL),
        topLevelPairsBufferSize(0u), topLevelPairsPingBufferKeysSize(0u), topLevelPairsPingBufferValuesSize(0u),
        leafLevelPairsBuffer(NULL), leafLevelPairsPingBufferKeys(NULL), leafLevelPairsPingBufferValues(NULL),
        leafLevelPairsBufferSize(0u), leafLevelPairsPingBufferKeysSize(0u), leafLevelPairsPingBufferValuesSize(0u)
    {}

    //////////////////////////////////////////////////////////////////////////
    //construction related
    //////////////////////////////////////////////////////////////////////////
    size_t getMemorySize() const
    {
        return primitiveIndicesSize + refCountsBufferSize + cellCountsBufferSize +
            2u * topLevelPairsBufferSize + 2u * leafLevelPairsBufferSize;
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
    TwoLevelGrid getParameters() const
    {
        TwoLevelGrid retval;
        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.cellSize = getCellSize();
        retval.cellSizeRCP = getCellSizeRCP();
        retval.cells = cellsPtrDevice;
        retval.leaves = leavesDevice;
        retval.primitives = primitiveIndices;
        return retval;
    }

    HOST void copyCellsDeviceToHost();

    HOST void copyCellsHostToDevice();

    HOST void copyLeavesDeviceToHost();

    HOST void copyLeavesHostToDevice();

    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    HOST cudaPitchedPtr allocateHostCells();

    HOST cudaPitchedPtr allocateDeviceCells();

    HOST void setDeviceCellsToZero();

    HOST t_Leaf* allocateHostLeaves(const size_t aNumLeaves);

    HOST t_Leaf* allocateDeviceLeaves(const size_t aNumLeaves);

    HOST void setDeviceLeavesToZero();


    HOST uint* allocatePrimitiveIndicesBuffer(const size_t aNumPrimitives);

    HOST void allocateRefCountsBuffer(const size_t aNumSlots);
    HOST void allocateCellCountsBuffer(const size_t aNumCells);
    HOST void allocateTopLevelPairsBufferPair(const size_t aNumPairs);
    HOST void allocateLeafLevelPairsBufferPair(const size_t aNumPairs);

    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    HOST void freeCellMemoryDevice();

    HOST void freeCellMemoryHost();

    HOST void freeLeafMemoryDevice();

    HOST void freeLeafMemoryHost();

    HOST void freePrimitiveIndicesBuffer();

    HOST void freeRefCountsBuffer();
    HOST void freeCellCountsBuffer();
    HOST void freeTopLevelPairsBufferPair();
    HOST void freeLeafLevelPairsBufferPair();

    HOST void cleanup();
    //////////////////////////////////////////////////////////////////////////
    //debug related
    //////////////////////////////////////////////////////////////////////////
    HOST void checkResolution();



};

#endif // TLGRIDMEMORYMANAGER_H_INCLUDED_79124A76_4D54_49D8_A49F_680A78886D0F
