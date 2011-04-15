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
    uint* topLevelPairsPingBuffer;
    size_t topLevelPairsBufferSize;
    uint* leafLevelPairsBuffer;
    uint* leafLevelPairsPingBuffer;
    size_t leafLevelPairsBufferSize;


    TLGridMemoryManager()
        :resX(0), resY(0), resZ(0), bounds(BBox::empty()),leavesHost(NULL),
        leavesDevice(NULL), leavesSize(0), primitiveIndices(NULL),
        primitiveIndicesSize(0u), refCountsBuffer(NULL), refCountsBufferHost(NULL),
        refCountsBufferSize(0u),cellCountsBuffer(NULL),cellCountsBufferHost(NULL),
        cellCountsBufferSize(0u), topLevelPairsBuffer(NULL), topLevelPairsPingBuffer(NULL),
        topLevelPairsBufferSize(0u), leafLevelPairsBuffer(NULL),
        leafLevelPairsPingBuffer(NULL), leafLevelPairsBufferSize(0u)
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
