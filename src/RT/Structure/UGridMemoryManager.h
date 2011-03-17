#ifdef _MSC_VER
#pragma once
#endif

#ifndef UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922
#define UGRIDMEMORYMANAGER_H_INCLUDED_CB606475_C661_4834_9F48_D6A86C7D2922

#include "CUDAStdAfx.h"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/UniformGrid.h"

struct UniformGridMemoryManager
{
    typedef uint2 Cell;

    int resX, resY, resZ;
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


    UniformGridMemoryManager()
        :resX(0), resY(0), resZ(0), bounds(BBox::empty()),
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

    HOST void bindDeviceDataToTexture();

    HOST void reBindDeviceDataToTexture(cudaStream_t& aStream);

    HOST void bindHostDataToTexture();

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
