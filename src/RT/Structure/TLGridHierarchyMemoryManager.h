#ifdef _MSC_VER
#pragma once
#endif

#ifndef TLGRIDHIERARCHYMEMORYMANAGER_H_INCLUDED_4A2B08AD_A911_47AC_93E1_CE0E7B7AFB76
#define TLGRIDHIERARCHYMEMORYMANAGER_H_INCLUDED_4A2B08AD_A911_47AC_93E1_CE0E7B7AFB76

#include "CUDAStdAfx.h"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/TwoLevelGridHierarchy.h"

class TLGridHierarchyMemoryManager
{
public:
    typedef uint2                   t_Cell;
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

    uint*           instanceIndicesDevice;
    uint*           instanceIndicesHost;
    size_t          instanceIndicesSize;

    GeometryInstance* instancesDevice;
    GeometryInstance* instancesHost;
    size_t            instancesSize;
    
    UniformGrid*    gridsDevice;
    UniformGrid*    gridsHost;
    size_t          gridsSize;

    uint* primitiveIndices;
    uint* primitiveIndicesHost;
    size_t primitiveIndicesSize;

    uint* paramPtrHost;
    uint* paramPtrDevice;
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


    TLGridHierarchyMemoryManager()
        :resX(0), resY(0), resZ(0), oldResX(0), oldResY(0), oldResZ(0), bounds(BBox::empty()),
        leavesHost(NULL), leavesDevice(NULL), leavesSize(0), primitiveIndices(NULL), primitiveIndicesHost(NULL), primitiveIndicesSize(0u),
        instanceIndicesDevice(NULL), instanceIndicesHost(NULL), instanceIndicesSize(0),
        instancesDevice(NULL), instancesHost(NULL), instancesSize(0),
        gridsDevice(NULL), gridsHost(NULL), gridsSize(0),
        paramPtrHost(NULL),paramPtrDevice(NULL),
        refCountsBuffer(NULL), refCountsBufferHost(NULL),
        refCountsBufferSize(0u),cellCountsBuffer(NULL),cellCountsBufferHost(NULL),
        cellCountsBufferSize(0u),
        topLevelPairsBuffer(NULL), topLevelPairsPingBufferKeys(NULL), topLevelPairsPingBufferValues(NULL),
        topLevelPairsBufferSize(0u), topLevelPairsPingBufferKeysSize(0u), topLevelPairsPingBufferValuesSize(0u),
        leafLevelPairsBuffer(NULL), leafLevelPairsPingBufferKeys(NULL), leafLevelPairsPingBufferValues(NULL),
        leafLevelPairsBufferSize(0u), leafLevelPairsPingBufferKeysSize(0u), leafLevelPairsPingBufferValuesSize(0u)
    {
        cellsPtrDevice.ptr = NULL;
        cellsPtrHost.ptr = NULL;
    }

    //////////////////////////////////////////////////////////////////////////
    //construction related
    //////////////////////////////////////////////////////////////////////////
    HOST size_t getMemorySize() const
    {
        return primitiveIndicesSize + refCountsBufferSize + cellCountsBufferSize +
            2u * topLevelPairsBufferSize + 2u * leafLevelPairsBufferSize;
    }
    HOST const float3 getResolution() const
    {
        float3 retval;
        retval.x = static_cast<float>(resX);
        retval.y = static_cast<float>(resY);
        retval.z = static_cast<float>(resZ);
        return retval;
    }

    HOST float3 getCellSize() const
    {
        return bounds.diagonal() / getResolution();
    }

    HOST float3 getCellSizeRCP() const
    {
        return getResolution() / bounds.diagonal();
    }

    //////////////////////////////////////////////////////////////////////////
    //data transfer related
    //////////////////////////////////////////////////////////////////////////
    HOST TwoLevelGridHierarchy getParameters()
    {
        TwoLevelGridHierarchy retval;
        retval.allocatePtrs();

        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.setCellSize(getCellSize());
        retval.setCellSizeRCP(getCellSizeRCP());
        retval.cells = cellsPtrDevice;
        retval.setInstanceIndices(instanceIndicesDevice);
        retval.setInstances(instancesDevice);
        retval.setGrids(gridsDevice);
        retval.setLeaves(leavesDevice);
        retval.primitives = primitiveIndices;
        retval.setNumInstances((uint)(instancesSize / sizeof(GeometryInstance)));
        //retval.numPrimitiveReferences = primitiveIndicesSize / sizeof(uint);
        return retval;
    }

    HOST TwoLevelGridHierarchy getParametersHost()
    {
        TwoLevelGridHierarchy retval;
        retval.allocatePtrs();

        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.setCellSize(getCellSize());
        retval.setCellSizeRCP(getCellSizeRCP());
        retval.cells = cellsPtrHost;
        retval.setInstanceIndices(instanceIndicesHost);
        retval.setInstances(instancesHost);
        retval.setGrids(gridsHost);
        retval.setLeaves(leavesHost);
        retval.primitives = primitiveIndicesHost;
        retval.setNumInstances((uint)(instancesSize / sizeof(GeometryInstance)));
        //retval.numPrimitiveReferences = primitiveIndicesSize / sizeof(uint);
        return retval;
    }


    HOST void copyCellsDeviceToHost();

    HOST void copyCellsHostToDevice();

    HOST void copyInstancesHostToDevice();
    HOST void copyInstancesDeviceToHost();
    HOST void copyInstanceIndicesHostToDevice();
    HOST void copyInstanceIndicesDeviceToHost();

    HOST void copyGridsDeviceToHost();
    HOST void copyGridsHostToDevice();

    HOST void copyLeavesDeviceToHost();
    HOST void copyLeavesHostToDevice();

    HOST void copyPrimitiveIndicesDeviceToHost();
    HOST void copyPrimitiveIndicesHostToDevice();


    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    HOST cudaPitchedPtr allocateHostCells();
    HOST cudaPitchedPtr allocateDeviceCells();
    HOST void setDeviceCellsToZero();

    HOST void allocateInstanceIndices(const size_t aNumIndices);
    HOST GeometryInstance* allocateDeviceInstances(const size_t aNumInstances);
    HOST GeometryInstance* allocateHostInstances(const size_t aNumInstances);
    HOST UniformGrid* allocateGrids(const size_t aNumGrids);

    HOST t_Leaf* allocateHostLeaves(const size_t aNumLeaves);
    HOST t_Leaf* allocateDeviceLeaves(const size_t aNumLeaves);
    HOST void setDeviceLeavesToZero();

    HOST uint* allocatePrimitiveIndicesBuffer(const size_t aNumPrimitives);
    HOST uint* allocatePrimitiveIndicesBufferHost(const size_t aNumPrimitives);

    HOST void allocateRefCountsBuffer(const size_t aNumSlots);
    HOST void allocateCellCountsBuffer(const size_t aNumCells);
    HOST void allocateTopLevelPairsBufferPair(const size_t aNumPairs);
    HOST void allocateTopLevelKeyValueBuffers(const size_t aNumKeys);

    HOST void allocateLeafLevelPairsBufferPair(const size_t aNumPairs);
    HOST void allocateLeafLevelKeyValueBuffers(const size_t aNumKeys);

    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    HOST void freeCellMemoryDevice();
    HOST void freeCellMemoryHost();
    HOST void freeInstanceIndices();
    HOST void freeInstanceMemory();

    HOST void freeGridMemory();

    HOST void freeLeafMemoryDevice();
    HOST void freeLeafMemoryHost();

    HOST void freePrimitiveIndicesBuffer();

    HOST void freeRefCountsBuffer();
    HOST void freeCellCountsBuffer();
    HOST void freeTopLevelPairsBufferPair();
    HOST void freeTopLevelKeyValueBuffers();
    HOST void freeLeafLevelPairsBufferPair();
    HOST void freeeafLevelKeyValueBuffers();

    HOST void cleanup();
    //////////////////////////////////////////////////////////////////////////
    //debug related
    //////////////////////////////////////////////////////////////////////////
    HOST void checkResolution();

};

#endif // TLGRIDHIERARCHYMEMORYMANAGER_H_INCLUDED_4A2B08AD_A911_47AC_93E1_CE0E7B7AFB76
