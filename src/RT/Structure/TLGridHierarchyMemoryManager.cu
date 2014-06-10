/****************************************************************************/
/* Copyright (c) 2013, Javor Kalojanov
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
#include "RT/Structure/TLGridHierarchyMemoryManager.h"
#include "RT/Structure/MemoryManager.h"

//////////////////////////////////////////////////////////////////////////
//data transfer related
//////////////////////////////////////////////////////////////////////////

HOST void TLGridHierarchyMemoryManager::copyCellsDeviceToHost()
{
    cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
    cpyParamsDownloadPtr.srcPtr  = cellsPtrDevice;
    cpyParamsDownloadPtr.dstPtr  = cellsPtrHost;
    cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
    cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
}

HOST void TLGridHierarchyMemoryManager::copyCellsHostToDevice()
{
    cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
    cpyParamsUploadPtr.srcPtr  = cellsPtrHost;
    cpyParamsUploadPtr.dstPtr  = cellsPtrDevice;
    cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
    cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
}

HOST void TLGridHierarchyMemoryManager::copyInstancesHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(instancesDevice, instancesHost, instancesSize, cudaMemcpyHostToDevice));
}

HOST void TLGridHierarchyMemoryManager::copyInstancesDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(instancesHost, instancesDevice, instancesSize, cudaMemcpyDeviceToHost));
}

HOST void TLGridHierarchyMemoryManager::copyInstanceIndicesHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(instanceIndicesDevice, instanceIndicesHost, instanceIndicesSize, cudaMemcpyHostToDevice));
}

HOST void TLGridHierarchyMemoryManager::copyInstanceIndicesDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(instanceIndicesHost, instanceIndicesDevice, instanceIndicesSize, cudaMemcpyDeviceToHost));
}


HOST void TLGridHierarchyMemoryManager::copyGridsDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(gridsHost, gridsDevice, gridsSize, cudaMemcpyDeviceToHost));
    if(leavesHost != NULL)
    {
        char* basePtr = (char*)gridsHost[0].cells.ptr;
        size_t numGrids = gridsSize / sizeof(UniformGrid);    
        for(size_t gridId = 0; gridId < numGrids; ++gridId)
        {
            char* ptr = (char*)gridsHost[gridId].cells.ptr;
            gridsHost[gridId].cells.ptr = (char*)leavesHost + ((char*)ptr - (char*)basePtr);
        }
    }
    else
    {
        cudastd::logger::out << "Warning: Copied uniform grids form device to host, without correcting their cells-pointers\n";
    }
}

HOST void TLGridHierarchyMemoryManager::copyGridsHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(gridsDevice, gridsHost, gridsSize, cudaMemcpyHostToDevice));
}

HOST void TLGridHierarchyMemoryManager::copyLeavesHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(leavesDevice, leavesHost, leavesSize, cudaMemcpyHostToDevice));
}

HOST void TLGridHierarchyMemoryManager::copyLeavesDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(leavesHost, leavesDevice, leavesSize, cudaMemcpyDeviceToHost));
}

HOST void TLGridHierarchyMemoryManager::copyPrimitiveIndicesDeviceToHost()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(primitiveIndicesHost, primitiveIndices, primitiveIndicesSize, cudaMemcpyDeviceToHost));
}

HOST void TLGridHierarchyMemoryManager::copyPrimitiveIndicesHostToDevice()
{
    MY_CUDA_SAFE_CALL(cudaMemcpy(primitiveIndices, primitiveIndicesHost, primitiveIndicesSize, cudaMemcpyHostToDevice));
}


/////////////////////////////////////////////////////////////////////////
//memory allocation
//////////////////////////////////////////////////////////////////////////
HOST cudaPitchedPtr TLGridHierarchyMemoryManager::allocateHostCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ && cellsPtrHost.ptr != NULL)
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

HOST cudaPitchedPtr TLGridHierarchyMemoryManager::allocateDeviceCells()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ && cellsPtrDevice.ptr != NULL)
    {
        return cellsPtrDevice;
    }

    freeCellMemoryDevice();

    cudaExtent cellDataExtent = 
        make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);

    MY_CUDA_SAFE_CALL( cudaMalloc3D(&cellsPtrDevice, cellDataExtent) );

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;


    return cellsPtrDevice;
}

HOST void TLGridHierarchyMemoryManager::allocateInstanceIndices(const size_t aNumIndices)
{
    MemoryManager::allocateHostDeviceArrayPair((void**)&instanceIndicesDevice, (void**)&instanceIndicesHost,
        aNumIndices * sizeof(uint), (void**)&instanceIndicesDevice, (void**)&instanceIndicesHost, instanceIndicesSize);
}

HOST GeometryInstance* TLGridHierarchyMemoryManager::allocateDeviceInstances(const size_t aNumInstances)
{
    MemoryManager::allocateDeviceArray((void**)&instancesDevice, aNumInstances * sizeof(GeometryInstance),
        (void**)&instancesDevice, instancesSize);

    return instancesDevice;
}

HOST GeometryInstance* TLGridHierarchyMemoryManager::allocateHostInstances(const size_t aNumInstances)
{
    if(aNumInstances * sizeof(GeometryInstance) > instancesSize || instancesHost == NULL)
    {
        MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&instancesHost, aNumInstances * sizeof(GeometryInstance), cudaHostAllocDefault));
    }
    return instancesHost;
}

HOST UniformGrid* TLGridHierarchyMemoryManager::allocateGrids( const size_t aNumGrids )
{
    if(gridsSize < aNumGrids * sizeof(UniformGrid) || gridsHost == NULL)
    {
        MY_CUDA_SAFE_CALL( cudaFreeHost(gridsHost) );
        MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&gridsHost,
            aNumGrids * sizeof(UniformGrid), cudaHostAllocDefault));
    }
    MemoryManager::allocateDeviceArray((void**)&gridsDevice, aNumGrids * sizeof(UniformGrid),
        (void**)&gridsDevice, gridsSize);

    return gridsDevice;
}

HOST void TLGridHierarchyMemoryManager::setDeviceCellsToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(cellsPtrDevice.ptr, 0 ,
        cellsPtrDevice.pitch * resY * resZ ) );

    //does not work!
    //cudaExtent cellDataExtent = 
    //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
    //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
}

HOST TLGridHierarchyMemoryManager::t_Leaf* TLGridHierarchyMemoryManager::allocateHostLeaves(const size_t aNumLeaves)
{

    leavesSize = aNumLeaves * sizeof(t_Leaf);
    MY_CUDA_SAFE_CALL( cudaFreeHost(leavesHost) );
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&leavesHost,
        leavesSize, cudaHostAllocDefault));

    return leavesHost;
}

HOST void TLGridHierarchyMemoryManager::setDeviceLeavesToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(leavesDevice, 0 ,leavesSize) );

    //does not work!
    //cudaExtent cellDataExtent = 
    //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
    //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
}

HOST TLGridHierarchyMemoryManager::t_Leaf* TLGridHierarchyMemoryManager::allocateDeviceLeaves(const size_t aNumLeaves)
{
    MemoryManager::allocateDeviceArray((void**)&leavesDevice, aNumLeaves * sizeof(t_Leaf),
        (void**)&leavesDevice, leavesSize);

    return leavesDevice;
}

HOST uint* TLGridHierarchyMemoryManager::allocatePrimitiveIndicesBuffer(const size_t aNumIndices)
{
    MemoryManager::allocateDeviceArray((void**)&primitiveIndices, aNumIndices * sizeof(uint),
        (void**)&primitiveIndices, primitiveIndicesSize);

    return primitiveIndices;
}


HOST uint* TLGridHierarchyMemoryManager::allocatePrimitiveIndicesBufferHost( const size_t aNumPrimitives )
{
    primitiveIndicesSize = aNumPrimitives * sizeof(uint);
    MY_CUDA_SAFE_CALL( cudaFreeHost(primitiveIndicesHost) );
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&primitiveIndicesHost,
        primitiveIndicesSize, cudaHostAllocDefault));

    return primitiveIndicesHost;

}


HOST void TLGridHierarchyMemoryManager::allocateRefCountsBuffer(const size_t aNumSlots)
{
    MemoryManager::allocateMappedDeviceArray(
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, aNumSlots * sizeof(uint),
        (void**)&refCountsBuffer, (void**)&refCountsBufferHost, refCountsBufferSize);

    MY_CUDA_SAFE_CALL( cudaMemset(refCountsBuffer + aNumSlots - 1, 0, sizeof(uint)) );
}

HOST void TLGridHierarchyMemoryManager::allocateCellCountsBuffer(const size_t aNumCells)
{
    MemoryManager::allocateMappedDeviceArray(
        (void**)&cellCountsBuffer, (void**)&cellCountsBufferHost, aNumCells * sizeof(uint),
        (void**)&cellCountsBuffer, (void**)&cellCountsBufferHost, cellCountsBufferSize);

    MY_CUDA_SAFE_CALL( cudaMemset(cellCountsBuffer + aNumCells - 1, 0, sizeof(uint)) );
}


HOST void TLGridHierarchyMemoryManager::allocateTopLevelPairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBufferKeys, aNumPairs * sizeof(uint2),
        (void**)&topLevelPairsBuffer, (void**)&topLevelPairsPingBufferKeys, topLevelPairsBufferSize);
}

HOST void TLGridHierarchyMemoryManager::allocateLeafLevelPairsBufferPair(const size_t aNumPairs)
{
    MemoryManager::allocateDeviceArrayPair(
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBufferKeys, aNumPairs * sizeof(uint2),
        (void**)&leafLevelPairsBuffer, (void**)&leafLevelPairsPingBufferKeys, leafLevelPairsBufferSize);
}


//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void TLGridHierarchyMemoryManager::freeCellMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)cellsPtrDevice.ptr) );
    cellsPtrDevice.ptr = NULL;
}

HOST void TLGridHierarchyMemoryManager::freeCellMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost((char*)cellsPtrHost.ptr) );
    cellsPtrHost.ptr  = NULL;
}

HOST void TLGridHierarchyMemoryManager::freeInstanceIndices()
{
    MY_CUDA_SAFE_CALL( cudaFree(instanceIndicesDevice) );
    instanceIndicesDevice = NULL;
    MY_CUDA_SAFE_CALL( cudaFreeHost(instanceIndicesHost) );
    instanceIndicesHost = NULL;
    instanceIndicesSize = 0;
}

HOST void TLGridHierarchyMemoryManager::freeInstanceMemory()
{
    MY_CUDA_SAFE_CALL( cudaFree(instancesDevice) );
    instancesDevice = NULL;
    MY_CUDA_SAFE_CALL( cudaFreeHost(instancesHost) );
    instancesHost = NULL;
    instancesSize = 0;
}


HOST void TLGridHierarchyMemoryManager::freeGridMemory()
{
    MY_CUDA_SAFE_CALL( cudaFree(gridsDevice) );
    gridsDevice = NULL;
    MY_CUDA_SAFE_CALL( cudaFreeHost(gridsHost) );
    gridsHost = NULL;
    gridsSize = 0;
}

HOST void TLGridHierarchyMemoryManager::freeLeafMemoryDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree(leavesDevice) );
    leavesDevice = NULL;
    leavesSize = 0u;
}

HOST void TLGridHierarchyMemoryManager::freeLeafMemoryHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost(leavesHost) );
    leavesHost = NULL;
    leavesSize = 0u;
}

HOST void TLGridHierarchyMemoryManager::freePrimitiveIndicesBuffer()
{
    if(primitiveIndicesSize != 0u)
    {
        primitiveIndicesSize = 0u;
        MY_CUDA_SAFE_CALL( cudaFree(primitiveIndices) );
        primitiveIndices = NULL;
    }
}

HOST void TLGridHierarchyMemoryManager::freeRefCountsBuffer()
{
    if(refCountsBufferSize != 0u)
    {
        refCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(refCountsBufferHost, refCountsBuffer);
        refCountsBufferHost = NULL;
        refCountsBuffer = NULL;
    }
}

HOST void TLGridHierarchyMemoryManager::freeCellCountsBuffer()
{
    if(cellCountsBufferSize != 0u)
    {
        cellCountsBufferSize = 0u;
        MemoryManager::freeMappedDeviceArray(cellCountsBufferHost, cellCountsBuffer);
        cellCountsBufferHost = NULL;
        cellCountsBuffer = NULL;
    }
}

HOST void TLGridHierarchyMemoryManager::freeTopLevelPairsBufferPair()
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

HOST void TLGridHierarchyMemoryManager::freeLeafLevelPairsBufferPair()
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

HOST void TLGridHierarchyMemoryManager::cleanup()
{
    oldResX = 0;
    oldResY = 0;
    oldResZ = 0;
    freeCellMemoryDevice();
    freeCellMemoryHost();
    freeInstanceIndices();
    freeInstanceMemory();


    freeGridMemory();
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
HOST void TLGridHierarchyMemoryManager::checkResolution()
{
    if (resX <= 0 || resY <= 0 || resZ <= 0)
    {
        cudastd::logger::out << "Invalid grid resolution!" 
            << " Setting grid resolution to 32 x 32 x 32\n";
        resX = resY = resZ = 32;
    }
}

