#ifdef _MSC_VER
#pragma once
#endif

#ifndef UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB
#define UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/BBox.hpp"

#include "Textures.h"

namespace cudastd 
{
    struct UniformGrid
    {
        int res[3];
        float3 cellSize;
        float3 cellSizeRCP;
        BBox bounds;
        uint* primitives;

        DEVICE float3 getCellSize() const
        {
            return cellSize;
        }

        DEVICE float3 getCellSizeRCP() const
        {
            return cellSizeRCP;
        }
    };

    struct UniformGridMemoryManager
    {
        typedef uint2 Cell;

        int resX, resY, resZ;
        BBox bounds;
        Cell* cpuCells;
        Cell* gpuCells;
        //bookkeeping for deallocation
        char* gpuCellsPitchPtr;
        char* cpuCellsPitchPtr;
        cudaArray* cellArray;
        uint* gpuPrimitives;
        size_t numPrimitives;

        UniformGridMemoryManager()
            :resX(0), resY(0), resZ(0), bounds(BBox::empty()),
            cpuCells(NULL), gpuCells(NULL), gpuPrimitives(NULL),
            numPrimitives(0u)
        {}

        //////////////////////////////////////////////////////////////////////////
        //traversal related
        //////////////////////////////////////////////////////////////////////////
        uint getCellId(uint aIdX, uint aIdY, uint aIdZ) const
        {
            return aIdX + aIdY * resX + aIdZ * resX * resY;
        }

        const Cell& getCell(uint aId) const
        {
            return cpuCells[aId];
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
            return (bounds.max - bounds.min) / getResolution();
        }

        float3 getCellSizeRCP() const
        {
            return getResolution() / (bounds.max - bounds.min);
        }

        //////////////////////////////////////////////////////////////////////////
        //data transfer related
        //////////////////////////////////////////////////////////////////////////
        UniformGrid getParameters() const
        {
            UniformGrid retval;
            retval.bounds = bounds;
            retval.res[0] = resX;
            retval.res[1] = resY;
            retval.res[2] = resZ;
            retval.cellSize = getCellSize();
            retval.cellSizeRCP = getCellSizeRCP();

            return retval;
        }

        void copyCellsDeviceToHost(
            const cudaPitchedPtr& aHostCells,
            const cudaPitchedPtr& aDeviceCells)
        {
            cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
            cpyParamsDownloadPtr.srcPtr  = aDeviceCells;
            cpyParamsDownloadPtr.dstPtr  = aHostCells;
            cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
            cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

            MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
        }

        void copyCellsHostToDevice(
            const cudaPitchedPtr& aDeviceCells,
            const cudaPitchedPtr& aHostCells)
        {
            cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
            cpyParamsUploadPtr.srcPtr  = aHostCells;
            cpyParamsUploadPtr.dstPtr  = aDeviceCells;
            cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
            cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

            MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
        }

        void bindDeviceDataToTexture(const cudaPitchedPtr& aDeviceCells)
        {
            cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
            cudaExtent res = make_cudaExtent(resX, resY, resZ);
            MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

            cudaMemcpy3DParms cpyParams = { 0 };
            cpyParams.srcPtr    = aDeviceCells;
            cpyParams.dstArray  = cellArray;
            cpyParams.extent    = res;
            cpyParams.kind      = cudaMemcpyDeviceToDevice;


            MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

            MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
        }

        void reBindDeviceDataToTexture(const cudaPitchedPtr& aDeviceCells, cudaStream_t& aStream)
        {
            MY_CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
            MY_CUDA_SAFE_CALL( cudaUnbindTexture(texGridCells) );

            cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
            cudaExtent res = make_cudaExtent(resX, resY, resZ);
            MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

            cudaMemcpy3DParms cpyParams = { 0 };
            cpyParams.srcPtr    = aDeviceCells;
            cpyParams.dstArray  = cellArray;
            cpyParams.extent    = res;
            cpyParams.kind      = cudaMemcpyDeviceToDevice;


            MY_CUDA_SAFE_CALL( cudaMemcpy3DAsync(&cpyParams, aStream) );

            MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
        }

        void bindHostDataToTexture(const cudaPitchedPtr& aHostCells)
        {
            cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
            cudaExtent res = make_cudaExtent(resX, resY, resZ);
            MY_CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

            cudaMemcpy3DParms cpyParams = { 0 };
            cpyParams.srcPtr    = aHostCells;
            cpyParams.dstArray  = cellArray;
            cpyParams.extent    = res;
            cpyParams.kind      = cudaMemcpyHostToDevice;

            MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

            MY_CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
        }

        //////////////////////////////////////////////////////////////////////////
        //memory allocation
        //////////////////////////////////////////////////////////////////////////
        cudaPitchedPtr allocateHostCells()
        {
            checkResolution();

            MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
                resX * resY * resZ * sizeof(Cell)));

            cudaPitchedPtr pitchedPtrCPUCells = 
                make_cudaPitchedPtr(cpuCells, resX * sizeof(Cell), resX, resY);

            cpuCellsPitchPtr = (char*)pitchedPtrCPUCells.ptr;

            return pitchedPtrCPUCells;
        }

        cudaPitchedPtr allocateDeviceCells()
        {
            checkResolution();

            cudaPitchedPtr pitchedPtrGPUCells =
                make_cudaPitchedPtr(gpuCells, resX * sizeof(Cell), resX, resY);

            cudaExtent cellDataExtent = 
                make_cudaExtent(resX * sizeof(Cell), resY, resZ);

            MY_CUDA_SAFE_CALL( cudaMalloc3D(&pitchedPtrGPUCells, cellDataExtent) );

            gpuCellsPitchPtr = (char*)pitchedPtrGPUCells.ptr;

            return pitchedPtrGPUCells;
        }

        void setDeviceCellsToZero(const cudaPitchedPtr& aDeviceCells)
        {
            MY_CUDA_SAFE_CALL( cudaMemset(aDeviceCells.ptr, 0 ,
                aDeviceCells.pitch * resY * resZ ) );

            //does not work!
            //cudaExtent cellDataExtent = 
            //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
            //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
        }

        uint* allocsatePrimitiveIndicesBuffer(const size_t aNumPrimitives)
        {
            if(aNumPrimitives < numPrimitives)
            {
                return gpuPrimitives;
            }

            MY_CUDA_SAFE_CALL( cudaFree(gpuPrimitives) );
            MY_CUDA_SAFE_CALL( cudaMalloc((void**)&gpuPrimitives, aNumPrimitives * sizeof(uint)) );
            numPrimitives = aNumPrimitives;            
        }


        //////////////////////////////////////////////////////////////////////////
        //memory deallocation
        //////////////////////////////////////////////////////////////////////////
        void freeCellMemoryDevice()
        {
            MY_CUDA_SAFE_CALL( cudaFree(gpuCellsPitchPtr) );
        }

        void freeCellMemoryHost()
        {
            MY_CUDA_SAFE_CALL( cudaFreeHost(cpuCellsPitchPtr) );
        }

        void freePrimitiveIndicesBuffer()
        {
            numPrimitives = 0u;
            MY_CUDA_SAFE_CALL( cudaFree(gpuPrimitives) );
        }

        void cleanup()
        {
            MY_CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
        }
        //////////////////////////////////////////////////////////////////////////
        //debug related
        //////////////////////////////////////////////////////////////////////////
        void checkResolution()
        {
            if (resX <= 0 || resY <= 0 || resZ <= 0)
            {
                cudastd::logger::out << "Invalid grid resolution!" 
                    << " Setting grid resolution to 32 x 32 x 32\n";
                resX = resY = resZ = 32;
            }
        }
    };

}; //namespace cudastd

#endif // UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB
