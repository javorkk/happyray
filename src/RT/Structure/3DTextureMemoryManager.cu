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
#include "Core/Algebra.hpp"
#include "RT/Structure/3DTextureMemoryManager.h"

HOST void TextureMemoryManager::checkResolution()
{
    if (resX <= 0 || resY <= 0 || resZ <= 0)
    {
        cudastd::logger::out << "Invalid texture resolution!" 
            << " Setting texture resolution to 32 x 32 x 32\n";
        resX = resY = resZ = 32;
    }
}

//////////////////////////////////////////////////////////////////////////
//data transfer related
//////////////////////////////////////////////////////////////////////////

HOST void TextureMemoryManager::copyDataDeviceToHost()
{
    cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
    cpyParamsDownloadPtr.srcPtr  = texelPtrDevice;
    cpyParamsDownloadPtr.dstPtr  = texelPtrHost;
    cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(float3), resY, resZ);
    cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
}

HOST void TextureMemoryManager::copyDataHostToDevice()
{
    cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
    cpyParamsUploadPtr.srcPtr  = texelPtrHost;
    cpyParamsUploadPtr.dstPtr  = texelPtrDevice;
    cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(float3), resY, resZ);
    cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

    MY_CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
}

//////////////////////////////////////////////////////////////////////////
//memory allocation
//////////////////////////////////////////////////////////////////////////
HOST cudaPitchedPtr TextureMemoryManager::allocateDataHost()
{
    checkResolution();
    
    freeDataHost();

    MY_CUDA_SAFE_CALL( cudaMallocHost((void**)&texelsHost,
        resX * resY * resZ * sizeof(float3)));

    texelPtrHost = 
        make_cudaPitchedPtr(texelsHost, resX * sizeof(float3), resX, resY);

    return texelPtrHost;
}

HOST cudaPitchedPtr TextureMemoryManager::allocateDataDevice()
{
    checkResolution();

    if(oldResX == resX && oldResY == resY && oldResZ == resZ)
    {
        return texelPtrDevice;
    }

    freeDataDevice();

    //texelPtrDevice =
    //    make_cudaPitchedPtr(texelsDevice, resX * sizeof(float3), resX, resY);

    cudaExtent cellDataExtent = 
        make_cudaExtent(resX * sizeof(float3), resY, resZ);

    MY_CUDA_SAFE_CALL( cudaMalloc3D(&texelPtrDevice, cellDataExtent) );

    oldResX = resX;
    oldResY = resY;
    oldResZ = resZ;

    return texelPtrDevice;
}

HOST void TextureMemoryManager::setDeviceCellsToZero()
{
    MY_CUDA_SAFE_CALL( cudaMemset(texelPtrDevice.ptr, 0 , texelPtrDevice.pitch * resY * resZ ) );
}

//////////////////////////////////////////////////////////////////////////
//memory deallocation
//////////////////////////////////////////////////////////////////////////
HOST void TextureMemoryManager::freeDataDevice()
{
    MY_CUDA_SAFE_CALL( cudaFree((char*)texelPtrDevice.ptr) );
    texelPtrDevice.ptr = NULL;
}

HOST void TextureMemoryManager::freeDataHost()
{
    MY_CUDA_SAFE_CALL( cudaFreeHost((char*)texelPtrHost.ptr) );
    texelPtrHost.ptr = NULL;
}

HOST void TextureMemoryManager::cleanup()
{
    if(oldResX + oldResY + oldResZ != 0)
    {
        freeDataDevice();
        freeDataHost();

        oldResX = oldResY = oldResZ = 0;
        resX = resY = resZ = 1;
    }
}
