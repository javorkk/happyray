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

#ifndef MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B
#define MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B

#include "CUDAStdAfx.h"


class MemoryManager
{
    static const size_t OFFSET_FOR_ALIGNMENT = 32*sizeof(size_t);
public:
    HOST static void allocateMappedDeviceArray(void** aDevicePtr, void** aHostPtr, size_t aSize,
        void** aOldDevicePtr, void** aOldHostPtr, size_t& aOldSize)
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120
        if (aOldSize < aSize)
        {
            MY_CUDA_SAFE_CALL( cudaFreeHost(*aOldHostPtr) );
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaHostAlloc(aOldHostPtr,aSize, cudaHostAllocMapped) );
        }

        MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer(aOldDevicePtr, *aOldHostPtr, 0));

        *aDevicePtr = *aOldDevicePtr;
        *aHostPtr = *aOldHostPtr;
#else
        allocateHostDeviceArrayPair(aDevicePtr, aHostPtr, aSize,
        aOldDevicePtr, aOldHostPtr, aOldSize);


#endif

    }

    HOST static void freeMappedDeviceArray(void* aHostPtr, void* aDevicePtr)
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120           
        MY_CUDA_SAFE_CALL( cudaFreeHost(aHostPtr) );
#else
       freeHostDeviceArrayPair(aHostPtr, aDevicePtr);
#endif
    }

    HOST static void allocateHostDeviceArrayPair(void** aDevicePtr, void** aHostPtr, size_t aSize,
        void** aOldDevicePtr, void** aOldHostPtr, size_t& aOldSize)
    {
        if (aOldSize < aSize)
        {
            MY_CUDA_SAFE_CALL( cudaFreeHost(*aOldHostPtr) );
            MY_CUDA_SAFE_CALL( cudaFree(*aOldDevicePtr) );
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaHostAlloc(aOldHostPtr,aSize, cudaHostAllocDefault) );
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldDevicePtr,aSize) );
        }

        *aDevicePtr = *aOldDevicePtr;
        *aHostPtr = *aOldHostPtr;
    }

    HOST static void freeHostDeviceArrayPair(void* aHostPtr, void* aDevicePtr)
    {

        MY_CUDA_SAFE_CALL( cudaFreeHost(aHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFree(aDevicePtr) );
    }

    HOST static void allocateDeviceArray(void** aPtr, size_t aSize,
        void** aOldPtr, size_t& aOldSize)
    {
        if (aOldSize < aSize)
        {
            //MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr) );
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr, aSize));
        }

        *aPtr = *aOldPtr;
    }

    HOST static void allocateDeviceArrayPair(void** aPtr1, void** aPtr2,
        size_t aSize, void** aOldPtr1, void** aOldPtr2, size_t& aOldSize)
    {
        if (aOldSize < aSize + OFFSET_FOR_ALIGNMENT)
        {
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr1) );
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr2) );
            aOldSize = aSize + OFFSET_FOR_ALIGNMENT;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr1, aSize));
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr2, aSize));
        }

        *aPtr1 = *aOldPtr1;
        *aPtr2 = *aOldPtr2;
    }

    HOST static void allocateDeviceArrayTriple(
        void** aPtr1, void** aPtr2, void** aPtr3,
        size_t aSize1, size_t aSize2, size_t aSize3,
        void** aOldPtr1, void** aOldPtr2, void** aOldPtr3,
        size_t& aOldSize1,size_t& aOldSize2,size_t& aOldSize3)
    {
        if (aOldSize1 < aSize1 + OFFSET_FOR_ALIGNMENT)
        {
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr1) );
            aOldSize1 = aSize1 + OFFSET_FOR_ALIGNMENT;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr1, aSize1));
        }
        if (aOldSize2 < aSize2 + OFFSET_FOR_ALIGNMENT)
        {
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr2) );
            aOldSize2 = aSize2 + OFFSET_FOR_ALIGNMENT;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr2, aSize2));
        }
        if (aOldSize3 < aSize3 + OFFSET_FOR_ALIGNMENT)
        {
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr3) );
            aOldSize3 = aSize3 + OFFSET_FOR_ALIGNMENT;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr3, aSize3));
        }

        *aPtr1 = *aOldPtr1;
        *aPtr2 = *aOldPtr2;
        *aPtr3 = *aOldPtr3;
    }


};


#endif // MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B
