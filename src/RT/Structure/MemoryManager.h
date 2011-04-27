#ifdef _MSC_VER
#pragma once
#endif

#ifndef MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B
#define MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B

#include "CUDAStdAfx.h"


class MemoryManager
{
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
#else
        if (aOldSize < aSize)
        {
            MY_CUDA_SAFE_CALL( cudaFreeHost(*aOldHostPtr) );
            MY_CUDA_SAFE_CALL( cudaFree(*aOldDevicePtr) );
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaHostAlloc(aOldHostPtr,aSize, cudaHostAllocDefault) );
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldDevicePtr,aSize) );
        }

#endif
        *aDevicePtr = *aOldDevicePtr;
        *aHostPtr = *aOldHostPtr;

    }

    HOST static void freeMappedDeviceArray(void* aHostPtr, void* aDevicePtr)
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120           
        MY_CUDA_SAFE_CALL( cudaFreeHost(aHostPtr) );
#else
        MY_CUDA_SAFE_CALL( cudaFreeHost(aHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFree(aDevicePtr) );
#endif
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
        if (aOldSize < aSize)
        {
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr1) );
            MY_CUDA_SAFE_CALL( cudaFree(*aOldPtr2) );
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr1, aSize));
            MY_CUDA_SAFE_CALL( cudaMalloc(aOldPtr2, aSize));
        }

        *aPtr1 = *aOldPtr1;
        *aPtr2 = *aOldPtr2;
    }


};


#endif // MEMORYMANAGER_H_INCLUDED_2F89BBF8_5464_4327_8C50_C4D69372B62B
