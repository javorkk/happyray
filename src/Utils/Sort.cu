/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
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

#include "Sort.h"

#if HAPPYRAY__CUDA_ARCH__ < 200
#   define USE_CHAG_PP_SORT
#endif

#ifdef USE_CHAG_PP_SORT
#   include "chag/pp/sort.cuh"
#else
#   include <thrust/system/cuda/detail/detail/b40c/radixsort_api.h>
#   include <thrust/detail/util/align.h>

/**
 * Kernels to convert between arrays of uint2 and uint arrays 
 */
GLOBAL void PairsToSingles(
    uint2*  aPairs,
    uint*   oKeys,
    uint*   oValues,
    uint    aNumElements)
    {
        for(uint myIndex = globalThreadId1D(); myIndex < aNumElements;
            myIndex += numThreads())
        {
            uint2 input = aPairs[myIndex];
            oKeys[myIndex] = input.x;
            oValues[myIndex] = input.y;
        }

    }
GLOBAL void SinglesToPairs(
    uint2*  oPairs,
    uint*   aKeys,
    uint*   aValues,
    uint    aNumElements)
{
    for(uint myIndex = globalThreadId1D(); myIndex < aNumElements;
        myIndex += numThreads())
    {
        uint2 input;
        input.x = aKeys[myIndex];
        input.y = aValues[myIndex];
        oPairs[myIndex] = input;
    }

}

/**
 * Custom sorting storage-management structure for device vectors
 */
struct PersistentRadixSortStorage : public thrust::system::cuda::detail::detail::b40c_thrust::RadixSortStorage<uint, uint>
{
    PersistentRadixSortStorage(uint* keys = NULL, uint* values = NULL): RadixSortStorage<uint,uint>(keys, values)
    {}

    // DO NOT Clean up non-results storage !!!
	cudaError_t CleanupTempStorage() 
	{
        //if (d_alt_keys) cudaFree(d_alt_keys);
        //if (d_alt_values) cudaFree(d_alt_values);
		if (d_spine) cudaFree(d_spine);
		if (d_from_alt_storage) cudaFree(d_from_alt_storage);
		
		return cudaSuccess;
	}
};

#endif


Sort::Sort()
{
#ifdef USE_CHAG_PP_SORT
	size_t count = 16*1024; //XXX Stupid hack. should use JSetup::JOB_BUFFER_SIZE

	cudaMalloc( (void**)&counts, sizeof(chag::pp::SizeType)*count );
	cudaMalloc( (void**)&offsets, sizeof(chag::pp::SizeType)*count );
#endif
}
Sort::~Sort()
{
#ifdef USE_CHAG_PP_SORT
	cudaFree( offsets );
	cudaFree( counts );
#endif
}

void Sort::operator()(uint *&pData0, 
                      uint *&pData1,
                      uint aNumElements,
                      uint aNumBits
                      ) const
{
#ifdef USE_CHAG_PP_SORT

	using namespace chag::pp;
	using namespace chag::pp::aspect;

#	define DO_PASS_(Low_,High_) \
		Sorter< \
			sort::InputType<uint2>, \
			sort::Predicate<BitUnsetForPairs>, \
			sort::LimitLow<Low_>, \
			sort::LimitHigh<High_> \
		>::sort( (uint2*)pData0, (uint2*)pData0+aNumElements, (uint2*)pData1, (uint2*)pData0, \
			(chag::pp::SizeType*)offsets, (chag::pp::SizeType*)counts) \
		/*ENDM*/

	if( aNumBits < 10 ) DO_PASS_(0u,10u);
	//else if( aNumBits < 12 ) DO_PASS_(0u,12u);
	//else if( aNumBits < 14 ) DO_PASS_(0u,14u);
	else if( aNumBits < 16 ) DO_PASS_(0u,16u);
	//else if( aNumBits < 18 ) DO_PASS_(0u,18u);
	else if( aNumBits < 20 ) DO_PASS_(0u,20u);
	//else if( aNumBits < 22 ) DO_PASS_(0u,22u);
	else if( aNumBits < 24 ) DO_PASS_(0u,24u);
	//else if( aNumBits < 26 ) DO_PASS_(0u,26u);
	//else if( aNumBits < 28 ) DO_PASS_(0u,28u);
	//else if( aNumBits < 30 ) DO_PASS_(0u,30u);
	else DO_PASS_(0u,32u);

#	undef DO_PASS_

#else
    //DEBUG
    //cudaEvent_t sortEvent;
    //cudaEventCreate(&sortEvent);

    uint* keys = pData1;
    uint* values = pData1 + aNumElements;

    //align values pointer if necessary
    if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*values), 2*sizeof(uint)))
    {
        size_t rem = thrust::detail::uintptr_t(values) % (2*sizeof(uint));
        values = reinterpret_cast<uint*>(thrust::detail::uintptr_t(values) + 2*sizeof(uint) - rem);
    }

    uint* alt_keys = pData0;
    uint* alt_values = pData0 + aNumElements;

    //align values pointer if necessary
    if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*alt_values), 2*sizeof(uint)))
    {
        size_t rem = thrust::detail::uintptr_t(alt_values) % (2*sizeof(uint));
        alt_values = reinterpret_cast<uint*>(thrust::detail::uintptr_t(alt_values) + 2*sizeof(uint) - rem);
    }

    dim3 blockSize = 256;
    dim3 gridSize  = 128;
    PairsToSingles<<< gridSize, blockSize>>>((uint2*)alt_keys, keys, values, aNumElements);

    //DEBUG
    //if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*alt_values), 2*sizeof(uint)))
    //{
    //    cudastd::logger::out << "Error: Radix Sort Ping Values : pointer is not aligned!\n";
    //}
    //DEBUG
    //if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*values), 2*sizeof(uint)))
    //{
    //    cudastd::logger::out << "Error: Radix Sort Values :  Pointer is not aligned!\n";
    //}

    thrust::system::cuda::detail::detail::b40c_thrust::RadixSortingEnactor<uint,uint> sorter(aNumElements);
    PersistentRadixSortStorage  storage;

    //typedef thrust::detail::cuda_device_space_tag space;
    // allocate temporary buffers
    //thrust::detail::uninitialized_array<uint,    space> temp_keys(num_elements);
    //thrust::detail::uninitialized_array<uint,    space> temp_values(num_elements);
    //thrust::system::cuda::detail::uninitialized_array<int> temp_spine(sorter.SpineElements());
    //thrust::system::cuda::detail::uninitialized_array<bool> temp_from_alt(2);
    int* temp_spine;
    cudaMalloc((void**)&temp_spine, sorter.SpineElements() * sizeof(int));
    bool* temp_from_alt;
    cudaMalloc((void**)&temp_from_alt, 2 * sizeof(bool));

    // define storage
    storage.d_keys             = thrust::raw_pointer_cast(&*keys);
    storage.d_values           = thrust::raw_pointer_cast(&*values);
    storage.d_alt_keys         = thrust::raw_pointer_cast(&*alt_keys);
    storage.d_alt_values       = thrust::raw_pointer_cast(&*alt_values);
    storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
    storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);

    // perform the sort
    sorter.EnactSort(storage);

    //DEBUG
    //cudaEventSynchronize(sortEvent);
    //MY_CUT_CHECK_ERROR("Sort failed!\n");

    uint* pData2;
    // radix sort sometimes leaves results in the alternate buffers
    if (storage.using_alternate_storage)
    {
        pData0 = keys;
        pData1 = alt_keys;
        pData2 = alt_values;
    }
    else
    {
        pData0 = alt_keys;
        pData1 = keys;
        pData2 = values;
    }

    SinglesToPairs<<< gridSize, blockSize>>>((uint2*)pData0, pData1, pData2, aNumElements);
    cudaFree(temp_spine);
    cudaFree(temp_from_alt);
    //DEBUG
    //cudaEventSynchronize(sortEvent);    
    //cudaEventDestroy(sortEvent);

#endif
}

#undef USE_CHAG_PP_SORT
