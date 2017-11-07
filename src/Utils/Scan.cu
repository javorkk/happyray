/****************************************************************************/
/* Copyright (c) 2009, Stefan Popov, Javor Kalojanov
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
#include "Scan.h"

#define CUB_SCAN
#ifndef CUB_SCAN
#   include <thrust/device_ptr.h>
#   include <thrust/scan.h>
#else
#   include <cub/util_allocator.cuh>
#   include <cub/device/device_scan.cuh>

cub::CachingDeviceAllocator  gScanAllocator(true);
#endif


void ExclusiveScan::operator()(
        uint* aIn, 
        const uint aNumElements
        ) const
{
#ifndef CUB_SCAN
    thrust::device_ptr<unsigned int> dev_ptr = thrust::device_pointer_cast(aIn);
    thrust::exclusive_scan(dev_ptr, (dev_ptr + aNumElements), dev_ptr);
#else
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    //Initialize
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    CubDebugExit(gScanAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    //Scan
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    //Cleanup
    if (d_temp_storage) CubDebugExit(gScanAllocator.DeviceFree(d_temp_storage));
#endif
}

void InclusiveScan::operator()(
                       uint* aIn, 
                       const uint aNumElements
                       ) const
{
#ifndef CUB_SCAN
    thrust::device_ptr<unsigned int> dev_ptr = thrust::device_pointer_cast(aIn);
    thrust::inclusive_scan(dev_ptr, (dev_ptr + aNumElements), dev_ptr);
#else
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    //Initialize
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    CubDebugExit(gScanAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    //Scan
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    //Cleanup
    if (d_temp_storage) CubDebugExit(gScanAllocator.DeviceFree(d_temp_storage));
#endif
}

void InclusiveFloatScan::operator()(
                       float* aIn, 
                       const uint aNumElements
                       ) const
{
#ifndef CUB_SCAN
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(aIn);
    thrust::inclusive_scan(dev_ptr, (dev_ptr + aNumElements), dev_ptr);
#else
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    //Initialize
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    CubDebugExit(gScanAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    //Scan
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, aIn, aIn, aNumElements));
    //Cleanup
    if (d_temp_storage) CubDebugExit(gScanAllocator.DeviceFree(d_temp_storage));
#endif
}

#ifdef CUB_SCAN
#undef CUB_SCAN
#endif