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

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

cub::CachingDeviceAllocator  gSortAllocator(true);  // Caching allocator for device memory

void RadixSort::operator()(
    uint *&aKeysPing,
    uint *&aKeysPong,
    uint *&aValuesPing,
    uint *&aValuesPong,
    uint aNumElements
    )
{
    cub::DoubleBuffer<uint> d_keys(aKeysPing, aKeysPong);
    cub::DoubleBuffer<uint> d_values(aValuesPing, aValuesPong);

    size_t  temp_storage_bytes = 0;
    void    *d_temp_storage = NULL;

    //Initialize
    CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, aNumElements));
    CubDebugExit(gSortAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    //Sort
    CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, aNumElements));
    //Cleanup
    if (d_temp_storage) CubDebugExit(gSortAllocator.DeviceFree(d_temp_storage));

}