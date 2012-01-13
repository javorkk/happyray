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

#include "Scan.h"

#if HAPPYRAY__CUDA_ARCH__ < 200
#   define USE_CHAG_PP_SCAN
#endif

#ifdef USE_CHAG_PP_SCAN
#   include "chag/pp/prefix.cuh"
#else
#   include <thrust/detail/backend/cuda/scan.h>
#endif




void ExclusiveScan::operator()(
        uint* aIn, 
        const uint aNumElements
        ) const
{
#ifdef USE_CHAG_PP_SCAN
    chag::pp::prefix(aIn, (aIn + aNumElements), aIn, (uint*)0);
#else
    thrust::detail::backend::cuda::exclusive_scan(aIn, (aIn + aNumElements), aIn, 0u, OperatorPlus());
#endif
}

void InclusiveScan::operator()(
                       uint* aIn, 
                       const uint aNumElements
                       ) const
{
#ifdef USE_CHAG_PP_SCAN
    chag::pp::prefix_inclusive(aIn, (aIn + aNumElements), aIn, (uint*)0);
#else
    thrust::detail::backend::cuda::inclusive_scan(aIn, (aIn + aNumElements), aIn, OperatorPlus());
#endif

}

void InclusiveFloatScan::operator()(
                       float* aIn, 
                       const uint aNumElements
                       ) const
{
#ifdef USE_CHAG_PP_SCAN
    chag::pp::prefix_inclusive(aIn, (aIn + aNumElements), aIn, (float*)0);
#else
    thrust::detail::backend::cuda::inclusive_scan(aIn, (aIn + aNumElements), aIn, OperatorPlus());
#endif

}

#undef USE_CHAG_PP_SCAN
