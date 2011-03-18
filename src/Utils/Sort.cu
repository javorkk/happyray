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
#include "chag/pp/sort.cuh"

Sort::Sort()
{
	size_t count = 16*1024; //XXX Stupid hack. should use JSetup::JOB_BUFFER_SIZE

	cudaMalloc( (void**)&counts, sizeof(chag::pp::SizeType)*count );
	cudaMalloc( (void**)&offsets, sizeof(chag::pp::SizeType)*count );
}
Sort::~Sort()
{
	cudaFree( offsets );
	cudaFree( counts );
}

void Sort::operator()(uint2 *pData0, 
                      uint2 *pData1,
                      uint aNumElements,
                      uint aNumBits
                      ) const
{
	using namespace chag::pp;
	using namespace chag::pp::aspect;

#	define DO_PASS_(Low_,High_) \
		Sorter< \
			sort::InputType<uint2>, \
			sort::Predicate<BitUnsetForPairs>, \
			sort::LimitLow<Low_>, \
			sort::LimitHigh<High_> \
		>::sort( pData0, pData0+aNumElements, pData1, pData0, \
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
}
