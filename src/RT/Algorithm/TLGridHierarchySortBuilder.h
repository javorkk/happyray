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

#ifdef _MSC_VER
#pragma once
#endif

#ifndef TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448
#define TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/TLGridHierarchyMemoryManager.h"


class TLGridHierarchySortBuilder
{
    static const uint   sNUM_COUNTER_THREADS    = 128u;
    static const uint   sNUM_COUNTER_BLOCKS     =  90u;
    //NOTE: WRITE and COUNTER threads and blocks have to be exactly the same
    static const uint   sNUM_WRITE_THREADS      = sNUM_COUNTER_THREADS;
    static const uint   sNUM_WRITE_BLOCKS       = sNUM_COUNTER_BLOCKS;
    static const uint   sNUM_CELL_SETUP_THREADS = 256u;
    static const uint   sNUM_CELL_SETUP_BLOCKS  = 240u;

    cudaEvent_t mStart, mDataUpload, mScan;
    cudaEvent_t mTopLevel, mLeafCellCount, mLeafRefsCount,
        mLeafRefsWrite, mSortLeafPairs, mEnd;

    bool mSetResolution;

public:
    HOST void init(
        TLGridHierarchyMemoryManager&   aMemoryManager,
        const uint                      aNumInstances,
        const float                     aTopLevelDensity = 1.2f,
        const float                     aLeafLevelDensity = 5.0f,
        const bool                      aResetGridResolution = true
        );

    HOST void build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        PrimitiveArray<Triangle>&         aPrimitiveArray);

    HOST void build(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint                                aNumUniqueInstances,
        uint*                               aPrimitiveCounts,
        PrimitiveArray<Triangle>&         aPrimitiveArray);

    //dummy to use instead of init
    HOST void initEvents(
        TLGridHierarchyMemoryManager&   aMemoryManager,
        const uint                      aNumInstances,
        const float                     aTopLevelDensity = 0.0625f,
        const float                     aLeafLevelDensity = 1.2f
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        cudaEventSynchronize(mStart);
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void buildLevelTwo(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint*                               aPrimitiveCounts,
        const uint                          aNumGrids,
        PrimitiveArray<Triangle>&         aPrimitiveArray);

    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mTopLevel);
        cudaEventDestroy(mLeafCellCount);
        cudaEventDestroy(mLeafRefsCount);
        cudaEventDestroy(mLeafRefsWrite);
        cudaEventDestroy(mSortLeafPairs);
        cudaEventDestroy(mEnd);
    }
    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::logger::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::logger::out << "Data upload:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mTopLevel);
        cudastd::logger::out << "Top Level:        " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mEnd);
        cudastd::logger::out << "Leaf Level:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mLeafCellCount);
        cudastd::logger::out << "Leaf Cells Count: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafCellCount, mLeafRefsCount);
        cudastd::logger::out << "Leaf Refs Count:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsCount, mLeafRefsWrite);
        cudastd::logger::out << "Leaf Refs Write:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsWrite, mSortLeafPairs);
        cudastd::logger::out << "Leaf Refs Sort:   " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime,mSortLeafPairs, mEnd);
        cudastd::logger::out << "Prep. Leaf cells: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::logger::out << "Total:            " << elapsedTime << "ms\n";
        //////////////////////////////////////////////////////////////////////////
    }

    HOST void test(
        TLGridHierarchyMemoryManager&       aMemoryManager,
        uint                                aNumUniqueInstances,
        uint*                               aPrimitiveCounts,
        PrimitiveArray<Triangle>&         aPrimitiveArray);
};

#endif // TLGRIDHIERARCHYSORTBUILDER_H_INCLUDED_CAEFCB99_E3D6_4FAA_8E63_9E388B12B448
