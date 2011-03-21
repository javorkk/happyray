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

#ifndef UNIFORMGRIDBUILDKERNELS_H_974ACF4A_E8D2_4D1D_8471_7D922CF44556
#define UNIFORMGRIDBUILDKERNELS_H_974ACF4A_E8D2_4D1D_8471_7D922CF44556

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"


template<class tPrimitive, template<class> class tPrimitiveArray, int taBlockSize>
GLOBAL void countPairs(
                      tPrimitiveArray<tPrimitive>   aPrimitiveArray,
                      const uint                    aNumPrimitives,
                      const float3                  aGridRes,
                      const float3                  aBoundsMin,
                      const float3                  aCellSize,
                      const float3                  aCellSizeRCP,
                      uint*                         oRefCounts
                      )
{
    extern SHARED uint shMem[];
    shMem[threadId1D()] = 0u;

    const float EPSILON = (aCellSizeRCP.x + aCellSizeRCP.y + aCellSizeRCP.z) * 10E-5f;

    for(int primitiveId = globalThreadId1D(); primitiveId < aNumPrimitives; primitiveId += numThreads())
    {
        const tPrimitive prim = aPrimitiveArray[primitiveId];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);
        
        float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        minCellIdf =
            max(rep(0.f), (bounds.vtx[0] - aBoundsMin) * aCellSizeRCP + rep(-EPSILON));
        const float3 maxCellIdf =
            min(aGridRes - rep(1.f), (bounds.vtx[1] - aBoundsMin) * aCellSizeRCP + rep(EPSILON));

        const uint minCellIdX =  (uint)(minCellIdf.x);
        const uint minCellIdY =  (uint)(minCellIdf.y);
        const uint minCellIdZ =  (uint)(minCellIdf.z);

        const uint maxCellIdX =  (uint)(maxCellIdf.x);
        const uint maxCellIdY =  (uint)(maxCellIdf.y);
        const uint maxCellIdZ =  (uint)(maxCellIdf.z);

        shMem[threadId1D()] += (maxCellIdX - minCellIdX + 1)
            * (maxCellIdY - minCellIdY + 1)
            * (maxCellIdZ - minCellIdZ + 1);
    }

    SYNCTHREADS;

#if HAPPYRAY__CUDA_ARCH__ >= 120

    //reduction
    if (taBlockSize >= 512) { if (threadId1D() < 256) { shMem[threadId1D()] += shMem[threadId1D() + 256]; } SYNCTHREADS;   }
    if (taBlockSize >= 256) { if (threadId1D() < 128) { shMem[threadId1D()] += shMem[threadId1D() + 128]; } SYNCTHREADS;   }
    if (taBlockSize >= 128) { if (threadId1D() <  64) { shMem[threadId1D()] += shMem[threadId1D() +  64]; } SYNCTHREADS;   }
    if (taBlockSize >=  64) { if (threadId1D() <  32) { shMem[threadId1D()] += shMem[threadId1D() +  32]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  32) { if (threadId1D() <  16) { shMem[threadId1D()] += shMem[threadId1D() +  16]; } EMUSYNCTHREADS;}
    if (taBlockSize >=  16) { if (threadId1D() <   8) { shMem[threadId1D()] += shMem[threadId1D() +   8]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   8) { if (threadId1D() <   4) { shMem[threadId1D()] += shMem[threadId1D() +   4]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   4) { if (threadId1D() <   2) { shMem[threadId1D()] += shMem[threadId1D() +   2]; } EMUSYNCTHREADS;}
    if (taBlockSize >=   2) { if (threadId1D() <   1) { shMem[threadId1D()] += shMem[threadId1D() +   1]; } EMUSYNCTHREADS;}

    // write out block sum 
    if (threadId1D() == 0) oRefCounts[blockId1D()] = shMem[0];

#else

    oRefCounts[globalThreadId1D()] = shMem[threadId1D()];

#endif
}

template<class tPrimitive, template<class> class tPrimitiveArray>
GLOBAL void writePairs(
                        tPrimitiveArray<tPrimitive> aPrimitiveArray,
                        uint*                       oPairs,
                        const uint                  aNumPrimitives,
                        uint*                       aStartId,
                        const float3                 aGridRes,
                        const float3                 aBoundsMin,
                        const float3                 aCellSize,
                        const float3                 aCellSizeRCP
                              )
{
    extern SHARED uint shMem[];

#if HAPPYRAY__CUDA_ARCH__ >= 120

    if (threadId1D() == 0)
    {
        shMem[0] = aStartId[blockId1D()];
    }

    SYNCTHREADS;

#else

    uint startPosition = aStartId[globalThreadId1D()];

#endif

    const float EPSILON = (aCellSizeRCP.x + aCellSizeRCP.y + aCellSizeRCP.z) * 10E-5f;

    for(int primitiveId = globalThreadId1D(); primitiveId < aNumPrimitives; primitiveId += numThreads())
    {
        const tPrimitive prim = aPrimitiveArray[primitiveId];
        BBox bounds = BBoxExtractor<tPrimitive>::get(prim);
        
        //float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        float3 minCellIdf =
            max(rep(0.f), (bounds.vtx[0] - aBoundsMin) * aCellSizeRCP + rep(-EPSILON));
        const float3 maxCellIdf =
            min(aGridRes - rep(1.f), (bounds.vtx[1] - aBoundsMin) * aCellSizeRCP + rep(EPSILON));

        const uint minCellIdX =  (uint)(minCellIdf.x);
        const uint minCellIdY =  (uint)(minCellIdf.y);
        const uint minCellIdZ =  (uint)(minCellIdf.z);

        const uint maxCellIdX =  (uint)(maxCellIdf.x);
        const uint maxCellIdY =  (uint)(maxCellIdf.y);
        const uint maxCellIdZ =  (uint)(maxCellIdf.z);

        const uint numCells =
            (maxCellIdX - minCellIdX + 1u) *
            (maxCellIdY - minCellIdY + 1u) *
            (maxCellIdZ - minCellIdZ + 1u);

#if HAPPYRAY__CUDA_ARCH__ >= 120
        uint nextSlot  = atomicAdd(&shMem[0], numCells);
#else
        uint nextSlot = startPosition;
        startPosition += numCells;
#endif

        for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
        {
            for (uint y = minCellIdY; y <= maxCellIdY; ++y)
            {
                for (uint x = minCellIdX; x <= maxCellIdX; ++x, ++nextSlot)
                {
                    oPairs[2 * nextSlot] = x +
                        y * (uint)aGridRes.x +
                        z * (uint)(aGridRes.x * aGridRes.y);
                    oPairs[2 * nextSlot + 1] = primitiveId;
                }//end for z
            }//end for y
        }//end for x

    }
}

template<int taBlockSize>
GLOBAL void prepareCellRanges(
                              uint*             oPrimitiveIndices,
                              uint2*            aSortedPairs,
                              const uint        aNumInstances,
                              cudaPitchedPtr    aGridCellsPtr,
                              const uint        aGridResX,
                              const uint        aGridResY,
                              const uint        aGridResZ
                              )
{
    extern SHARED uint shMem[];

    //padding
    if (threadId1D() == 0)
    {
        shMem[0] = 0u;
        shMem[taBlockSize] = 0u;
    }

    uint *padShMem = shMem + 1;
    padShMem[threadId1D()] = 0u;

    SYNCTHREADS;

    for(int instanceId = globalThreadId1D();
        instanceId < aNumInstances + blockSize() - 1;
        instanceId += numThreads())
    {
        //load blockSize() + 2 input elements in shared memory

        SYNCTHREADS;

        if (threadId1D() == 0 && instanceId > 0u)
        {
            //padding left
            shMem[0] = aSortedPairs[instanceId - 1].x;
        }
        if (threadId1D() == 0 && instanceId + blockSize() < aNumInstances)
        {
            //padding right
            padShMem[blockSize()] = aSortedPairs[instanceId + blockSize()].x;
        }
        if (instanceId < aNumInstances)
        {
            padShMem[threadId1D()] = aSortedPairs[instanceId].x;
        }

        SYNCTHREADS;

        //Check if the two neighboring cell indices are different
        //which means that at this point there is an end of and a begin of a range

        //compare left neighbor
        if (instanceId > 0 && instanceId < aNumInstances && padShMem[threadId1D()] != shMem[threadId1D()])
        {
            //begin of range
            uint cellIdX =  padShMem[threadId1D()] % aGridResX;
            uint cellIdY = (padShMem[threadId1D()] % (aGridResX * aGridResY)) / aGridResX;
            uint cellIdZ =  padShMem[threadId1D()] / (aGridResX * aGridResY);

            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                + cellIdY * aGridCellsPtr.pitch
                + cellIdZ * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + cellIdX;

            cell->x = instanceId;
        }

        //compare right neighbor
        if (instanceId < aNumInstances && padShMem[threadId1D()] != padShMem[threadId1D() + 1])
        {
            //end of range
            uint cellIdX =  padShMem[threadId1D()] % aGridResX;
            uint cellIdY = (padShMem[threadId1D()] % (aGridResX * aGridResY)) / aGridResX;
            uint cellIdZ =  padShMem[threadId1D()] / (aGridResX * aGridResY);

            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                + cellIdY * aGridCellsPtr.pitch
                + cellIdZ * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + cellIdX;

            cell->y = instanceId + 1;
        }

    }//end for(uint startId = blockId1D() * blockSize()...

    SYNCTHREADS;

    //compact triangle indices from aInIndices to the output indices
    for(int instanceId = globalThreadId1D();
        instanceId < aNumInstances;
        instanceId += numThreads())
    {
        oPrimitiveIndices[instanceId] = aSortedPairs[instanceId].y;
    }

}


template<class tPrimitive, template<class> class tPrimitiveArray>
GLOBAL void checkGridCells(
                    tPrimitiveArray<tPrimitive>     aPrimitives,
                    uint*                           aPrimitiveIndexIndirection,
                    cudaPitchedPtr                  aGridCellsPtr,
                    const float3                     aGridRes)
{

    uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
        + blockIdx.x * aGridCellsPtr.pitch
        + blockIdx.y * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + threadIdx.x;

    uint2 cellRange = *cell;

    for(uint it = cellRange.x; it != cellRange.y; ++ it)
    {
        tPrimitive prim = aPrimitives[aPrimitiveIndexIndirection[it]];
    }

    cellRange.x = aGridCellsPtr.pitch / sizeof(uint2);
}


#endif // UNIFORMGRIDBUILDKERNELS_H_974ACF4A_E8D2_4D1D_8471_7D922CF44556
