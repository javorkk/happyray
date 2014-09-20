#include "CUDAStdAfx.h"
#include "RT/Algorithm/UniformGridBuildKernels.h"

//////////////////////////////////////////////////////////////////////////
//axis tests
//////////////////////////////////////////////////////////////////////////

DEVICE bool axisTest(
    const float a,
    const float b,
    const float fa,
    const float fb,
    const float v0a,
    const float v0b,
    const float v1a,
    const float v1b,
    const float aCellSizeHALFa,
    const float aCellSizeHALFb)
{
    const float p0 = a * v0a + b * v0b;
    const float p1 = a * v1a + b * v1b;

    const float minP = fminf(p0, p1);
    const float maxP = fmaxf(p0, p1);

    const float rad = fa * aCellSizeHALFa + fb * aCellSizeHALFb;

    return ! (minP > rad + EPS || maxP + EPS < -rad);

}

#define AXISTEST_X01(e, fe, v0, v1, v2, s)                                     \
    axisTest(e.z, -e.y, fe.z, fe.y, v0.y, v0.z, v2.y, v2.z, s.y, s.z)

#define AXISTEST_X2(e, fe, v0, v1, v2, s)                                      \
    axisTest(e.z, -e.y, fe.z, fe.y, v0.y, v0.z, v1.y, v1.z, s.y, s.z)

#define AXISTEST_Y02(e, fe, v0, v1, v2, s)                                     \
    axisTest(-e.z, e.x, fe.z, fe.x, v0.x, v0.z, v2.x, v2.z, s.x, s.z)

#define AXISTEST_Y1(e, fe, v0, v1, v2, s)                                      \
    axisTest(-e.z, e.x, fe.z, fe.x, v0.x, v0.z, v1.x, v1.z, s.x, s.z)

#define AXISTEST_Z12(e, fe, v0, v1, v2, s)                                     \
    axisTest(e.y, -e.x, fe.y, fe.x, v1.x, v1.y, v2.x, v2.y, s.x, s.y)

#define AXISTEST_Z0(e, fe, v0, v1, v2, s)                                      \
    axisTest(e.y, -e.x, fe.y, fe.x, v0.x, v0.y, v1.x, v1.y, s.x, s.y)

//////////////////////////////////////////////////////////////////////////

template<>
GLOBAL void writePairs<Triangle, PrimitiveArray<Triangle>, true>(
    PrimitiveArray<Triangle> aPrimitiveArray,
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

    for(int triangleId = globalThreadId1D(); triangleId < aNumPrimitives; triangleId += numThreads())
    {
        const Triangle triangle = aPrimitiveArray[triangleId];
        BBox bounds = BBoxExtractor<Triangle>::get(triangle);

        //float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        const float3 minCellIdf = (bounds.vtx[0] - aBoundsMin) * aCellSizeRCP;
        const float3 maxCellIdPlus1f = (bounds.vtx[1] - aBoundsMin) * aCellSizeRCP + rep(1.f);

        const int minCellIdX =   max(0, (int)(minCellIdf.x));
        const int minCellIdY =   max(0, (int)(minCellIdf.y));
        const int minCellIdZ =   max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)aGridRes.x, (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)aGridRes.y, (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)aGridRes.z, (int)(maxCellIdPlus1f.z));
        const int numCells   = 
            (maxCellIdP1X - minCellIdX )
            * (maxCellIdP1Y - minCellIdY )
            * (maxCellIdP1Z - minCellIdZ );

#if HAPPYRAY__CUDA_ARCH__ >= 120
        uint nextSlot  = atomicAdd(&shMem[0], numCells);
#else
        uint nextSlot = startPosition;
        startPosition += numCells;
#endif



        const float3 normal =
            ~((triangle.vtx[1] - triangle.vtx[0]) %
            (triangle.vtx[2] - triangle.vtx[0]));

        const float3 gridCellSizeHALF = aCellSize * 0.505f; //1% extra as epsilon
        float3 minCellCenter;
        minCellCenter.x = (float)(minCellIdX);
        minCellCenter.y = (float)(minCellIdY);
        minCellCenter.z = (float)(minCellIdZ);
        minCellCenter =  minCellCenter * aCellSize;
        minCellCenter = minCellCenter + aBoundsMin + gridCellSizeHALF;

        float3 cellCenter;
        cellCenter.z = minCellCenter.z - aCellSize.z;

        for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
        {
            cellCenter.z += aCellSize.z;
            cellCenter.y = minCellCenter.y - aCellSize.y;

            for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
            {
                cellCenter.y += aCellSize.y;
                cellCenter.x = minCellCenter.x - aCellSize.x;

                for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                {
                    cellCenter.x += aCellSize.x;

                    //////////////////////////////////////////////////////////////////////////
                    //coordinate transform origin -> cellCenter
                    const float3 v0 = triangle.vtx[0] - cellCenter;
                    const float3 v1 = triangle.vtx[1] - cellCenter;
                    const float3 v2 = triangle.vtx[2] - cellCenter;
                    const float3 e0 = v1 - v0;
                    const float3 e1 = v2 - v1;
                    const float3 e2 = v0 - v2;

                    bool passedAllTests = true;
                    //9 tests for separating axis
                    float3 fe;
                    fe.x = fabsf(e0.x);
                    fe.y = fabsf(e0.y);
                    fe.z = fabsf(e0.z);

                    passedAllTests = passedAllTests && AXISTEST_X01(e0, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y02(e0, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z12(e0, fe, v0, v1, v2, gridCellSizeHALF);

                    fe.x = fabsf(e1.x);
                    fe.y = fabsf(e1.y);
                    fe.z = fabsf(e1.z);

                    passedAllTests = passedAllTests && AXISTEST_X01(e1, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y02(e1, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z0(e1, fe, v0, v1, v2, gridCellSizeHALF);

                    fe.x = fabsf(e2.x);
                    fe.y = fabsf(e2.y);
                    fe.z = fabsf(e2.z);

                    passedAllTests = passedAllTests && AXISTEST_X2(e2, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y1(e2, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z12(e2, fe, v0, v1, v2, gridCellSizeHALF);

                    //////////////////////////////////////////////////////////////////////////
                    //Plane/box overlap test
                    float3 vmin, vmax;
                    vmin.x = (normal.x > 0.f) ? -gridCellSizeHALF.x : gridCellSizeHALF.x;
                    vmin.y = (normal.y > 0.f) ? -gridCellSizeHALF.y : gridCellSizeHALF.y;
                    vmin.z = (normal.z > 0.f) ? -gridCellSizeHALF.z : gridCellSizeHALF.z;

                    vmax = -vmin;
                    vmax = vmax - v0;
                    vmin = vmin - v0;

                    passedAllTests = passedAllTests && dot(normal, vmin) <= 0.f && dot(normal, vmax) > 0.f;
                    //////////////////////////////////////////////////////////////////////////

                    if (passedAllTests)
                    {
                        oPairs[2 * nextSlot] = x +
                            y * (uint)aGridRes.x +
                            z * (uint)(aGridRes.x * aGridRes.y);

                        oPairs[2 * nextSlot + 1] =
                            triangleId;
                    }
                    else
                    {
                        oPairs[2 * nextSlot] = 
                            (uint)(aGridRes.x * aGridRes.y * aGridRes.z);

                        oPairs[2 * nextSlot + 1] = 
                            triangleId;
                    }
                }//end for z
            }//end for y
        }//end for x
    }
}

template<>
GLOBAL void writeKeysAndValues<Triangle, PrimitiveArray<Triangle>, true>(
    PrimitiveArray<Triangle> aPrimitiveArray,
    uint*                       oKeys,
    uint*                       oValues,
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

    for(int triangleId = globalThreadId1D(); triangleId < aNumPrimitives; triangleId += numThreads())
    {
        const Triangle triangle = aPrimitiveArray[triangleId];
        BBox bounds = BBoxExtractor<Triangle>::get(triangle);

        //float3& minCellIdf = ((float3*)(shMem + blockSize()))[threadId1D()];
        const float3 minCellIdf = (bounds.vtx[0] - aBoundsMin) * aCellSizeRCP;
        const float3 maxCellIdPlus1f = (bounds.vtx[1] - aBoundsMin) * aCellSizeRCP + rep(1.f);

        const int minCellIdX =   max(0, (int)(minCellIdf.x));
        const int minCellIdY =   max(0, (int)(minCellIdf.y));
        const int minCellIdZ =   max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)aGridRes.x, (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)aGridRes.y, (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)aGridRes.z, (int)(maxCellIdPlus1f.z));
        const int numCells   = 
            (maxCellIdP1X - minCellIdX )
            * (maxCellIdP1Y - minCellIdY )
            * (maxCellIdP1Z - minCellIdZ );

#if HAPPYRAY__CUDA_ARCH__ >= 120
        uint nextSlot  = atomicAdd(&shMem[0], numCells);
#else
        uint nextSlot = startPosition;
        startPosition += numCells;
#endif



        const float3 normal =
            ~((triangle.vtx[1] - triangle.vtx[0]) %
            (triangle.vtx[2] - triangle.vtx[0]));

        const float3 gridCellSizeHALF = aCellSize * 0.505f; //1% extra as epsilon
        float3 minCellCenter;
        minCellCenter.x = (float)(minCellIdX);
        minCellCenter.y = (float)(minCellIdY);
        minCellCenter.z = (float)(minCellIdZ);
        minCellCenter =  minCellCenter * aCellSize;
        minCellCenter = minCellCenter + aBoundsMin + gridCellSizeHALF;

        float3 cellCenter;
        cellCenter.z = minCellCenter.z - aCellSize.z;

        for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
        {
            cellCenter.z += aCellSize.z;
            cellCenter.y = minCellCenter.y - aCellSize.y;

            for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
            {
                cellCenter.y += aCellSize.y;
                cellCenter.x = minCellCenter.x - aCellSize.x;

                for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                {
                    cellCenter.x += aCellSize.x;

                    //////////////////////////////////////////////////////////////////////////
                    //coordinate transform origin -> cellCenter
                    const float3 v0 = triangle.vtx[0] - cellCenter;
                    const float3 v1 = triangle.vtx[1] - cellCenter;
                    const float3 v2 = triangle.vtx[2] - cellCenter;
                    const float3 e0 = v1 - v0;
                    const float3 e1 = v2 - v1;
                    const float3 e2 = v0 - v2;

                    bool passedAllTests = true;
                    //9 tests for separating axis
                    float3 fe;
                    fe.x = fabsf(e0.x);
                    fe.y = fabsf(e0.y);
                    fe.z = fabsf(e0.z);

                    passedAllTests = passedAllTests && AXISTEST_X01(e0, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y02(e0, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z12(e0, fe, v0, v1, v2, gridCellSizeHALF);

                    fe.x = fabsf(e1.x);
                    fe.y = fabsf(e1.y);
                    fe.z = fabsf(e1.z);

                    passedAllTests = passedAllTests && AXISTEST_X01(e1, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y02(e1, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z0(e1, fe, v0, v1, v2, gridCellSizeHALF);

                    fe.x = fabsf(e2.x);
                    fe.y = fabsf(e2.y);
                    fe.z = fabsf(e2.z);

                    passedAllTests = passedAllTests && AXISTEST_X2(e2, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Y1(e2, fe, v0, v1, v2, gridCellSizeHALF);
                    passedAllTests = passedAllTests && AXISTEST_Z12(e2, fe, v0, v1, v2, gridCellSizeHALF);

                    //////////////////////////////////////////////////////////////////////////
                    //Plane/box overlap test
                    float3 vmin, vmax;
                    vmin.x = (normal.x > 0.f) ? -gridCellSizeHALF.x : gridCellSizeHALF.x;
                    vmin.y = (normal.y > 0.f) ? -gridCellSizeHALF.y : gridCellSizeHALF.y;
                    vmin.z = (normal.z > 0.f) ? -gridCellSizeHALF.z : gridCellSizeHALF.z;

                    vmax = -vmin;
                    vmax = vmax - v0;
                    vmin = vmin - v0;

                    passedAllTests = passedAllTests && dot(normal, vmin) <= 0.f && dot(normal, vmax) > 0.f;
                    //////////////////////////////////////////////////////////////////////////

                    if (passedAllTests)
                    {
                        oKeys[nextSlot] = x +
                            y * (uint)aGridRes.x +
                            z * (uint)(aGridRes.x * aGridRes.y);

                        oValues[nextSlot + 1] =
                            triangleId;
                    }
                    else
                    {
                        oKeys[nextSlot] = 
                            (uint)(aGridRes.x * aGridRes.y * aGridRes.z);

                        oValues[nextSlot + 1] = 
                            triangleId;
                    }
                }//end for z
            }//end for y
        }//end for x
    }
}
