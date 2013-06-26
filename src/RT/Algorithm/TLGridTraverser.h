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

#ifndef TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34
#define TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34


#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

#include "RT/Structure/TwoLevelGrid.h"
#include "RT/Structure/PrimitiveArray.h"

/////////////////////////////////////////////////////////////////
//Two-Level Grid Traversal Classes
/////////////////////////////////////////////////////////////////

#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

template<class tPrimitive, class tIntersector, bool taIsShadowRay = false>
class TLGridTraverser
{
public:
    DEVICE void operator()(
        PrimitiveArray<tPrimitive>  aPrimitiveArray,
        TwoLevelGrid                dcGrid,
        float3&                     rayOrg,
        float3&                     rayDirRCP,
        float&                      rayT,
        uint&                       bestHit,
        bool                        traversalFlag,
        uint*                       sharedMemNew
        )
    {
        tIntersector intersector;
        //////////////////////////////////////////////////////////////////////////
        //Traversal State
        float tMax[3];
        int cellId[3];
        float tEntry;
        //////////////////////////////////////////////////////////////////////////

        if(taIsShadowRay)
        {
            if(rayT < 0.9999f)
            {
                traversalFlag = false;
            }
        }

        if (traversalFlag)
        {
            //////////////////////////////////////////////////////////////////////////
            //ray/box intersection test
            const float3 t1 = (dcGrid.vtx[0] - rayOrg) * rayDirRCP;
            float3 tFar = (dcGrid.vtx[1] - rayOrg) * rayDirRCP;

            const float3 tNear = min(t1, tFar);
            tFar = max(t1, tFar);

            tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);

            //BBox bounds = BBoxExtractor<TwoLevelGrid>::get(dcGrid);
            //bounds.fastClip(rayOrg, rayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////
            tEntry = fmaxf(0.f, tEntry);
            const float3 entryPt = rayOrg + rep(tEntry) / rayDirRCP;

            float3 cellIdf = 
                (entryPt - dcGrid.vtx[0]) * dcGrid.getCellSizeRCP();

            cellIdf.x = floorf(cellIdf.x);
            cellIdf.y = floorf(cellIdf.y);
            cellIdf.z = floorf(cellIdf.z);

            cellIdf.x = min((float)dcGrid.res[0] - 1.f, max(cellIdf.x, 0.f));
            cellIdf.y = min((float)dcGrid.res[1] - 1.f, max(cellIdf.y, 0.f));
            cellIdf.z = min((float)dcGrid.res[2] - 1.f, max(cellIdf.z, 0.f));

            float3 tmp;
            tmp.x = (rayDirRCP.x > 0.f) ? 1.f : 0.f;
            tmp.y = (rayDirRCP.y > 0.f) ? 1.f : 0.f;
            tmp.z = (rayDirRCP.z > 0.f) ? 1.f : 0.f;

            tMax[0] = ((cellIdf.x + tmp.x) * dcGrid.getCellSize().x + dcGrid.vtx[0].x - rayOrg.x) * rayDirRCP.x;
            tMax[1] = ((cellIdf.y + tmp.y) * dcGrid.getCellSize().y + dcGrid.vtx[0].y - rayOrg.y) * rayDirRCP.y;
            tMax[2] = ((cellIdf.z + tmp.z) * dcGrid.getCellSize().z + dcGrid.vtx[0].z - rayOrg.z) * rayDirRCP.z;

            cellId[0] = static_cast<int>(cellIdf.x);
            cellId[1] = static_cast<int>(cellIdf.y);
            cellId[2] = static_cast<int>(cellIdf.z);

            traversalFlag = traversalFlag && (  
                (cellId[0] < dcGrid.res[0] && cellId[0] > -1) && 
                (cellId[1] < dcGrid.res[1] && cellId[1] > -1) && 
                (cellId[2] < dcGrid.res[2] && cellId[2] > -1) 
                );
        }
        //////////////////////////////////////////////////////////////////////////
        //Outer loop
        while (ANY(traversalFlag))
        {
            TwoLevelGrid::t_Cell cell;

            if (traversalFlag)
            {
                cell = dcGrid.getCell(cellId[0], cellId[1], cellId[2]);
            }

            bool secondLvlFlag = traversalFlag && cell.notEmpty();
            float tMaxLvl2[3];
            int cellIdLvl2[3];
            float subCellSize[3];//TODO: remove?


            if (secondLvlFlag)
            {

                const float3 entryPoint = rayOrg + 
                    fastDivide(rep(tEntry), rayDirRCP);

                float3 cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const float3 subCellSizeRCP = dcGrid.getCellSizeRCP() * cellRes; 
                float3 minBound;
                minBound.x = (float)cellId[0] * dcGrid.getCellSize().x
                    + dcGrid.vtx[0].x;
                minBound.y = (float)cellId[1] * dcGrid.getCellSize().y
                    + dcGrid.vtx[0].y;
                minBound.z = (float)cellId[2] * dcGrid.getCellSize().z
                    + dcGrid.vtx[0].z;

                float3 cellIdf;
                cellIdf.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                cellIdf.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                cellIdf.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subCellSize[0] = fastDivide(1.f, subCellSizeRCP.x);
                subCellSize[1] = fastDivide(1.f, subCellSizeRCP.y);
                subCellSize[2] = fastDivide(1.f, subCellSizeRCP.z);

                cellIdf = min(cellRes - rep(1.f), max(cellIdf, rep(0.f)));

                tMaxLvl2[0] = ((cellIdf.x + 
                    ((rayDirRCP.x > 0.f) ? 1.f : 0.f ))
                    * subCellSize[0] + minBound.x
                    - rayOrg.x) * rayDirRCP.x;
                tMaxLvl2[1] = ((cellIdf.y + 
                    ((rayDirRCP.y > 0.f) ? 1.f : 0.f ))
                    * subCellSize[1] + minBound.y
                    - rayOrg.y) * rayDirRCP.y;
                tMaxLvl2[2] = ((cellIdf.z +
                    ((rayDirRCP.z > 0.f) ? 1.f : 0.f ))
                    * subCellSize[2] + minBound.z
                    - rayOrg.z) * rayDirRCP.z;

                cellIdLvl2[0] = static_cast<int>(cellIdf.x);
                cellIdLvl2[1] = static_cast<int>(cellIdf.y);
                cellIdLvl2[2] = static_cast<int>(cellIdf.z);

            }

            /////////////////////////////////////////////////////////////////////////
            //Inner loop
            while (ANY(secondLvlFlag))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (secondLvlFlag)
                {
                    cellRange = dcGrid.leaves[cell.getLeafRangeBegin()
                        + cellIdLvl2[0] + cell[0] * cellIdLvl2[1]
                    + cell[0] * cell[1] * cellIdLvl2[2]
                    ];
                }

                intersector(rayOrg, rayDirRCP, rayT, bestHit,
                    cellRange, dcGrid.primitives, aPrimitiveArray, sharedMemNew);

                if (secondLvlFlag)
                {
                    /////////////////////////////////////////////////////////////////////////
                    //Traverse to next leaf
                    const int tMinDimension =
                        MIN_DIMENSION(tMaxLvl2[0], tMaxLvl2[1], tMaxLvl2[2]);

                    if(taIsShadowRay)
                    {
                        if(rayT < 0.9999f)
                        {
                            secondLvlFlag = false;
                            traversalFlag = false;
                        }
                    }

                    if(rayT < tMaxLvl2[tMinDimension])
                    {
                        secondLvlFlag = false;
                        traversalFlag = false;
                    }

                    cellIdLvl2[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                    tMaxLvl2[tMinDimension] += subCellSize[tMinDimension] * fabsf(toPtr(rayDirRCP)[tMinDimension]);

                    secondLvlFlag = secondLvlFlag &&
                        cellIdLvl2[tMinDimension] != cell[tMinDimension]
                    && cellIdLvl2[tMinDimension] != -1;
                    //////////////////////////////////////////////////////////////////////////
                }
            }//end traversal inner loop

            if(traversalFlag)
            {
                /////////////////////////////////////////////////////////////////////////
                //Traverse to next cell
                const int tMinDimension =
                    MIN_DIMENSION(tMax[0], tMax[1], tMax[2]);

                traversalFlag = traversalFlag && rayT > tMax[tMinDimension];

                tEntry = tMax[tMinDimension];
                cellId[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                tMax[tMinDimension] += toPtr(dcGrid.getCellSize())[tMinDimension] * 
                    fabsf(toPtr(rayDirRCP)[tMinDimension]);

                traversalFlag = traversalFlag &&
                    cellId[tMinDimension] != dcGrid.res[tMinDimension]
                && cellId[tMinDimension] != -1;
            }
        }
        //end traversal outer loop
        ////////////////////////////////////////////////////////////////////////////
    }
};


#undef  MIN_DIMENSION

#endif // TLGRIDTRAVERSER_H_5A1BB847_16FB_4C5F_B509_73260E4D0F34

