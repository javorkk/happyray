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

#ifndef TLGRIDHIERARCHYTRAVERSER_H_INCLUDED_0B0B4785_A48A_499A_A074_914B2408323E
#define TLGRIDHIERARCHYTRAVERSER_H_INCLUDED_0B0B4785_A48A_499A_A074_914B2408323E


#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

#include "RT/Structure/TwoLevelGridHierarchy.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Algorithm/UGridTraverser.h"

/////////////////////////////////////////////////////////////////
//Two Level Grid Hierarchy Traversal Classes
/////////////////////////////////////////////////////////////////
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

template<class tPrimitive, class tIntersector, bool taIsShadowRay = false>
class GeometryInstanceTraverser
{
public:
    DEVICE void operator()(
        float3&                             aRayOrg,
        float&                              oRayT,
        uint&                               oBestHit,
        uint&                               oBestHitInstance,
        const uint2&                        aIdRange,
        const uint*                         aInstanceIndirection,
        GeometryInstance*                   aInstances,
        UniformGrid*                        aGrids,
        uint*                               aPrimitiveIndirection,
        PrimitiveArray<tPrimitive>&         aPrimitiveArray,
        uint*                               aSharedMemory
        ) const
    {
        //UniformGrid grid = aGrids[0];
        //float3& aRayDirRCP = (((float3*)aSharedMemory)[threadId1D32()]);
        //UGridTraverser<tPrimitive, tIntersector, taIsShadowRay> traverse;
        //bool traversalFlag = true;
        //traverse(aPrimitiveArray, grid, aRayOrg, aRayDirRCP, oRayT, oBestHit, traversalFlag, aSharedMemory);

        //grid = aGrids[1];
        //traversalFlag = true;
        //traverse(aPrimitiveArray, grid, aRayOrg, aRayDirRCP, oRayT, oBestHit, traversalFlag, aSharedMemory);
        
        float3& aRayDirRCP = (((float3*)aSharedMemory)[threadId1D32()]);
        float3 rayDirRCPtmp;//backup ray direction
        rayDirRCPtmp.x = aRayDirRCP.x;
        rayDirRCPtmp.y = aRayDirRCP.y;
        rayDirRCPtmp.z = aRayDirRCP.z;

        for (uint it = aIdRange.x; it < aIdRange.y; ++ it)
        {
            const GeometryInstance instance = aInstances[aInstanceIndirection[it]];
            BBox boundsLvl1 = BBoxExtractor<GeometryInstance>::get(instance);
            //if(bounds.vtx[0].x > bounds.vtx[1].x || bounds.vtx[0].y > bounds.vtx[1].y || bounds.vtx[0].z > bounds.vtx[1].z)
            //    continue;

            float tEntryLvl1;
            float tExitLvl1;

            boundsLvl1.fastClip(aRayOrg, aRayDirRCP, tEntryLvl1, tExitLvl1);

            if(!(tExitLvl1 > tEntryLvl1 && tExitLvl1 >= 0.f && tEntryLvl1 < oRayT))
                continue;

            
            float3 rayOrgT = instance.transformRay(aRayOrg, aRayDirRCP);

            UniformGrid grid = aGrids[instance.index];
            bool traversalFlag = true;
            uint bestHitNew = (uint)-1;
            float rayT = oRayT;

            tIntersector intersector;
            //////////////////////////////////////////////////////////////////////////
            //Traversal State
            float tMax[3];
            int cellId[3];
            //////////////////////////////////////////////////////////////////////////

            
            //////////////////////////////////////////////////////////////////////////
            //ray/box intersection test
            //const float3 t1 = (aGrid.vtx[0] - rayOrg) * rayDirRCP;
            //float3 tFar = (aGrid.vtx[1] - rayOrg) * rayDirRCP;

            //const float3 tNear = min(t1, tFar);
            //tFar = max(t1, tFar);

            //const float tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            //const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);
            float tEntry, tExit;
            BBox bounds = BBoxExtractor<UniformGrid>::get(grid);
            bounds.fastClip(rayOrgT, aRayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f && tEntry < rayT);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////

            const float3 entryPt = (tEntry >= 0.f) ? rayOrgT + rep(tEntry + EPS) / aRayDirRCP : rayOrgT;

            float3 cellIdf =
                (entryPt - grid.vtx[0]) * grid.getCellSizeRCP();

            cellIdf.x = floorf(cellIdf.x);
            cellIdf.y = floorf(cellIdf.y);
            cellIdf.z = floorf(cellIdf.z);

            cellIdf.x = (aRayDirRCP.x > 0.f) ? min((float)grid.res[0] - 1.f, cellIdf.x) : max(cellIdf.x, 0.f);
            cellIdf.y = (aRayDirRCP.y > 0.f) ? min((float)grid.res[1] - 1.f, cellIdf.y) : max(cellIdf.y, 0.f);
            cellIdf.z = (aRayDirRCP.z > 0.f) ? min((float)grid.res[2] - 1.f, cellIdf.z) : max(cellIdf.z, 0.f);


            float3 tmp;
            tmp.x = (aRayDirRCP.x > 0.f) ? 1.f : 0.f;
            tmp.y = (aRayDirRCP.y > 0.f) ? 1.f : 0.f;
            tmp.z = (aRayDirRCP.z > 0.f) ? 1.f : 0.f;

            tMax[0] = ((cellIdf.x + tmp.x) * grid.getCellSize().x + grid.vtx[0].x - rayOrgT.x) * aRayDirRCP.x;
            tMax[1] = ((cellIdf.y + tmp.y) * grid.getCellSize().y + grid.vtx[0].y - rayOrgT.y) * aRayDirRCP.y;
            tMax[2] = ((cellIdf.z + tmp.z) * grid.getCellSize().z + grid.vtx[0].z - rayOrgT.z) * aRayDirRCP.z;

            cellId[0] = static_cast<int>(cellIdf.x);
            cellId[1] = static_cast<int>(cellIdf.y);
            cellId[2] = static_cast<int>(cellIdf.z);

            if (toPtr(aRayDirRCP)[0] > 0.f)
                traversalFlag = traversalFlag && cellId[0] < grid.res[0];
            else
                traversalFlag = traversalFlag && cellId[0] > -1;

            if (toPtr(aRayDirRCP)[1] > 0.f)
                traversalFlag = traversalFlag && cellId[1] < grid.res[1];
            else
                traversalFlag = traversalFlag && cellId[1] > -1;

            if (toPtr(aRayDirRCP)[2] > 0.f)
                traversalFlag = traversalFlag && cellId[2] < grid.res[2];
            else
                traversalFlag = traversalFlag && cellId[2] > -1;

            //traversalFlag = traversalFlag && (  
            //    (cellId[0] < grid.res[0] && cellId[0] > -1) && 
            //    (cellId[1] < grid.res[1] && cellId[1] > -1) && 
            //    (cellId[2] < grid.res[2] && cellId[2] > -1) 
            //    );

            //////////////////////////////////////////////////////////////////////////
            //Traversal loop
            while (traversalFlag)
            {
                uint2 cellRange = make_uint2(0u, 0u);
                if( cellId[0] < grid.res[0] && cellId[0] > -1 &&
                    cellId[1] < grid.res[1] && cellId[1] > -1 &&
                    cellId[2] < grid.res[2] && cellId[2] > -1)
                {
                    cellRange = grid.getCell(cellId[0], cellId[1], cellId[2]);
                }
                intersector(rayOrgT, aRayDirRCP, rayT, bestHitNew,
                    cellRange, grid.primitives, aPrimitiveArray, aSharedMemory);


                /////////////////////////////////////////////////////////////////////////
                //Traverse to next cell
                const int tMinDimension =
                    MIN_DIMENSION(tMax[0], tMax[1], tMax[2]);

                traversalFlag = traversalFlag && rayT > tMax[tMinDimension];

                cellId[tMinDimension] += (toPtr(aRayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
                tMax[tMinDimension] += toPtr(grid.getCellSize())[tMinDimension] * 
                    fabsf(toPtr(aRayDirRCP)[tMinDimension]);

                if (toPtr(aRayDirRCP)[tMinDimension] > 0.f)
                    traversalFlag = traversalFlag && cellId[tMinDimension] < grid.res[tMinDimension];
                else
                    traversalFlag = traversalFlag && cellId[tMinDimension] > -1;


                //traversalFlag = traversalFlag &&
                //    cellId[tMinDimension] < grid.res[tMinDimension]
                //&& cellId[tMinDimension] > -1;
                //////////////////////////////////////////////////////////////////////////

            }
            //end traversal loop
            //////////////////////////////////////////////////////////////////////////

            if (bestHitNew != (uint)-1 && rayT < oRayT && rayT >= tEntryLvl1 && rayT < tExitLvl1)
            {
                oRayT = rayT;
                oBestHitInstance = aInstanceIndirection[it];
                oBestHit = bestHitNew;
            }
            //restore ray direction
            aRayDirRCP.x = rayDirRCPtmp.x;
            aRayDirRCP.y = rayDirRCPtmp.y;
            aRayDirRCP.z = rayDirRCPtmp.z;

        }//end for all intersection candidates
    }//end operator()
};

template<class tPrimitive, class tIntersector, bool taIsShadowRay = false>
class TLGridHierarchyTraverser
{
public:
    DEVICE void operator()(
        PrimitiveArray<tPrimitive>  aPrimitiveArray,
        TwoLevelGridHierarchy       dcGrid,
        float3&                     rayOrg,
        float3&                     rayDirRCP,
        float&                      rayT,
        uint&                       bestHit,
        bool                        traversalFlag,
        uint*                       sharedMemNew
        )
    {
        GeometryInstanceTraverser< tPrimitive, tIntersector, taIsShadowRay > traverser;
        uint bestHitInstance = (uint)-1;
        //////////////////////////////////////////////////////////////////////////
        //Traversal State
        float tMax[3];
        int cellId[3];
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
            //const float3 t1 = (dcGrid.vtx[0] - rayOrg) * rayDirRCP;
            //float3 tFar = (dcGrid.vtx[1] - rayOrg) * rayDirRCP;

            //const float3 tNear = min(t1, tFar);
            //tFar = max(t1, tFar);

            //const float tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            //const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);

            float tEntry;
            float tExit;
            BBox bounds = BBoxExtractor<TwoLevelGridHierarchy>::get(dcGrid);
            bounds.fastClip(rayOrg, rayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////

            const float3 entryPt = (tEntry >= 0.f) ? rayOrg + rep(tEntry + EPS) / rayDirRCP : rayOrg;

            float3 cellIdf = 
                (entryPt - dcGrid.vtx[0]) * dcGrid.getCellSizeRCP();

            cellIdf.x = floorf(cellIdf.x);
            cellIdf.y = floorf(cellIdf.y);
            cellIdf.z = floorf(cellIdf.z);

            //cellIdf.x = (rayDirRCP.x > 0.f) ? min((float)dcGrid.res[0] - 1.f, cellIdf.x) : max(cellIdf.x, 0.f);
            //cellIdf.y = (rayDirRCP.y > 0.f) ? min((float)dcGrid.res[1] - 1.f, cellIdf.y) : max(cellIdf.y, 0.f);
            //cellIdf.z = (rayDirRCP.z > 0.f) ? min((float)dcGrid.res[2] - 1.f, cellIdf.z) : max(cellIdf.z, 0.f);

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
            
            if (toPtr(rayDirRCP)[0] > 0.f)
                traversalFlag = traversalFlag && cellId[0] < dcGrid.res[0];
            else
                traversalFlag = traversalFlag && cellId[0] > -1;

            if (toPtr(rayDirRCP)[1] > 0.f)
                traversalFlag = traversalFlag && cellId[1] < dcGrid.res[1];
            else
                traversalFlag = traversalFlag && cellId[1] > -1;

            if (toPtr(rayDirRCP)[2] > 0.f)
                traversalFlag = traversalFlag && cellId[2] < dcGrid.res[2];
            else
                traversalFlag = traversalFlag && cellId[2] > -1;

            //traversalFlag = traversalFlag && (  
            //    ((rayDirRCP.x > 0.f) ? cellId[0] < dcGrid.res[0] : cellId[0] > -1) && 
            //    ((rayDirRCP.y > 0.f) ? cellId[1] < dcGrid.res[1] : cellId[1] > -1) && 
            //    ((rayDirRCP.z > 0.f) ? cellId[2] < dcGrid.res[2] : cellId[2] > -1) 
            //    );
            //if(traversalFlagOld && ! traversalFlag )
            //{
            //    printf("cell id %d %d %d grid res %d %d %d", cellId[0], cellId[1], cellId[2],
            //        dcGrid.res[0], dcGrid.res[1], dcGrid.res[2]);
            //}
        }
        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (traversalFlag)
        {
            uint2 cellRange = make_uint2(0u, 0u);
            if(cellId[0] < dcGrid.res[0] && cellId[0] > -1 &&
               cellId[1] < dcGrid.res[1] && cellId[1] > -1 &&
               cellId[2] < dcGrid.res[2] && cellId[2] > -1)
            {
                cellRange = dcGrid.getCell(cellId[0], cellId[1], cellId[2]);
            }

            traverser(rayOrg, rayT, bestHit, bestHitInstance, cellRange,
                dcGrid.getInstanceIndices(), dcGrid.getInstances(), dcGrid.getGrids(), dcGrid.primitives,
                aPrimitiveArray, sharedMemNew);

            /////////////////////////////////////////////////////////////////////////
            //Traverse to next cell
            const int tMinDimension =
                MIN_DIMENSION(tMax[0], tMax[1], tMax[2]);

            if(taIsShadowRay)
            {
                if(rayT < 0.9999f)
                {
                    traversalFlag = false;
                }
            }

            traversalFlag = traversalFlag && rayT > tMax[tMinDimension];

            cellId[tMinDimension] += (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? 1 : -1;
            tMax[tMinDimension] += toPtr(dcGrid.getCellSize())[tMinDimension] * 
                fabsf(toPtr(rayDirRCP)[tMinDimension]);

            if (toPtr(rayDirRCP)[tMinDimension] > 0.f)
            {
                traversalFlag = traversalFlag && cellId[tMinDimension] < dcGrid.res[tMinDimension];
            }
            else
            {
                traversalFlag = traversalFlag && cellId[tMinDimension] > -1;
            }
            //traversalFlag = traversalFlag &&
            //    (toPtr(rayDirRCP)[tMinDimension] > 0.f) ? cellId[tMinDimension] < dcGrid.res[tMinDimension] : cellId[tMinDimension] > -1;
            //////////////////////////////////////////////////////////////////////////

        }
        //end traversal loop
        //////////////////////////////////////////////////////////////////////////
        
        //transform the ray into the local coordinates of the hit instance
        if(bestHitInstance != (uint)-1)
        {
            const GeometryInstance instance = dcGrid.getInstances()[bestHitInstance];
            float3 rayOrgT = instance.transformRay(rayOrg, rayDirRCP);
            rayOrg = rayOrgT;
        }
    }
};

#undef  MIN_DIMENSION

#endif // TLGRIDHIERARCHYTRAVERSER_H_INCLUDED_0B0B4785_A48A_499A_A074_914B2408323E
