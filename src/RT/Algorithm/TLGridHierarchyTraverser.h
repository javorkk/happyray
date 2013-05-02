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

template<class tPrimitive, class tIntersector, bool taIsShadowRay = false>
class GeometryInstanceIntersector
{
public:
    DEVICE void operator()(
        float3&                             aRayOrg,
        float&                              oRayT,
        uint&                               oBestHit,
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

        for (uint it = aIdRange.x; it < aIdRange.y; ++ it)
        {
            const GeometryInstance instance = aInstances[aInstanceIndirection[it]];
            BBox bounds = BBoxExtractor<GeometryInstance>::get(instance);

            float tEntry;
            float tExit;
            float3& aRayDirRCP = (((float3*)aSharedMemory)[threadId1D32()]);

            bounds.fastClip(aRayOrg, aRayDirRCP, tEntry, tExit);

            if(!(tExit > tEntry && tExit >= 0.f && tEntry < oRayT))
                continue;

            float3 rayOrgT = instance.rotation0 * aRayOrg.x + instance.rotation1 * aRayOrg.y + 
                instance.rotation2 * aRayOrg.z + instance.translation;
            
            float3 rayDirT = instance.rotation0 / aRayDirRCP.x + instance.rotation1 / aRayDirRCP.y + 
                instance.rotation2 / aRayDirRCP.z;

            float3 rayDirRCPtmp;//backup ray direction
            rayDirRCPtmp.x = aRayDirRCP.x;
            rayDirRCPtmp.y = aRayDirRCP.y;
            rayDirRCPtmp.z = aRayDirRCP.z;
            
            aRayDirRCP.x = 1.f / rayDirT.x;
            aRayDirRCP.y = 1.f / rayDirT.y;
            aRayDirRCP.z = 1.f / rayDirT.z;

            UniformGrid grid = aGrids[instance.index];
            UGridTraverser<tPrimitive, tIntersector, taIsShadowRay> traverse;
            bool traversalFlag = true;
            traverse(aPrimitiveArray, grid, rayOrgT, aRayDirRCP, oRayT, oBestHit, traversalFlag, aSharedMemory);

            //restore ray direction
            aRayDirRCP.x = rayDirRCPtmp.x;
            aRayDirRCP.y = rayDirRCPtmp.y;
            aRayDirRCP.z = rayDirRCPtmp.z;

        }//end for all intersection candidates
    }//end operator()
};

#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

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
        GeometryInstanceIntersector< tPrimitive, tIntersector, taIsShadowRay > intersector;
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
            const float3 t1 = (dcGrid.vtx[0] - rayOrg) * rayDirRCP;
            float3 tFar = (dcGrid.vtx[1] - rayOrg) * rayDirRCP;

            const float3 tNear = min(t1, tFar);
            tFar = max(t1, tFar);

            const float tEntry = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
            const float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);

            //BBox bounds = BBoxExtractor<UniformGrid>::get(dcGrid);
            //bounds.fastClip(rayOrg, rayDirRCP, tEntry, tExit);

            traversalFlag = traversalFlag && (tExit > tEntry && tExit >= 0.f);
            //end ray/box intersection test
            //////////////////////////////////////////////////////////////////////////

            const float3 entryPt = (tEntry >= 0.f) ?
                rayOrg + rep(tEntry + EPS) / rayDirRCP : rayOrg;

            float3 cellIdf = 
                (entryPt - dcGrid.vtx[0]) * dcGrid.getCellSizeRCP();

            cellIdf.x = floorf(cellIdf.x);
            cellIdf.y = floorf(cellIdf.y);
            cellIdf.z = floorf(cellIdf.z);

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
                (cellId[0] != ((rayDirRCP.x > 0.f) ? dcGrid.res[0] : -1)) 
                &&  (cellId[1] != ((rayDirRCP.y > 0.f) ? dcGrid.res[1] : -1))
                &&  (cellId[2] != ((rayDirRCP.z > 0.f) ? dcGrid.res[2] : -1)) 
                );
        }
        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                cellRange = //make_uint2(0u, 0u);
                    dcGrid.getCell(cellId[0], cellId[1], cellId[2]);
                //cellRange =  make_uint2(0u, 0u);
            }

            intersector(rayOrg, rayT, bestHit, cellRange,
                dcGrid.instanceIndices, dcGrid.instances, dcGrid.grids, dcGrid.primitives,
                aPrimitiveArray, sharedMemNew);

            if (traversalFlag)
            {

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

                traversalFlag = traversalFlag &&
                    cellId[tMinDimension] != dcGrid.res[tMinDimension]
                && cellId[tMinDimension] != -1;
                //////////////////////////////////////////////////////////////////////////
            }
        }
        //end traversal loop
        //////////////////////////////////////////////////////////////////////////

    }
};

#undef  MIN_DIMENSION

#endif // TLGRIDHIERARCHYTRAVERSER_H_INCLUDED_0B0B4785_A48A_499A_A074_914B2408323E
