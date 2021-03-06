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

#ifndef BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
#define BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"


//An axis aligned bounding box
class BBox : public Primitive<2>
{
public:
    //float3 vtx[2]; //inherited


    //Returns the entry and exit distances of the ray with the
    //	bounding box.
    //If the first returned distance > the second, than
    //	the ray does not intersect the bounding box at all
    DEVICE HOST void clip(const float3 &aRayOrg, const float3& aRayDir, float& oEntry, float& oExit) const
    {
        const float3 t1 = (vtx[0] - aRayOrg) / aRayDir;
        float3 tMax = (vtx[1] - aRayOrg) / aRayDir;

        const float3 tMin = min(t1, tMax);
        tMax = max(t1, tMax);
#ifdef __CUDACC__
        oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
        oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
        oEntry = cudastd::max(cudastd::max(tMin.x, tMin.y), tMin.z);
        oExit = cudastd::min(cudastd::min(tMax.x, tMax.y), tMax.z);
#endif

    }

    DEVICE HOST void fastClip(const float3 &aRayOrg, const float3& aRayDirRCP, float& oEntry, float& oExit) const
    {
        const float3 t1 = (vtx[0] - aRayOrg) * aRayDirRCP;
        float3 tMax = (vtx[1] - aRayOrg) * aRayDirRCP;

        const float3 tMin = min(t1, tMax);
        tMax = max(t1, tMax);
#ifdef __CUDACC__
        oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
        oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
        oEntry = cudastd::max(cudastd::max(tMin.x, tMin.y), tMin.z);
        oExit = cudastd::min(cudastd::min(tMax.x, tMax.y), tMax.z);
#endif

    }

    //Extend the bounding box with a point
    DEVICE HOST void extend(const float3 &aPoint)
    {
        vtx[0] = min(vtx[0], aPoint);
        vtx[1] = max(vtx[1], aPoint);
    }

    //Extend the bounding box with another bounding box
    DEVICE HOST void extend(const BBox &aBBox)
    {
        vtx[0] = min(vtx[0], aBBox.vtx[0]);
        vtx[1] = max(vtx[1], aBBox.vtx[1]);
    }

    //Tighten the bounding box around another bounding box
    DEVICE HOST void tighten(const BBox &aBBox)
    {
        vtx[0] = max(vtx[0], aBBox.vtx[0]);
        vtx[1] = min(vtx[1], aBBox.vtx[1]);
    }

    //Tighten the bounding box around two points
    DEVICE HOST void tighten(const float3 &aMin, const float3 &aMax)
    {
        vtx[0] = max(vtx[0], aMin);
        vtx[1] = min(vtx[1], aMax);
    }


    //Returns an "empty" bounding box. Extending such a bounding
    //	box with a point will always create a bbox around the point
    //	and with a bbox - will simply copy the bbox.
    DEVICE HOST static BBox empty()
    {
        BBox ret;
        ret.vtx[0].x = FLT_MAX;
        ret.vtx[0].y = FLT_MAX;
        ret.vtx[0].z = FLT_MAX;
        ret.vtx[1].x = -FLT_MAX;
        ret.vtx[1].y = -FLT_MAX;
        ret.vtx[1].z = -FLT_MAX;
        return ret;
    }

    const float3 diagonal() const
    {
        return vtx[1] - vtx[0];
    }
};

template<class tPrimitive>
class BBoxExtractor
{
public:
    DEVICE HOST static BBox get(const tPrimitive& aPrimitive)
    {
        BBox result = BBox::empty();

//#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            result.extend(aPrimitive.vtx[i]);
        }

        return result;
    }
};


#endif // BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
