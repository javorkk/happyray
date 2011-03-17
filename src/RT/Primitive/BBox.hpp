#ifdef _MSC_VER
#pragma once
#endif

#ifndef BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
#define BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"

namespace cudastd
{

    //An axis aligned bounding box
    struct BBox
    {
        float3 min, max;


        //Returns the entry and exit distances of the ray with the
        //	bounding box.
        //If the first returned distance > the second, than
        //	the ray does not intersect the bounding box at all
        DEVICE HOST void clip(const float3 &aRayOrg, const float3& aRayDir, float& oEntry, float& oExit) const
        {
            const float3 t1 = (min - aRayOrg) / aRayDir;
            float3 tMax = (max - aRayOrg) / aRayDir;

            const float3 tMin = min(t1, tMax);
            tMax = max(t1, tMax);
#ifdef __CUDACC__
            oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
            oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
            oEntry = max(max(tMin.x, tMin.y), tMin.z);
            oExit = min(min(tMax.x, tMax.y), tMax.z);
#endif

        }

        DEVICE HOST void fastClip(const float3 &aRayOrg, const float3& aRayDirRCP, float& oEntry, float& oExit) const
        {
            const float3 t1 = (min - aRayOrg) * aRayDirRCP;
            float3 tMax = (max - aRayOrg) * aRayDirRCP;

            const float3 tMin = min(t1, tMax);
            tMax = max(t1, tMax);
#ifdef __CUDACC__
            oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
            oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
            oEntry = max(max(tMin.x, tMin.y), tMin.z);
            oExit = min(min(tMax.x, tMax.y), tMax.z);
#endif

        }

        //Extend the bounding box with a point
        DEVICE HOST void extend(const float3 &aPoint)
        {
            min = min(min, aPoint);
            max = max(max, aPoint);
        }

        //Extend the bounding box with another bounding box
        DEVICE HOST void extend(const BBox &aBBox)
        {
            min = min(min, aBBox.min);
            max = max(max, aBBox.max);
        }

        //Tighten the bounding box around another bounding box
        DEVICE HOST void tighten(const BBox &aBBox)
        {
            min = max(min, aBBox.min);
            max = min(max, aBBox.max);
        }

        //Tighten the bounding box around two points
        DEVICE HOST void tighten(const float3 &aMin, const float3 &aMax)
        {
            min = max(min, aMin);
            max = min(max, aMax);
        }


        //Returns an "empty" bounding box. Extending such a bounding
        //	box with a point will always create a bbox around the point
        //	and with a bbox - will simply copy the bbox.
        DEVICE HOST static BBox empty()
        {
            BBox ret;
            ret.min.x = FLT_MAX;
            ret.min.y = FLT_MAX;
            ret.min.z = FLT_MAX;
            ret.max.x = -FLT_MAX;
            ret.max.y = -FLT_MAX;
            ret.max.z = -FLT_MAX;
            return ret;
        }

        const float3 diagonal() const
        {
            return max - min;
        }
    };

    template<class Primitive>
    class BBoxExtractor
    {
    public:
        DEVICE HOST static BBox get(const Primitive& aPrimitive)
        {
            BBox result = BBox::empty();

#pragma unroll Primitive::NUM_VERTICES
            for(uint i = 0; i < Primitive::NUM_VERTICES; ++i)
            {
                result.extend(aPrimitive.vtx[i]);
            }

            return result;
        }
    }

};//namespace cudastd

#endif // BBOX_HPP_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
