#ifdef _MSC_VER
#pragma once
#endif

#ifndef RAYTRIANGLEINTERSECTOR_H_81C76214_B137_402A_8609_0234EC809946
#define RAYTRIANGLEINTERSECTOR_H_81C76214_B137_402A_8609_0234EC809946

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Structure/PrimitiveArray.h"

class MollerTrumboreIntersectionTest
{
public:
    DEVICE void operator()(
        const float3&                       aRayOrg,
        const float3&                       aRayDirRCP,
        float&                              oRayT,
        uint&                               oBestHit,
        const uint2&                        aIdRange,
        const uint*                         aIndexIndirection,
        const PrimitiveArray<Triangle>&     aTriangleArray,
        uint*                               aDummy
) const
    {
        for (uint it = aIdRange.x; it != aIdRange.y; ++ it)
        {
            Triangle tri = aTriangleArray[aIndexIndirection[it]];
            float3& org   = tri.vtx[0];
            float3& edge1 = tri.vtx[1];
            float3& edge2 = tri.vtx[2];

            edge1 = edge1 - org;
            edge2 = edge2 - org;

            float3 rayDir;
            rayDir = fastDivide(rep(1.f), aRayDirRCP);


            float3 pvec      = rayDir % edge2;
            float detRCP    = 1.f / dot(edge1, pvec);

            //if(fabsf(detRCP) <= EPS_RCP) continue;

            float3 tvec  = aRayOrg - org;
            float alpha = detRCP * dot(tvec, pvec);

            //if(alpha < 0.f || alpha > 1.f) continue;

            tvec        = tvec % edge1;
            float beta  = detRCP * dot(tvec, rayDir);

            //if(beta < 0.f || beta + alpha > 1.f) continue;

            float dist  = detRCP * dot(edge2, tvec);

            if (alpha >= 0.f        &&
                beta >= 0.f         &&
                alpha + beta <= 1.f &&
                dist > EPS          &&
                dist < oRayT)
            {
                oRayT  = dist;
                oBestHit = it;
            }
        }//end for all intersection candidates
    }//end operator()
};

#endif // RAYTRIANGLEINTERSECTOR_H_81C76214_B137_402A_8609_0234EC809946
