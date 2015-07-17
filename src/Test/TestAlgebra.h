#ifdef _MSC_VER
#pragma once
#endif

#ifndef TESTALGEBRA_H_INCLUDED_FD8D3A12_EA2A_4DE1_BAD1_A9303EFA309C
#define TESTALGEBRA_H_INCLUDED_FD8D3A12_EA2A_4DE1_BAD1_A9303EFA309C

#include "Core/Algebra.hpp"

class TestQuaternions
{
    void initRay(float3& aOrg, float3& aDirRCP) const;

    bool checkTransformations(
        float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32,

        float n00, float n10, float n20, float n30,
        float n01, float n11, float n21, float n31,
        float n02, float n12, float n22, float n32 ) const;

    bool checkRay(const float3& aRayOrg1, const float3& aRayDirRCP1,
        const float3& aRayOrg2, const float3& aRayDirRCP2) const;
public:

    bool run() const;
};

#endif // TESTALGEBRA_H_INCLUDED_FD8D3A12_EA2A_4DE1_BAD1_A9303EFA309C
