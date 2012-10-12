#ifdef _MSC_VER
#pragma once
#endif

#ifndef PLANE_HPP_INCLUDED_A720B928_8295_4F54_82C8_A3CBB7498A86
#define PLANE_HPP_INCLUDED_A720B928_8295_4F54_82C8_A3CBB7498A86


#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"

class InfPlane : public Primitive<2>
{
public:
    //float3 vtx[2]; //inherited, vtx[0] -> point, vtx[1] -> normal
    DEVICE HOST float3 getPt() const { return vtx[0]; }
    DEVICE HOST float3& getPt() { return vtx[0]; }
    DEVICE HOST float3 getN() const { return vtx[1]; }
    DEVICE HOST float3& getN() { return vtx[1]; }
    DEVICE HOST float calculateSignedDistance( const float3 &aPt ) const
    {
        return dot(aPt,getN()) - dot(getPt(),getN());
    }
    DEVICE HOST bool equal( const InfPlane &aPlane, float aAngleEPS, float aSpatialEPS ) const
    {
        return dot(getN(),aPlane.getN()) >= 1.0f - aAngleEPS &&
            fabsf(calculateSignedDistance(aPlane.getPt())) <= aSpatialEPS;
    }
    DEVICE HOST float3 projectedPoint( const float3& aPt)
    {
        return aPt - calculateSignedDistance(aPt) * getN();
    }
};


template<>
class BBoxExtractor<InfPlane>
{
public:
    DEVICE HOST static BBox get(const InfPlane& aPlane)
    {
        return BBox::empty();
    }
};
#endif // PLANE_HPP_INCLUDED_A720B928_8295_4F54_82C8_A3CBB7498A86
