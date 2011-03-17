#include "StdAfx.hpp"
#include "RT/Primitive/Triangle.hpp"
#include "Core/Algebra.hpp"
#include "Core/SSEAlgebra.hpp"

using namespace cudastd;


ShevtsovTriAccel::ShevtsovTriAccel(Triangle aTriangle)
{
    float3 normal = aTriangle.vtx[0];
    //register reuse: state.tMax should be edge1
    float3 edge0  = aTriangle.vtx[1];
    //register reuse: state.cellId should be edge2
    float3 edge1  = aTriangle.vtx[2];

    edge0 -= normal;
    edge1 -= normal;

    normal = (edge0 % edge1);

#define MAX_DIMENSION(aX, aY, aZ)	                           \
    (aX > aY) ? ((aX > aZ) ? 0u : 2u)	: ((aY > aZ) ? 1u : 2u)

    dimW =
        MAX_DIMENSION(fabsf(toPtr(normal)[0]), fabsf(toPtr(normal)[1]), fabsf(toPtr(normal)[2]));

#undef  MAX_DIMENSION

    uint mod3[5] = {0,1,2,0,1};
    dimU = mod3[dimW + 1];
    dimV = mod3[dimW + 2];

    nu = toPtr(normal)[dimU] / toPtr(normal)[dimW];
    nv = toPtr(normal)[dimV] / toPtr(normal)[dimW];

    pu = toPtr(aTriangle.vtx[0])[dimU];
    pv = toPtr(aTriangle.vtx[0])[dimV];

    np = nu * toPtr(aTriangle.vtx[0])[dimU]
    + nv * toPtr(aTriangle.vtx[0])[dimV]
    + toPtr(aTriangle.vtx[0])[dimW];

    float minusOnePowW = (dimW == 1) ? 1.f : 1.f;
    e0u = minusOnePowW * toPtr(edge0)[dimU] / toPtr(normal)[dimW];
    e0v = minusOnePowW * toPtr(edge0)[dimV] / toPtr(normal)[dimW];
    e1u = minusOnePowW * toPtr(edge1)[dimU] / toPtr(normal)[dimW];
    e1v = minusOnePowW * toPtr(edge1)[dimV] / toPtr(normal)[dimW];

}

WoopTriAccel::WoopTriAccel(const Triangle& aTriangle)
{
    float3 edge1;
    float3 edge2;
    float3 normal;
    float3 vtx0;

    vtx0 = aTriangle.vtx[0];
    edge1 = aTriangle.vtx[1];
    edge2 = aTriangle.vtx[2];

    edge1 -= vtx0;
    edge2 -= vtx0;

    normal = (edge1 % edge2);

    float matrix[16];
    matrix[ 0] = normal.x;  matrix[ 1] = edge2.x;   matrix[ 2] = edge1.x;   matrix[ 3] = vtx0.x;
    matrix[ 4] = normal.y;  matrix[ 5] = edge2.y;   matrix[ 6] = edge1.y;   matrix[ 7] = vtx0.y; 
    matrix[ 8] = normal.z;  matrix[ 9] = edge2.z;   matrix[10] = edge1.z;   matrix[11] = vtx0.z;
    matrix[12] = 0.f;       matrix[13] = 0.f;       matrix[14] = 0.f;       matrix[15] = 1.f;

    InvertMatrix4x4()(matrix);

    memcpy((void*)data, (const void*)matrix, 12*sizeof(float));


}
