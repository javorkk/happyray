#ifdef _MSC_VER
#pragma once
#endif

#ifndef TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683
#define TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683

#include "CUDAStdAfx.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/UniformGrid.h"



struct  GeometryInstance : public Primitive<2>
{
    //float3 vtx[2]; //inherited -> bounding box
    uint index;
    //Transformation
    //float3 rotation0, rotation1, rotation2, translation;
    float3 irotation0, irotation1, irotation2, itranslation;

    //returns the new origin, overwrites the old direction
    DEVICE HOST float3 transformRay(const float3 aRayOrg, float3& oRayDirRCP) const
    {
        float3 rayOrgT = aRayOrg + itranslation;
        rayOrgT = irotation0 * aRayOrg.x + irotation1 * aRayOrg.y + irotation2 * aRayOrg.z + itranslation;

        float3 rayDirT = irotation0 / oRayDirRCP.x + irotation1 / oRayDirRCP.y + 
            irotation2 / oRayDirRCP.z;

        oRayDirRCP.x = 1.f / rayDirT.x;
        oRayDirRCP.y = 1.f / rayDirT.y;
        oRayDirRCP.z = 1.f / rayDirT.z;

        return rayOrgT;
    }

};

template<>
class BBoxExtractor<GeometryInstance>
{
public:
    DEVICE HOST static BBox get(const GeometryInstance& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

class TwoLevelGridHierarchy : public UniformGrid
{
public:
    typedef uint2                      t_Cell;
    typedef uint2                      t_Leaf;

    //float3 vtx[2];            //inherited -> bounding box
    //int res[3];               //inherited -> uniform grid
    //float3 cellSize;          //inherited -> uniform grid
    //float3 cellSizeRCP;       //inherited -> uniform grid
    //cudaPitchedPtr  cells;    //inherited -> uniform grid
    //uint* primitives;         //inherited -> uniform grid
    uint*           instanceIndices;
    GeometryInstance* instances;
    UniformGrid*    grids;
    t_Leaf*         leaves;

    uint            numInstances;
    //uint  numPrimitiveReferences;

    //////////////////////////////////////////////////////////////////////////
    //inherited -> uniform grid

    //HOST DEVICE const float3 getResolution() const; 

    //HOST DEVICE float3 getCellSize() const;

    //HOST DEVICE float3 getCellSizeRCP() const;

    //HOST DEVICE int getCellIdLinear(int aIdX, int aIdY, int aIdZ);

    //HOST DEVICE int3 getCellId3D(int aLinearId);

    //HOST DEVICE t_Cell getCell(int aIdX, int aIdY, int aIdZ);

    //HOST DEVICE void setCell(int aIdX, int aIdY, int aIdZ, uint2 aVal);

    //HOST DEVICE int3 getCellIdAt(float3 aPosition);

    //HOST DEVICE uint2 getCellAt(float3 aPosition);

    //HOST DEVICE float3 getCellCenter (int aIdX, int aIdY, int aIdZ) const;

    //HOST DEVICE uint getPrimitiveId(uint aId);

    //////////////////////////////////////////////////////////////////////////
};

template<>
class BBoxExtractor<TwoLevelGridHierarchy>
{
public:
    DEVICE HOST static BBox get(const TwoLevelGridHierarchy& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

#endif // TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683
