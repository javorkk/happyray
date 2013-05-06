#ifdef _MSC_VER
#pragma once
#endif

#ifndef TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683
#define TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683

#include "CUDAStdAfx.h"
#include "Textures.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/UniformGrid.h"

struct  GeometryInstance : public Primitive<2>
{
    //float3 vtx[2]; //inherited -> bounding box
    uint index;
    //Transformation
    float3 rotation0, rotation1, rotation2, translation;
    float3 irotation0, irotation1, irotation2, itranslation;

    //returns the new origin, overwrites the old direction
    DEVICE HOST float3 transformRay(float3& aRayOrg, float3& oRayDirRCP) const
    {
        float3 rayOrgT = rotation0 * aRayOrg.x + rotation1 * aRayOrg.y + 
            rotation2 * aRayOrg.z + translation;

        float3 rayDirT = rotation0 / oRayDirRCP.x + rotation1 / oRayDirRCP.y + 
            rotation2 / oRayDirRCP.z;

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

class TwoLevelGridHierarchy : public Primitive<2>
{
public:
    typedef uint2                      t_Cell;
    typedef uint2                      t_Leaf;

    //float3 vtx[2]; //inherited -> bounding box
    int res[3];
    float3 cellSize;
    float3 cellSizeRCP;
    cudaPitchedPtr  cells;
    uint*           instanceIndices;
    GeometryInstance* instances;
    UniformGrid*    grids;
    t_Leaf*         leaves;
    uint*           primitives;
    uint            numInstances;
    //uint  numPrimitiveReferences;


    DEVICE const float3 getResolution() const
    {
        float3 retval;
        retval.x = static_cast<float>(res[0]);
        retval.y = static_cast<float>(res[1]);
        retval.z = static_cast<float>(res[2]);
        return retval;
    }

    DEVICE float3 getCellSize() const
    {
        return cellSize;
        //return fastDivide(vtx[1] - vtx[0], getResolution());
    }

    DEVICE float3 getCellSizeRCP() const
    {
        return cellSizeRCP;
        //return fastDivide(getResolution(), vtx[1] - vtx[0]);
    }

    DEVICE t_Cell getCell(int aIdX, int aIdY, int aIdZ)
    {
        return *((t_Cell*)((char*)cells.ptr
            + aIdY * cells.pitch + aIdZ * cells.pitch * cells.ysize) + aIdX);
        //return tex3D(texGridCells, aIdX, aIdY,  aIdZ);
    } 

    DEVICE float3 getCellCenter (int aIdX, int aIdY, int aIdZ) const
    {
        float3 cellIdf = make_float3((float)aIdX + 0.5f, (float)aIdY + 0.5f, (float)aIdZ + 0.5f);
        return vtx[0] + cellIdf * cellSize;
    }

    DEVICE uint getPrimitiveId(uint aId)
    {
        return primitives[aId];
    }
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
