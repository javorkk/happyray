#ifdef _MSC_VER
#pragma once
#endif

#ifndef UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB
#define UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB

#include "CUDAStdAfx.h"
#include "Textures.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"


class UniformGrid : public Primitive<2>
{
public:
    //float3 vtx[2]; //inherited -> bounding box
    int res[3];
    float3 cellSize;
    float3 cellSizeRCP;
    cudaPitchedPtr cells;
    uint* primitives;

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

    DEVICE uint2 getCell(int aIdX, int aIdY, int aIdZ)
    {
        return *((uint2*)((char*)cells.ptr
            + aIdY * cells.pitch + aIdZ * cells.pitch * cells.ysize) + aIdX);
        //return tex3D(texGridCells, aIdX, 
        //             aIdY,  aIdZ);
    } 

    DEVICE uint getPrimitiveId(uint aId)
    {
        return primitives[aId];
    }
};

template<>
class BBoxExtractor<UniformGrid>
{
public:
    DEVICE HOST static BBox get(const UniformGrid& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

#endif // UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB
