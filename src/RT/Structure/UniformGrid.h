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

#ifndef UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB
#define UNIFORMGRID_H_C8280ED1_0974_408A_BD7C_9A509CA1C1DB

#include "CUDAStdAfx.h"
//#include "Textures.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"


class UniformGrid : public Primitive<2>
{
public:
    //float3 vtx[2]; //inherited -> bounding box
    int res[3];
    //float3 cellSize;
    //float3 cellSizeRCP;
    cudaPitchedPtr cells;
    uint* primitives;
    //uint  numPrimitiveReferences;

    HOST DEVICE const float3 getResolution() const
    {
        float3 retval;
        retval.x = static_cast<float>(res[0]);
        retval.y = static_cast<float>(res[1]);
        retval.z = static_cast<float>(res[2]);
        return retval;
    }


    HOST DEVICE float3 getCellSize() const
    {
        //return cellSize;
        return fastDivide(vtx[1] - vtx[0], getResolution());
    }

    HOST DEVICE void setCellSize(const float3& aCellSize)
    {
        //set the variable if it exists
        //cellSize = aCellSize;
        //...do nothing
    }

    HOST DEVICE float3 getCellSizeRCP() const
    {
        //return cellSizeRCP;
        return fastDivide(getResolution(), vtx[1] - vtx[0]);
    }

    HOST DEVICE void setCellSizeRCP(const float3& aCellSizeRCP)
    {
        //set the variable if it exits
        //cellSizeRCP = aCellSizeRCP;
        //...or do nothing
    }

    //convert a 3D cell index into a linear one
    HOST DEVICE int getCellIdLinear(int aIdX, int aIdY, int aIdZ) const
    {
        return aIdX + aIdY * res[0] + aIdZ * res[0] * res[1];
    }

    //convert a 3D cell index into a linear one
    HOST DEVICE int3 getCellId3D(int aLinearId) const
    {
        return make_int3(
            aLinearId % res[0],
            (aLinearId % (res[0] * res[1])) / res[0],
            aLinearId / (res[0] * res[1]) );
    }

    HOST DEVICE uint2 getCell(int aIdX, int aIdY, int aIdZ) const
    {
        return *((uint2*)((char*)cells.ptr
            + aIdY * cells.pitch + aIdZ * cells.pitch * cells.ysize) + aIdX);
        //return tex3D(texGridCells, aIdX, aIdY,  aIdZ);
    } 

    HOST DEVICE void setCell(int aIdX, int aIdY, int aIdZ, uint2 aVal)
    {
        *((uint2*)((char*)cells.ptr
            + aIdY * cells.pitch + aIdZ * cells.pitch * cells.ysize) + aIdX) = aVal;
    } 

    HOST DEVICE int3 getCellIdAt(float3 aPosition) const
    {
        float3 cellIdf = (aPosition - vtx[0]) * getCellSizeRCP();
        int3 cellId;
        cellId.x = static_cast<int>(cellIdf.x);
        cellId.y = static_cast<int>(cellIdf.y);
        cellId.z = static_cast<int>(cellIdf.z);
        return cellId;
    }

    HOST DEVICE uint2 getCellAt(float3 aPosition) const
    {
        float3 cellIdf = (aPosition - vtx[0]) * getCellSizeRCP();
        return getCell(static_cast<int>(cellIdf.x),  static_cast<int>(cellIdf.y), static_cast<int>(cellIdf.z));
    }

    HOST DEVICE float3 getCellCenter(int aIdX, int aIdY, int aIdZ) const
    {
        float3 cellIdf = make_float3((float)aIdX + 0.5f, (float)aIdY + 0.5f, (float)aIdZ + 0.5f);
        return vtx[0] + cellIdf * getCellSize();
    }

    HOST DEVICE uint getPrimitiveId(uint aId) const
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
