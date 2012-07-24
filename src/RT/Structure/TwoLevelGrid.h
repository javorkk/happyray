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

#ifndef TWOLEVELGRID_H_INCLUDED_7F0D3620_3404_471A_80CE_520434C43A5E
#define TWOLEVELGRID_H_INCLUDED_7F0D3620_3404_471A_80CE_520434C43A5E


#include "CUDAStdAfx.h"
#include "Textures.h"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"


struct TwoLevelGridCell
{
    typedef uint2                   t_data;

    //////////////////////////////////////////////////////////////////////////
    //8 bytes:
    //1st, 2nd, 3rd and 4th - begin of leaf array
    //5th                   - dummy | isEmpty flag | isValid flag | isLeaf flag
    //6th, 7th and 8th      - resolutions in z, y and x
    //////////////////////////////////////////////////////////////////////////
    t_data  data;

    HOST DEVICE TwoLevelGridCell()
    {
        data.x = 0u;
        data.y = 0u;
    }

    HOST DEVICE TwoLevelGridCell(t_data aData): data(aData)
    {}

    enum{
        SHIFTX              =   0,
        SHIFTY              =   8,
        SHIFTZ              =   16,
        SHIFTW              =   24,
        VALIDMASK           =   0x20,
        NOTEMPTYMASK        =   0x40,
        LEAFMASK            =   0x80,
        MASK                =   0xFF,
        LARGELEAFMASK       =   0x10000000,
        LARGEVALIDMASK      =   0x20000000,
        LARGENOTEMPTYMASK   =   0x40000000,
        LARGENOTEMPTYMASKNEG=   0x3FFFFFFF,
        LARGEVALIDMASKNEG   =   0x5FFFFFFF,
        LARGELEAFMASKNEG    =   0x6FFFFFFF,
    };

        HOST DEVICE uint notEmpty() const
        {
            return data.y & LARGENOTEMPTYMASK;
        }

        HOST DEVICE uint operator [] (const uint aId) const
        {
            return (data.y >> (aId * 8)) & MASK;
        }

        HOST DEVICE void clear()
        {
            data.y = 0u;
        }

        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setX(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTX;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setY(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTY;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setZ(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTZ;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setW(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTW;
        }

        HOST DEVICE static uint get(const uint aId, const uint aVal)
        {
            return (aVal >> (aId * 8)) & MASK;
        }

        HOST DEVICE void setEmpty()
        {
            data.y &= LARGENOTEMPTYMASKNEG;
        }

        HOST DEVICE void setNotEmpty()
        {
            data.y |= LARGENOTEMPTYMASK;
        }

        HOST DEVICE void setValid()
        {
            data.y |= LARGEVALIDMASK;
        }

        HOST DEVICE void setNotValid()
        {
            data.y &= LARGEVALIDMASKNEG;
        }

        HOST DEVICE void setLeaf()
        {
            data.y |= LARGELEAFMASK;
        }

        HOST DEVICE void setNotLeaf()
        {
            data.y &= LARGELEAFMASKNEG;
        }

        HOST DEVICE uint isLeaf()
        {
            return data.y & LARGELEAFMASKNEG;
        }

        HOST DEVICE void setLeafRangeBegin(const uint aVal)
        {
            data.x = aVal;
        }

        HOST DEVICE uint getLeafRangeBegin() const
        {
            return data.x;
        }
};

class TwoLevelGrid : public Primitive<2>
{
public:
    typedef uint2                           t_Leaf;
    typedef TwoLevelGridCell                t_Cell;

    //float3        vtx[2]; //inherited -> bounding box
    int             res[3];
    float3          cellSize;
    float3          cellSizeRCP;
    cudaPitchedPtr  cells;
    t_Leaf*         leaves;
    uint*           primitives;
    //uint            numPrimitiveReferences;

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

    DEVICE uint getPrimitiveId(uint aId)
    {
        return primitives[aId];
    }
};

template<>
class BBoxExtractor<TwoLevelGrid>
{
public:
    DEVICE HOST static BBox get(const TwoLevelGrid& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

#endif // TWOLEVELGRID_H_INCLUDED_7F0D3620_3404_471A_80CE_520434C43A5E
