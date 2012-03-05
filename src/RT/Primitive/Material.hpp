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

#ifndef MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8
#define MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"

#define INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY 10.f

class PhongMaterial
{
public:
    float4 diffuseReflectance; //.xyz -> diffuse reflectance .w -> index of refraction
    float4 specularReflectance; //.xyz -> specular reflectance .w -> specular exponent
    //float4 emission; //.xyz -> emission .w-> unused

    DEVICE HOST float3 getDiffuseReflectance(float xCoord = 0.f, float yCoord = 0.f, float zCoord = 0.f) const
    {
        float3 retval;
        retval.x = diffuseReflectance.x;
        retval.y = diffuseReflectance.y;
        retval.z = diffuseReflectance.z;
        return retval;
    }

    DEVICE HOST float3 getSpecularReflectance(float xCoord = 0.f, float yCoord = 0.f, float zCoord = 0.f) const
    {
        float3 retval;
        retval.x = specularReflectance.x;
        retval.y = specularReflectance.y;
        retval.z = specularReflectance.z;
        return retval;
    }

    //DEVICE HOST float3 getEmission() const
    //{
    //    float3 retval;
    //    retval.x = emission.x;
    //    retval.y = emission.y;
    //    retval.z = emission.z;
    //    return retval;
    //}

    DEVICE HOST float getSpecularExponent(float xCoord = 0.f, float yCoord = 0.f, float zCoord = 0.f) const
    {
        return specularReflectance.w;
    }

    DEVICE HOST bool isTransparent(float& oIndexOfRefraction) const
    {
        oIndexOfRefraction = diffuseReflectance.w -
            INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY;

        return oIndexOfRefraction > 0.f;
    }

};

class TexturedPhongMaterial : public PhongMaterial, public Primitive<2>
{
public:
    enum TexInterpolationMode
    {
        InterpolationModeNearestNeighbor    = 0,
        InterpolationModeTrilinear
    };

    TexInterpolationMode interpolationMode;
    //float3  vtx[2]; //inherited -> bounding box
    //float4 diffuseReflectance; //inherited, not used
    int     res[3];
    float3  cellSizeRCP;
    cudaPitchedPtr diffuseReflectanceData;
    

    TexturedPhongMaterial(): interpolationMode(InterpolationModeNearestNeighbor), cellSizeRCP(rep(1.f))
    {
        res[0] = 1;
        res[1] = 1;
        res[2] = 1;
    }

    DEVICE HOST float3 pointToCellId(float xCoord = 0.f, float yCoord = 0.f, float zCoord = 0.f) const
    {
        return (make_float3(xCoord, yCoord, zCoord) - vtx[0]) * cellSizeRCP;
    }

    DEVICE HOST int3 cellIdToCellIndex(float3 aCellIdf, float3& oInterpolationWeights) const
    {
        oInterpolationWeights.x = aCellIdf.x - floorf(aCellIdf.x);
        oInterpolationWeights.y = aCellIdf.y - floorf(aCellIdf.y);
        oInterpolationWeights.z = aCellIdf.z - floorf(aCellIdf.z);
        return make_int3(
            static_cast<int>(floorf(aCellIdf.x)),
            static_cast<int>(floorf(aCellIdf.y)),
            static_cast<int>(floorf(aCellIdf.z)));
    }

    DEVICE HOST float3 trInterpolate(int3 &cellId, float3 &weights, const cudaPitchedPtr& aPtr) const
    {
        //lower 4 voxels
        float3 val000 = *((float3*)(
            (char*)aPtr.ptr +
            cellId.y * aPtr.pitch +
            cellId.z * aPtr.pitch * aPtr.ysize) +
            cellId.x);
        float3 val001 = *((float3*)(
            (char*)aPtr.ptr +
            cellId.y * aPtr.pitch +
            cellId.z * aPtr.pitch * aPtr.ysize) +
            min(cellId.x + 1, res[0] - 1));
        float3 val00 = val000 * (1.f - weights.x) + val001 * weights.x;
        float3 val010 = *((float3*)(
            (char*)aPtr.ptr +
            min(cellId.y, res[1] - 1) * aPtr.pitch +
            cellId.z * aPtr.pitch * aPtr.ysize) +
            cellId.x);
        float3 val011 = *((float3*)(
            (char*)aPtr.ptr +
            min(cellId.y, res[1] - 1) * aPtr.pitch +
            cellId.z * aPtr.pitch * aPtr.ysize) +
            min(cellId.x + 1, res[0] - 1));
        float3 val01 = val010 * (1.f - weights.x) + val011 * weights.x;
        float3 val0 = val00 * (1.f - weights.y) + val01 * weights.y;
        //upper 4 voxels
        float3 val100 = *((float3*)(
            (char*)aPtr.ptr +
            cellId.y * aPtr.pitch +
            min(cellId.z + 1, res[2] - 1) * aPtr.pitch * aPtr.ysize) +
            cellId.x);
        float3 val101 = *((float3*)(
            (char*)aPtr.ptr +
            cellId.y * aPtr.pitch +
            min(cellId.z + 1, res[2] - 1) * aPtr.pitch * aPtr.ysize) +
            min(cellId.x + 1, res[0] - 1));
        float3 val10 = val100 * (1.f - weights.x) + val101 * weights.x;
        float3 val110 = *((float3*)(
            (char*)aPtr.ptr +
            min(cellId.y, res[1] - 1) * aPtr.pitch +
            min(cellId.z + 1, res[2]) * aPtr.pitch * aPtr.ysize) +
            cellId.x);
        float3 val111 = *((float3*)(
            (char*)aPtr.ptr +
            min(cellId.y, res[1] - 1) * aPtr.pitch +
            min(cellId.z + 1, res[2] - 1) * aPtr.pitch * aPtr.ysize) +
            min(cellId.x + 1, res[0] - 1));
        float3 val11 = val110 * (1.f - weights.x) + val111 * weights.x;
        float3 val1 = val10 * (1.f - weights.y) + val11 * weights.y;

        return val0 * (1.f - weights.z) + val1 * weights.z;
    }

    DEVICE HOST float3 nearestNeighbor(int3 &cellId, float3 &weights, const cudaPitchedPtr& aPtr) const
    {
        if(weights.x > 0.5f)
            cellId.x = min(cellId.x + 1, res[0] - 1);
        if(weights.y > 0.5f)
            cellId.y = min(cellId.y + 1, res[1] - 1);
        if(weights.z > 0.5f)
            cellId.z = min(cellId.z + 1, res[2] - 1);
        return *((float3*)(
            (char*)aPtr.ptr +
            cellId.y * aPtr.pitch +
            cellId.z * aPtr.pitch * aPtr.ysize) +
            cellId.x);
        
    }

    DEVICE HOST float3 getDiffuseReflectance(float xCoord = 0.f, float yCoord = 0.f, float zCoord = 0.f) const
    {
        float3 retval;
        if(xCoord + yCoord + zCoord == 0.f)
        {
            retval.x = diffuseReflectance.x;
            retval.y = diffuseReflectance.y;
            retval.z = diffuseReflectance.z;
        }
        else
        {
            float3 cellIdf = pointToCellId(xCoord, yCoord, zCoord);
            float3 weights;
            int3 cellId = cellIdToCellIndex(cellIdf, weights);
            switch(interpolationMode)
            {
                case InterpolationModeTrilinear:
                    retval = trInterpolate(cellId, weights, diffuseReflectanceData);
                    break;
                case InterpolationModeNearestNeighbor:
                default:
                    retval = nearestNeighbor(cellId, weights, diffuseReflectanceData);
                    break;
            };
        }
        return retval;
    }

};

#undef INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY

#endif // MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8
