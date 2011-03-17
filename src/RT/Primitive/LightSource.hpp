/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
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

#ifndef LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
#define LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9

#include "CUDAStdAfx.h"

struct AreaLightSource
{
    float3 position, normal, intensity, edge1, edge2;

    DEVICE HOST float3 AreaLightSource::getPoint(float aXCoord, float aYCoord) const
    {
        float3 result;
        result.x = position.x + aXCoord * edge1.x + aYCoord * edge2.x;
        result.y = position.y + aXCoord * edge1.y + aYCoord * edge2.y;
        result.z = position.z + aXCoord * edge1.z + aYCoord * edge2.z;
        return result;
    }

    DEVICE HOST float AreaLightSource::getArea() const
    {
        float3 crossProduct;
        crossProduct.x = edge1.y * edge2.z - edge1.z * edge2.y;
        crossProduct.y = edge1.z * edge2.x - edge1.x * edge2.z;
        crossProduct.z = edge1.x * edge2.y - edge1.y * edge2.x;

        float dotProduct =
            crossProduct.x * crossProduct.x +
            crossProduct.y * crossProduct.y +
            crossProduct.z * crossProduct.z;

        return sqrtf(dotProduct);
        //return len((edge1 % edge2));
    }

    DEVICE HOST void AreaLightSource::init(
        const float3& aVtx0, const float3& aVtx1,
        const float3& aVtx2, const float3& aVtx3,
        const float3& aIntensity, const float3& aNormal)
    {
        position = aVtx0;
        normal = aNormal;
        edge1.x = aVtx1.x - aVtx0.x;
        edge1.y = aVtx1.y - aVtx0.y;
        edge1.z = aVtx1.z - aVtx0.z;
        edge2.x = aVtx3.x - aVtx0.x;
        edge2.y = aVtx3.y - aVtx0.y;
        edge2.z = aVtx3.z - aVtx0.z;
        intensity = aIntensity;
    }

};



#endif // LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
