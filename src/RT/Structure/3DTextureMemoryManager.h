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

#ifndef TEXTUREMEMORYMANAGER_H_INCLUDED_A00594FA_C85D_4080_847A_7415A320891E
#define TEXTUREMEMORYMANAGER_H_INCLUDED_A00594FA_C85D_4080_847A_7415A320891E

#include "CUDAStdAfx.h"
#include "RT/Primitive/Material.hpp"
#include "RT/Primitive/BBox.hpp"

class TextureMemoryManager
{
public:

    int resX, resY, resZ;
    int oldResX, oldResY, oldResZ;
    BBox bounds;

    cudaPitchedPtr texelPtrDevice;
    cudaPitchedPtr texelPtrHost;
    float3*        texelsDevice;
    float3*        texelsHost;


    TextureMemoryManager()
        :resX(-1), resY(-1), resZ(-1), oldResX(0), oldResY(0), oldResZ(0), bounds(BBox::empty()),
        texelsHost(NULL), texelsDevice(NULL)
    {}


    HOST void checkResolution();


    float3& getTexel(uint aX, uint aY = 0u, uint aZ = 0u)
    {
        aX = min(aX, resX - 1);
        aY = min(aY, resY - 1);
        aZ = min(aZ, resZ - 1);
        return *((float3*)((char*)texelPtrHost.ptr + aY * texelPtrHost.pitch + aZ * texelPtrHost.pitch * texelPtrHost.ysize) + aX);
    }


    const float3 getResolution() const
    {
        float3 retval;
        retval.x = static_cast<float>(resX);
        retval.y = static_cast<float>(resY);
        retval.z = static_cast<float>(resZ);
        return retval;
    }

    float3 getCellSize() const
    {
        return bounds.diagonal() / getResolution();
    }

    float3 getCellSizeRCP() const
    {
        return getResolution() / bounds.diagonal();
    }


    //////////////////////////////////////////////////////////////////////////
    //data transfer related
    //////////////////////////////////////////////////////////////////////////
    TexturedPhongMaterial getParameters() const
    {
        TexturedPhongMaterial retval;
        retval.vtx[0] = bounds.vtx[0]; //bounds min
        retval.vtx[1] = bounds.vtx[1]; //bounds max
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.cellSizeRCP = getCellSizeRCP();
        retval.diffuseReflectanceData = texelPtrDevice;
        return retval;
    }

    HOST void copyDataDeviceToHost();

    HOST void copyDataHostToDevice();

    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    HOST cudaPitchedPtr allocateDataHost();

    HOST cudaPitchedPtr allocateDataDevice();

    HOST void setDeviceCellsToZero();

    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    HOST void freeDataDevice();

    HOST void freeDataHost();

    HOST void cleanup();

};

#endif // 3DTEXTUREMEMORYMANAGER_H_INCLUDED_A00594FA_C85D_4080_847A_7415A320891E
