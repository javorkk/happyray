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

#ifndef CUDAAPPLICATION_HPP_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51
#define CUDAAPPLICATION_HPP_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51

#include "CUDAStdAfx.h"
#include "Utils/CUDAUtil.h"

#include "Application/CameraManager.hpp"
#include "Application/WFObject.hpp"
#include "Application/AnimationManager.hpp"

#include "Application/SceneLoader.hpp"


class CUDAApplication
{
    static float* sFrameBufferFloatPtr;
    static int sRESX;
    static int sRESY;
    static float sBACKGROUND_R;
    static float sBACKGROUND_G;
    static float sBACKGROUND_B;
public:
    static SceneLoader                  sSceneLoader;
    static AnimationManager             sAnimationManager;
    static AreaLightSourceCollection    sAreaLightSources;

    static void allocateHostBuffer(const int aResX, const int aResY)
    {
        deallocateHostBuffer();
        MY_CUDA_SAFE_CALL( cudaHostAlloc((void**)&sFrameBufferFloatPtr, aResX * aResY * 3 * sizeof(float), cudaHostAllocDefault) );
    }

    static void deallocateHostBuffer()
    {
        if (sFrameBufferFloatPtr != NULL)
        {
            MY_CUDA_SAFE_CALL(cudaFreeHost(sFrameBufferFloatPtr) );
            sFrameBufferFloatPtr = NULL;
        }
    }

    static void deviceInit(int argc, char* argv[])
    {
        cudastd::getBestCUDADevice(argc, argv);
    }

    static void changeWindowSize(const int aResX, const int aResY)
    {
       allocateHostBuffer(aResX, aResY);
    }

    static void initScene();
    static float nextFrame();

    static float generateFrame(
        CameraManager& aView,
        int& aImageId);

    static void updateBackgroundColor(float aR, float aG, float aB)
    {
        sBACKGROUND_R = aR;
        sBACKGROUND_G = aG;
        sBACKGROUND_B = aB;
    }

    static float*& getFrameBuffer()
    {
        return sFrameBufferFloatPtr;
    }

    static void cleanup();
};


#endif // CUDAAPPLICATION_H_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51
