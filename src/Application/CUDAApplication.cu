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
#include "CUDAStdAfx.h"
#include "Application/CUDAApplication.h"
#include "RT/RTEngine.h"
#include "RT/Structure/FrameBuffer.h"


float*              CUDAApplication::sFrameBufferFloatPtr;
int                 CUDAApplication::sRESX;
int                 CUDAApplication::sRESY;
float               CUDAApplication::sBACKGROUND_R;
float               CUDAApplication::sBACKGROUND_G;
float               CUDAApplication::sBACKGROUND_B;
SceneLoader         CUDAApplication::sSceneLoader;
AnimationManager    CUDAApplication::sAnimationManager;
AreaLightSource     CUDAApplication::sAreaLightSource;
StaticRTEngine      gStaticRTEngine;
FrameBuffer         gFrameBuffer;

void CUDAApplication::initScene()
{
        gStaticRTEngine.init();
        gStaticRTEngine.upload(sAnimationManager.getFrame(0));
}

void CUDAApplication::generateFrame(
           CameraManager& aView, int& aImageId,
           float& oRenderTime, float& oBuildTime)
{
    if(aView.getResX() != sRESX || aView.getResY() != sRESY)
    {
        sRESX = aView.getResX();
        sRESY = aView.getResY();
        allocateHostBuffer(aView.getResX(), aView.getResY());
        gFrameBuffer.cleanup();
        gFrameBuffer.init(sRESX, sRESY);
    }

    if(aImageId == 0)
    {
        gStaticRTEngine.setCamera(
            aView.getPosition(),
            aView.getOrientation(),
            aView.getUp(),
            aView.getFOV(),
            aView.getResX(),
            aView.getResY()
            );
    }

    gStaticRTEngine.renderFrame(gFrameBuffer, aImageId);

    gFrameBuffer.download((float3*)sFrameBufferFloatPtr, sRESX, sRESY);

    oBuildTime = 28.f;
    oRenderTime = 30.f;
}
