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
#include "Textures.h"
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
RTEngine            gRTEngine;
FrameBuffer         gFrameBuffer;

void CUDAApplication::initScene()
{
        gRTEngine.init();
        gRTEngine.upload(sAnimationManager.getFrame(0), sAnimationManager.getFrame(0), 1.f);
        gRTEngine.buildAccStruct();
        gFrameBuffer.init(sRESX, sRESY);

}

float CUDAApplication::nextFrame()
{
    sAnimationManager.nextFrame();
    const size_t frameId1 = sAnimationManager.getFrameId();
    const size_t frameId2 = sAnimationManager.getNextFrameId();

    gRTEngine.upload(
        sAnimationManager.getFrame(frameId1),
        sAnimationManager.getFrame(frameId2),
        sAnimationManager.getInterpolationCoefficient());
    
    cudaEvent_t mStart, mEnd;
    cudaEventCreate(&mStart);
    cudaEventCreate(&mEnd);
    cudaEventRecord(mStart, 0);

    gRTEngine.buildAccStruct();
    
    cudaEventRecord(mEnd, 0);
    cudaEventSynchronize(mEnd);

    float oBuildTime;
    cudaEventElapsedTime(&oBuildTime, mStart, mEnd);

    cudaEventDestroy(mStart);
    cudaEventDestroy(mEnd);
    
    return oBuildTime;
}

float CUDAApplication::generateFrame(
           CameraManager& aView, int& aImageId)
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
        gRTEngine.setCamera(
            aView.getPosition(),
            aView.getOrientation(),
            aView.getUp(),
            aView.getFOV(),
            aView.getResX(),
            aView.getResY()
            );
    }

    cudaEvent_t mStart, mEnd;
    cudaEventCreate(&mStart);
    cudaEventCreate(&mEnd);
    cudaEventRecord(mStart, 0);

    gRTEngine.renderFrame(gFrameBuffer, aImageId);

    cudaEventRecord(mEnd, 0);
    cudaEventSynchronize(mEnd);

    gFrameBuffer.download((float3*)sFrameBufferFloatPtr, sRESX, sRESY);

    float oRenderTime;
    cudaEventElapsedTime(&oRenderTime, mStart, mEnd);
    
    cudaEventDestroy(mStart);
    cudaEventDestroy(mEnd);

    return oRenderTime;
}
