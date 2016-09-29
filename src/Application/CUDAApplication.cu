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


float*                      CUDAApplication::sFrameBufferFloatPtr;
int                         CUDAApplication::sRESX;
int                         CUDAApplication::sRESY;
float                       CUDAApplication::sBACKGROUND_R;
float                       CUDAApplication::sBACKGROUND_G;
float                       CUDAApplication::sBACKGROUND_B;
SceneLoader                 CUDAApplication::sSceneLoader;
AnimationManager            CUDAApplication::sAnimationManager;
AreaLightSourceCollection   CUDAApplication::sAreaLightSources;
RTEngine                    gRTEngine;
FrameBuffer                 gFrameBuffer;


void CUDAApplication::deviceInit(int argc, char** argv)
{
    cudastd::getBestCUDADevice(argc, argv);

    if (argc > 2)
    {
        //cudastd::logger::out << argv[2] <<"\n";
        std::string rayFileName = argv[2];
        gRTEngine.setInputRayFileName(rayFileName);
    }
    if (argc > 4)
    {
        //cudastd::logger::out << argv[3] << "\n";
        //cudastd::logger::out << argv[4] << "\n";
        float topDensity = (float)atof(argv[3]);
        float leafDensity = (float)atof(argv[4]);
        gRTEngine.setGridDensities(topDensity, leafDensity);
    }
}

void CUDAApplication::initScene()
{
        gRTEngine.init();
        gRTEngine.upload(sAnimationManager.getFrame(0), sAnimationManager.getFrame(0), 1.f);
        gRTEngine.buildAccStruct();
        gRTEngine.setLights(sAreaLightSources);
        gFrameBuffer.init(sRESX, sRESY);

}

float CUDAApplication::getBBoxDiagonalLength()
{
    return gRTEngine.getBoundingBoxDiagonalLength();
}

float CUDAApplication::nextFrame(bool reverseDirection)
{
    if (reverseDirection)
    {
        sAnimationManager.previousFrame();
    }
    else
    {
        sAnimationManager.nextFrame();
    }
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
    cudaEventSynchronize(mStart);

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
           CameraManager& aView, int& aImageId, int aRenderMode)
{
    if(aView.getResX() != sRESX || aView.getResY() != sRESY)
    {
        sRESX = aView.getResX();
        sRESY = aView.getResY();
        allocateHostBuffer(sRESX, sRESY);
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
    //cudaEventSynchronize(mStart);

    gRTEngine.renderFrame(gFrameBuffer, aImageId, aRenderMode);

    cudaEventRecord(mEnd, 0);
    cudaEventSynchronize(mEnd);

    gFrameBuffer.download((float3*)sFrameBufferFloatPtr);

    float oRenderTime;
    cudaEventElapsedTime(&oRenderTime, mStart, mEnd);
    
    cudaEventDestroy(mStart);
    cudaEventDestroy(mEnd);
    MY_CUT_CHECK_ERROR("Error after frame buffer downloat (dev-host)!\n");

    return oRenderTime;
}


void CUDAApplication::setGridDensities(float aTopLevel, float aLeafLevel)
{
    gRTEngine.setGridDensities(aTopLevel, aLeafLevel);
}

void CUDAApplication::dumpFrames()
{
    if(sAnimationManager.getNumKeyFrames() <= 1u)
        return;

    cudastd::logger::out << "Dumping interpolated frame  ";
    size_t frameId1 = sAnimationManager.getFrameId();
    size_t frameId2 = sAnimationManager.getNextFrameId();
    size_t previousFrameId1 = frameId1;
    while (true)
    {
        for (int zeroes = 0; zeroes <= previousFrameId1 / 10; ++zeroes)
            cudastd::logger::out << "\b";
        
        cudastd::logger::out << frameId1;
        
        if (frameId1 != previousFrameId1 && frameId1 == 0u)
            break;

        sAnimationManager.dumpFrame();
        sAnimationManager.nextFrame();
        
        previousFrameId1 = frameId1;
        frameId1 = sAnimationManager.getFrameId();
        frameId2 = sAnimationManager.getNextFrameId();
    }

    cudastd::logger::out << "...done.\n";
}


void CUDAApplication::cleanup()
{
    gRTEngine.cleanup();
    gFrameBuffer.cleanup();
    deallocateHostBuffer();
}
