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

#ifndef SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F
#define SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F

//////////////////////////////////////////////////////////////////////////
//SDL is a cross-platform windowing...
//////////////////////////////////////////////////////////////////////////
#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__) && !defined(__NVCC__) && !defined(__CUDABE__) && !defined(__CUDANVVM__)
#if !defined(__APPLE__)
#    ifdef _WIN32
#        include "../contrib/include/SDL.h"
#    else
#        include <SDL2/SDL.h>
#    endif
#else
#    include <SDL2/SDL.h>
#endif //__APPLE__
#endif //HAS CUDA

//////////////////////////////////////////////////////////////////////////
//CUDA specific includes (host code only)
//////////////////////////////////////////////////////////////////////////
#include "Application/CameraManager.hpp"

class SDLGLApplication
{
    //////////////////////////////////////////////////////////////////////////
    //Window
    int mMouseX;
    int mMouseY;

    bool mMinimized, mQuit;
    bool mUpdateMouseCoords;
    bool mHasMouse, mCtrlDown;

    unsigned int mSDLVideoModeFlags;

    //////////////////////////////////////////////////////////////////////////
    //Camera
    float mMoveStep;
    bool mMoveLeft;
    bool mMoveRight;
    bool mMoveForward;
    bool mMoveBackward;
    bool mMoveUp;
    bool mMoveDown;
    float mVerticalRotationAngle, mHorizontalRotationAngle;
    float mUpOrientationAngleChange;
    CameraManager mInitialCamera;
    CameraManager mCamera;
    //////////////////////////////////////////////////////////////////////////
    //Misc
    int mNumImages;
    int mNumScreenshots;
    int mPixelSamplesPerDumpedFrame;
    bool mDumpFrames;
    bool mPauseAnimation;
    enum RenderMode { DEFAULT = 0, PATH_TRACE = 1, PATH_TRACE_SIMPLE = 2, AMBIENT_OCCLUSION = 3, OPEN_GL = 4 };
    RenderMode mRenderMode;
    //////////////////////////////////////////////////////////////////////////
    //IO
    const char* mMinimizedWindowName;
    const char* mActiveWindowName;
    std::string CONFIGURATION;
public:
    //////////////////////////////////////////////////////////////////////////
    //Window
    int mRESX;
    int mRESY;
    float mBACKGROUND_R;
    float mBACKGROUND_G;
    float mBACKGROUND_B;

    static SDL_Window* mainwindow;     //SDL window
    Uint16 mGammaRamp;
    //////////////////////////////////////////////////////////////////////////
    // OpenGL
    static SDL_GLContext maincontext; //OpenGL Context
    //////////////////////////////////////////////////////////////////////////

    SDLGLApplication():mRESX(256), mRESY(256), mBACKGROUND_R(0.5f),
        mBACKGROUND_G(0.5f), mBACKGROUND_B(0.5f),
        mMinimized(false), mQuit(false), mUpdateMouseCoords(false),
        mHasMouse(false), mCtrlDown(false),
        mMoveStep(16.f),
        mMoveLeft(false), mMoveRight(false), mMoveForward(false),
        mMoveBackward(false), mMoveUp(false), mMoveDown(false),
        mVerticalRotationAngle(0.f), mHorizontalRotationAngle(0.f),
        mUpOrientationAngleChange(0.f), mNumImages(0), mNumScreenshots(0),
        mPixelSamplesPerDumpedFrame(1),
        mDumpFrames(false),mPauseAnimation(false),
        mRenderMode(DEFAULT), mMinimizedWindowName("Happy Ray"),
        mActiveWindowName("Fps: "), CONFIGURATION("scene.cfg")
    {}

    ~SDLGLApplication();

    void init(int argc, char** argv);

    void initScene();

    bool dead()
    {
        return mQuit;
    }

    

    //////////////////////////////////////////////////////////////////////////
    //
    //SDL specific
    //
    //////////////////////////////////////////////////////////////////////////

    void displayFrame();

    //copy the float3 frame buffer w/o using openGL
    void drawFrameBuffer();

    /** Window is active again. **/
    void WindowActive();

    /** Window is inactive. **/
    void WindowInactive	();

    void writeScreenShot();

    void outputCameraParameters();

    /** Keyboard key has been released.
    @param iKeyEnum The key number.
    **/
    void KeyUp		(SDL_Keysym& aSym);

    /** Keyboard key has been pressed.
    @param iKeyEnum The key number.
    **/
    void KeyDown		(SDL_Keysym& aSym);


    /** The mouse has been moved.
    @param iButton	Specifies if a mouse button is pressed.
    @param iX	The mouse position on the X-axis in pixels.
    @param iY	The mouse position on the Y-axis in pixels.
    @param iRelX	The mouse position on the X-axis relative to the last position, in pixels.
    @param iRelY	The mouse position on the Y-axis relative to the last position, in pixels.

    @bug The iButton variable is always NULL.
    **/
    void MouseMoved		(const int& iButton, const int& iX, const int& iY, const int& iRelX, const int& iRelY);

    /** A mouse button has been released.
    @param iButton	Specifies if a mouse button is pressed.
    @param iX	The mouse position on the X-axis in pixels.
    @param iY	The mouse position on the Y-axis in pixels.
    @param iRelX	The mouse position on the X-axis relative to the last position, in pixels.
    @param iRelY	The mouse position on the Y-axis relative to the last position, in pixels.
    **/

    void MouseButtonUp	(const int& iButton,  const int& iX, const int& iY, const int& iRelX, const int& iRelY);

    void grabMouse();

    void releaseMouse();

    void fetchEvents();

    void processEvents();

    void resetCamera();

    void cameraChanged();

    void printHelp();

    void setResolution();

    void setBackgroundColor();

    void toggleFullSreenMode();

    void nextRenderMode();

    void dumpFrames();

    void pauseAnimation();

    void previousFrame();
    void nextFrame();

    //////////////////////////////////////////////////////////////////////////
    //
    //OpenGL specific
    //
    //////////////////////////////////////////////////////////////////////////
    void initVideo();

    //////////////////////////////////////////////////////////////////////////

    void cleanup(void);

    void changeWindowSize();
};

#endif // SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F
