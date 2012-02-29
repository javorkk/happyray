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

// OpenGL extensions
#include <GL/glew.h>
#ifdef _WIN32
#   include <GL/wglew.h>
#else
#   include <GL/glxew.h>
#endif

//#define GL3_PROTOTYPES 1 // Ensure we are using opengl's core profile only
//#include <GL3/gl3.h>

//SDL is a cross-platform windowing library
#ifdef _WIN32
#   include "SDL.h"
#else
#   include "SDL/SDL.h"
#endif

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
    bool mDumpFrames;
    bool mPauseAnimation;
    enum RenderMode {DEFAULT = 0, PATH_TRACE = 1, AMBIENT_OCCLUSION = 2};
    RenderMode mRenderMode;
    //////////////////////////////////////////////////////////////////////////
    //IO
    const char* mMinimizedWindowName;
    const char* mActiveWindowName;
    char* CONFIGURATION;
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
    static GLuint vao;//Vertex Array Object
    static GLuint vbo[3]; //tree Vertex Buffer Objects
    static const GLchar *vertexsource;//vtx shader source
    static const GLchar *fragmentsource;//fragment shader source
    static GLuint vertexshader, fragmentshader;//shaders
    static GLuint shaderprogram; //shader program
    //////////////////////////////////////////////////////////////////////////
    //Textures & related
    //GL_TEXTURE_RECTANGLE_ARB allows non-normalized texture coordinates
    static const int TEXTURE_TARGET = GL_TEXTURE_2D;
    static const int INTERNAL_FORMAT = GL_RGB32F_ARB;//GL_LUMINANCE32F_ARB;
    static const int TEXTURE_FORMAT = GL_RGB;//GL_LUMINANCE;
    //static float* frameBufferFloatPtr;
    static GLuint sFBTextureId;
    static GLuint sFBOId;
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
        mDumpFrames(false),mPauseAnimation(false),
        mRenderMode(DEFAULT), mMinimizedWindowName("Happy Ray"),
        mActiveWindowName("Fps: "), CONFIGURATION("scene.cfg")
    {}

    ~SDLGLApplication();

    void init(int argc, char* argv[]);

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

    //////////////////////////////////////////////////////////////////////////
    //
    //OpenGL specific
    //
    //////////////////////////////////////////////////////////////////////////
    void initVideo();

    static void initFrameBufferTexture(GLuint *aTextureId, const int aResX, const int aResY);

    static void initGLSL(void);

    void runGLSLShader(void);

    //////////////////////////////////////////////////////////////////////////

    void cleanup(void);

    void changeWindowSize();
};

#endif // SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F
