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

#include "StdAfx.hpp"
#include "Application/SDLGLApplication.hpp"
#include "Application/CUDAApplication.h"//CUDAApplication

#include "../Utils/ImagePNG.hpp"

#define USE_OPEN_GL

SDL_Window* SDLGLApplication::mainwindow;

SDL_GLContext SDLGLApplication::maincontext;
GLuint SDLGLApplication::vao;
GLuint SDLGLApplication::vbo[3]; 
const GLchar *SDLGLApplication::vertexsource  = "#version 150 \n               \
                        precision highp float;\n                               \
                        in  vec2 in_Position; \n                               \
                        in  vec2 in_TexCoord; \n                               \
                        out vec2 aTexCoord; \n                                 \
                        void main(void) { \n                                   \
                            gl_Position = vec4(in_Position, 0.0, 1.0);\n       \
                            aTexCoord = in_TexCoord;\n                         \
                        }";
const GLchar *SDLGLApplication::fragmentsource = "#version 150 \n              \
                        precision highp float; \n                              \
                        uniform sampler2D uTexture; \n                         \
                        in  vec2 aTexCoord; \n                                 \
                        out vec4 gl_FragColor; \n                              \
                                                                               \
                        vec4 gammaCorrection(float aGamma)                     \
                        {                                                      \
                        float gammaRecip = 1.0 / aGamma;                       \
                        return pow(texture2D(uTexture, aTexCoord.st),          \
                        vec4(gammaRecip, gammaRecip, gammaRecip, 1.0));        \
                        }                                                      \
                                                                               \
                        void main(void) { \n                                   \
                        gl_FragColor = gammaCorrection(2.2); \n                \
                        }";
GLuint SDLGLApplication::vertexshader;
GLuint SDLGLApplication::fragmentshader;
GLuint SDLGLApplication::shaderprogram;
GLuint SDLGLApplication::sFBTextureId;
GLuint SDLGLApplication::sFBOId;

const float MAXSTEPSIZE = 200.f;
const float MINSTEPSIZE = 0.01f;
const float ROTATESCALEFACTOR = 0.003f;
const float ROTATEUPSCALEFACTOR = 0.01f;

SDLGLApplication::~SDLGLApplication()
{
#ifdef USE_OPEN_GL
    //texture
    glDeleteTextures(1, &sFBTextureId);
    //shader & buffers
    glUseProgram(0);
    glDetachShader(shaderprogram, vertexshader);
    glDetachShader(shaderprogram, fragmentshader);
    glDeleteProgram(shaderprogram);
    glDeleteShader(vertexshader);
    glDeleteShader(fragmentshader);
    glDeleteBuffers(3, vbo);
    glDeleteVertexArrays(1, &vao);

    SDL_GL_DeleteContext(maincontext);
#endif

    CUDAApplication::cleanup();
    SDL_Quit();
}

void SDLGLApplication::init(int argc, char* argv[])
{
    //CUDA specific
    CUDAApplication::deviceInit(argc, argv);
    CUDAApplication::allocateHostBuffer(mRESX, mRESY);
    //initScene();

    
//#ifdef _WIN32
//    if(!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
//    {
//        cudastd::logger::out << "Failed to set process priority.\n" ;
//    }
//#endif

    if (argc > 1)
    {
        CONFIGURATION = argv[1];
    }

    initScene();
}

void SDLGLApplication::initScene()
{

    bool haveScene = CUDAApplication::sSceneLoader.loadScene(
        CONFIGURATION,
        CUDAApplication::sAnimationManager,
        mInitialCamera,
        CUDAApplication::sAreaLightSources);

    mCamera = mInitialCamera;

    CUDAApplication::initScene();
    mRESX = mInitialCamera.getResX();
    mRESY = mInitialCamera.getResY();
    mMoveStep = CUDAApplication::getBBoxDiagonalLength() * 0.05f;
    //changeWindowSize();

    //dumps all interpolated frames as .obj
    //CUDAApplication::dumpFrames();
}

void SDLGLApplication::WindowActive	()
{
    //SDL_WM_SetCaption(mActiveWindowName, "");
}

void SDLGLApplication::WindowInactive	() 
{
   //SDL_WM_SetCaption(mMinimizedWindowName, "");
}

void SDLGLApplication::writeScreenShot()
{

    Image img(mRESX, mRESY);

    float* frameBufferFloatPtr = CUDAApplication::getFrameBuffer();

    for(int y = 0; y < mRESY; ++y)
    {
        for(int x = 0; x < mRESX; ++x)
        {
            img(x,y).x = frameBufferFloatPtr[3 * (x + mRESX * y)    ];
            img(x,y).y = frameBufferFloatPtr[3 * (x + mRESX * y) + 1];
            img(x,y).z = frameBufferFloatPtr[3 * (x + mRESX * y) + 2];
        }
    }

    img.gammaCorrect(2.2f);
    std::string filename("screenshot");
    filename += itoa(mNumScreenshots++);
    filename += ".png";
    img.writePNG(filename.c_str());
    std::cout << "Wrote " << filename << "\n";
}

void SDLGLApplication::outputCameraParameters()
{
    std::cerr << "position\t\t" 
        << mCamera.getPosition().x << "\t"
        << mCamera.getPosition().y << "\t"
        << mCamera.getPosition().z << "\n"
        << "orientation\t" 
        << mCamera.getOrientation().x << "\t"
        << mCamera.getOrientation().y << "\t"
        << mCamera.getOrientation().z << "\n"
        << "up\t\t\t"
        << mCamera.getUp().x << "\t"
        << mCamera.getUp().y << "\t"
        << mCamera.getUp().z << "\n"
        << "rotation\t\t"
        << mCamera.getRotation().x << "\t"
        << mCamera.getRotation().y << "\t"
        << mCamera.getRotation().z << "\n"
        << "resX\t\t\t" << mCamera.getResX() << "\n"
        << "resY\t\t\t" << mCamera.getResY() << "\n"
        << "FOV\t\t\t"
        << mCamera.getFOV() << "\n";

}

void SDLGLApplication::resetCamera()
{
    mCamera = mInitialCamera;
    cameraChanged();
}

void SDLGLApplication::cameraChanged()
{
    mCamera.setResX(mRESX);
    mCamera.setResY(mRESY);
    mNumImages = 0;
}

void SDLGLApplication::KeyUp		(SDL_Keysym& aSym)
{
    if(!aSym.sym)
        return;

    switch(SDL_GetKeyFromScancode(aSym.scancode)) 
    {
    case SDLK_LEFTBRACKET:
        mMoveStep = std::max(MINSTEPSIZE, mMoveStep / 2.f);
        break;           
    case SDLK_RIGHTBRACKET:
        mMoveStep = std::min(MAXSTEPSIZE, mMoveStep * 2.f);
        break;           
    case SDLK_UP:
    case SDLK_w:
        mMoveForward = false;
        break;
    case SDLK_DOWN:
    case SDLK_s:
        mMoveBackward = false;
        break;
    case SDLK_LEFT:
    case SDLK_a:
        mMoveLeft = false;
        break;
    case SDLK_RIGHT:
    case SDLK_d:
        mMoveRight = false;
        break;
    case SDLK_q:
        mMoveUp = false;
        break;
    case SDLK_z:
        mMoveDown = false;
        break;
    case SDLK_t:
        writeScreenShot();
        break;
    case SDLK_c:
        outputCameraParameters();
        break;
    case SDLK_r:
        setResolution();
        break;
    case SDLK_m:
        nextRenderMode();
        break;
    case SDLK_n:
        dumpFrames();
        break;
    case SDLK_b:
        setBackgroundColor();
        break;
    case SDLK_p:
        pauseAnimation();
        break;
    case SDLK_RCTRL:
    case SDLK_LCTRL:
        mCtrlDown = false;
        break;
    case SDLK_SPACE:
        resetCamera();
        break;
    case SDLK_F1:
        printHelp();
        break;
    case SDLK_F2:
        toggleFullSreenMode();
        break;
    default:
        break;
    }
}

void SDLGLApplication::KeyDown		(SDL_Keysym& aSym)
{
    if(!aSym.sym)
        return;

    switch(SDL_GetKeyFromScancode(aSym.scancode)) 
    {
    case SDLK_UP:
    case SDLK_w:
        mMoveForward = true;
        break;
    case SDLK_DOWN:
    case SDLK_s:
        mMoveBackward = true;
        break;
    case SDLK_LEFT:
    case SDLK_a:
        mMoveLeft = true;
        break;
    case SDLK_RIGHT:
    case SDLK_d:
        mMoveRight = true;
        break;
    case SDLK_q:
        mMoveUp = true;
        break;
    case SDLK_z:
        mMoveDown = true;
        break;
    case SDLK_t:
        break;
    case SDLK_c:
        break;
    case SDLK_r:
        break;
    case SDLK_m:
        break;
    case SDLK_n:
        break;
    case SDLK_b:
        break;
    case SDLK_p:
        break;
    case SDLK_RCTRL:
    case SDLK_LCTRL:
        mCtrlDown = true;
        break;
    case SDLK_F1:
        break;
    case SDLK_F2:
        break;
    default:
        break;
    }
}

void SDLGLApplication::MouseMoved		(const int& iButton,
                                         const int& iX, 
                                         const int& iY, 
                                         const int& iRelX, 
                                         const int& iRelY)
{
    if(mUpdateMouseCoords)
    {
        SDL_GetRelativeMouseState(&mMouseX, &mMouseY);
        mUpdateMouseCoords = false;
    }
    else
    {
        int x,y;
        SDL_GetRelativeMouseState(&x, &y);
    }

    if (mHasMouse)
    {
        if(mCtrlDown)
        {
            mUpOrientationAngleChange +=
                static_cast<float>(iRelX) * ROTATEUPSCALEFACTOR;
        }
        else
        {
            mVerticalRotationAngle +=
                static_cast<float>(iRelY) * ROTATESCALEFACTOR;

            mHorizontalRotationAngle +=
                static_cast<float>(iRelX) * ROTATESCALEFACTOR;
        }
    }
}

void SDLGLApplication::MouseButtonUp	(const int& iButton, 
                                         const int& iX, 
                                         const int& iY, 
                                         const int& iRelX, 
                                         const int& iRelY) 
{
    switch(iButton)
    {
    case SDL_BUTTON_LEFT:
        if (mHasMouse)
        {
            mHasMouse = false;
            releaseMouse();
        }
        else
        {
            mHasMouse = true;
            grabMouse();
        }
        break;
    default:
        break;
    }
}

void SDLGLApplication::grabMouse()
{
    SDL_SetRelativeMouseMode(SDL_TRUE);
    //SDL_SelectMouse();
    //SDL_ShowCursor(0);
    int x, y;
    SDL_GetRelativeMouseState(&x, &y);
    mUpdateMouseCoords = true;

}

void SDLGLApplication::releaseMouse()
{
    SDL_SetRelativeMouseMode(SDL_FALSE);
    mUpdateMouseCoords = true;

}


void SDLGLApplication::fetchEvents()
{
    // Poll for events, and handle the ones we care about.
    SDL_Event event;
    
    while ( SDL_PollEvent(&event) ) 
    {
        switch ( event.type ) 
        {
        case SDL_KEYDOWN:
            KeyDown( event.key.keysym);
            break;
        case SDL_KEYUP:
            // If escape is pressed set the Quit-flag
            if (SDL_GetKeyFromScancode(event.key.keysym.scancode) == SDLK_ESCAPE)
            {
                mQuit = true;
                break;
            }
            KeyUp( event.key.keysym);
            break;

        case SDL_QUIT:
            mQuit = true;
            break;

        case SDL_MOUSEMOTION:
            MouseMoved(
                event.button.button, 
                event.motion.x, 
                event.motion.y, 
                event.motion.xrel, 
                event.motion.yrel);
            break;

        case SDL_MOUSEBUTTONDOWN:
            break;

        case SDL_MOUSEBUTTONUP:
            MouseButtonUp(
                event.button.button, 
                event.motion.x, 
                event.motion.y, 
                event.motion.xrel, 
                event.motion.yrel);
            break;
        case SDL_WINDOWEVENT:
            switch(event.window.event) 
            { 
            case SDL_WINDOWEVENT_RESIZED: 
                {
                    SDL_WindowEvent* windowEvent =
                        reinterpret_cast<SDL_WindowEvent*>(&event);
                    mRESX = std::max(windowEvent->data1, 128);
                    mRESY = std::max(windowEvent->data2, 128);
                    changeWindowSize();
                    cameraChanged();
                    break;
                }
            case SDL_WINDOWEVENT_RESTORED:
                {
                    mMinimized = false;
                    WindowActive();
                    break;
                }

            case SDL_WINDOWEVENT_MINIMIZED:
                {
                    mMinimized = true;
                    WindowInactive();
                    break;
                }
            case SDL_WINDOWEVENT_CLOSE:
                {
                    mQuit = true;
                    break;
                }
            }
            break;
            
        } // switch
    } // while (handling input)


    processEvents();

}

void SDLGLApplication::processEvents()
{
    bool cameraChangedFlag = mMoveLeft || mMoveRight || mMoveForward ||
        mMoveBackward || mMoveUp || mMoveDown || (mVerticalRotationAngle != 0.f)
        || (mHorizontalRotationAngle != 0.f) ||
        (mUpOrientationAngleChange != 0.f);

    if (cameraChangedFlag)
    {
        cameraChanged();
    }

    float moveLeftAmount     = mMoveLeft     ? mMoveStep : 0.f;
    float moveRightAmount    = mMoveRight    ? mMoveStep : 0.f;
    float moveForwardAmount  = mMoveForward  ? mMoveStep : 0.f;
    float moveBackwardAmount = mMoveBackward ? mMoveStep : 0.f;
    float moveUpAmount       = mMoveUp       ? mMoveStep : 0.f;
    float moveDownAmount     = mMoveDown     ? mMoveStep : 0.f;

    mCamera.moveRight(-moveLeftAmount);
    mCamera.moveRight(moveRightAmount);
    mCamera.moveForward(-moveBackwardAmount);
    mCamera.moveForward(moveForwardAmount);
    mCamera.moveUp(-moveDownAmount);
    mCamera.moveUp(moveUpAmount);

    // rotate direction vector vertically
    const float3 directionRotatedUp =
        ~(cosf(mVerticalRotationAngle) * mCamera.getOrientation() -
        sinf(mVerticalRotationAngle) * mCamera.getUp());

    // rotate up vector vertically
    const float3 upRotatedUp =
        ~(cross(directionRotatedUp, cross(mCamera.getUp(), directionRotatedUp)));

    const float3 finalDirection = mCamera.rotateVector(
        directionRotatedUp, mInitialCamera.getUp(), -mHorizontalRotationAngle);

    const float3 finalUp = mCamera.rotateVector(
        upRotatedUp, mInitialCamera.getUp(), -mHorizontalRotationAngle);

    const float3 finalUp2 = mCamera.rotateVector(
        finalUp, finalDirection, mUpOrientationAngleChange);

    mCamera.setOrientation(finalDirection);
    mCamera.setUp(finalUp2);
    mCamera.setRight(~(finalDirection % finalUp2));

    mVerticalRotationAngle = mHorizontalRotationAngle 
        = mUpOrientationAngleChange = 0.f;

}

void SDLGLApplication::printHelp()
{
    std::cerr << "----------------------------------------------------------\n";
    std::cerr << "Controls                                                  \n";
    std::cerr << "----------------------------------------------------------\n";
    std::cerr << "Start Camera Rotation:\n";
    std::cerr << "    Left click with the mouse inside the window.\n\n";
    std::cerr << "Move left/right/forward/back: a/d/w/s\n"; 
    std::cerr << "Move up/down: q/z\n\n";
    std::cerr << "Adjust movement step size: Mouse scroll\n\n";
    std::cerr << "Reset Camera: Space\n\n";
    std::cerr << "Pause Animation: p\n\n";
    std::cerr << "Output camera parameters: c\n\n";
    std::cerr << "Set window width and height: r\n\n";
    std::cerr << "Set background color: b\n\n";
    std::cerr << "Change render mode: m\n\n";
    std::cerr << "Toggle full-screen mode: F2\n\n";
    std::cerr << "Write screen-shot in output/output.png: t\n";
    std::cerr << "Dump frames: n\n";


}

void SDLGLApplication::setResolution()
{
    std::cout << "Window width:  ";
    std::cin >> mRESX;
    mRESX = std::max(mRESX, 128);
    std::cout << "Window height: ";
    std::cin >> mRESY;
    mRESY = std::max(mRESY, 128);

    
    changeWindowSize();
}

void SDLGLApplication::setBackgroundColor()
{
    std::cout << "Input background colors in float format.\n";
    std::cout << "Red channel:  ";
    std::cin >> mBACKGROUND_R;
    //BACKGROUND_R = std::min(1.f, std::max(BACKGROUND_R, 0.f));
    std::cout << "Green channel:  ";
    std::cin >> mBACKGROUND_G;
    //BACKGROUND_G = std::min(1.f, std::max(BACKGROUND_G, 0.f));
    std::cout << "Blue channel:  ";
    std::cin >> mBACKGROUND_B;
    //BACKGROUND_B = std::min(1.f, std::max(BACKGROUND_B, 0.f));

    cameraChanged();
    CUDAApplication::updateBackgroundColor(mBACKGROUND_R, mBACKGROUND_G, mBACKGROUND_B);
}

void SDLGLApplication::toggleFullSreenMode(void)
{
    //TODO: Not implemented
    //changeWindowSize();
}

void SDLGLApplication::nextRenderMode()
{
    switch ( mRenderMode ) 
    {
    case DEFAULT:
        mRenderMode = PATH_TRACE;
        break;
    case PATH_TRACE:
        mRenderMode = AMBIENT_OCCLUSION;
        break;
    case AMBIENT_OCCLUSION:
    default:
        mRenderMode = DEFAULT;
        break;
    }//switch ( mRenderMode )

    cameraChanged();
}

void SDLGLApplication::dumpFrames()
{
    mDumpFrames = !mDumpFrames;

    std::cout << "AA samples per pixel: ";
    std::cin >> mPixelSamplesPerDumpedFrame;
    mPixelSamplesPerDumpedFrame = std::max(1, mPixelSamplesPerDumpedFrame);

}

void SDLGLApplication::pauseAnimation()
{
    mPauseAnimation = !mPauseAnimation;
    cameraChanged();
}
//////////////////////////////////////////////////////////////////////////
//
//OpenGL specific
//
//////////////////////////////////////////////////////////////////////////

void SDLGLApplication::initVideo()
{
#ifdef USE_OPEN_GL
        mSDLVideoModeFlags = SDL_WINDOW_OPENGL| SDL_WINDOW_SHOWN | SDL_WINDOW_INPUT_FOCUS;
        
        // Request an opengl 3.2 context.
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

        // Enable multisampling for a nice antialiased effect
        //SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
        //SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

        //SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
        //SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
        //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);

        // Turn on double buffering with a 24bit Z buffer.
        // You may need to change this to 16, 24 or 32 for your system
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        if (SDL_Init(SDL_INIT_VIDEO) < 0) // Initialize SDL's Video subsystem
        {
            std::cerr << "Unable to initialize SDL\n";
            std::cerr << "GL_VENDOR      : " << glGetString(GL_VENDOR) << "\n";
            std::cerr << "GL_RENDERER    : " << glGetString(GL_RENDERER) << "\n";
            std::cerr << "GL_VERSION     : " << glGetString(GL_VERSION) << " (required >= 3_2)\n";
        }

        /* Create our window centered at RESX x RESY resolution */
        mainwindow = SDL_CreateWindow(mActiveWindowName,  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            mRESX, mRESY, mSDLVideoModeFlags );
        if (!mainwindow) /* Die if creation failed */
            std::cerr << "Unable to create window\n";


        /* Create our opengl context and attach it to our window */
        maincontext = SDL_GL_CreateContext(mainwindow);

        /* This makes our buffer swap syncronized with the monitor's vertical refresh */
        SDL_GL_SetSwapInterval(1);

        GLenum err=glewInit();
        if(err!=GLEW_OK)
        {
            //problem: glewInit failed, something is seriously wrong
            std::cerr << "Error: "<< glewGetErrorString(err) << "\n";
        }

        /* Enable Z depth testing so objects closest to the viewpoint are in front of objects further away */
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        int maxtexsize;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtexsize);
        if (maxtexsize < mRESX || maxtexsize < mRESY)
        {
            std::cerr << "GL_MAX_TEXTURE_SIZE " << " is " << maxtexsize 
                << std::endl;
            std::cerr << "Required sizes in X and Y are "<< mRESX << " "
                << mRESY << std::endl;

        }

        initGLSL();
        initFrameBufferTexture(&sFBTextureId, mRESX, mRESY);

        SDL_ShowWindow(mainwindow);

       
        glClearColor ( 1.0f, 0.0f, 0.0f, 1.0f );
        glClear ( GL_COLOR_BUFFER_BIT );
        SDL_GL_SwapWindow(mainwindow);
#else //Do not use OpenGL

    mSDLVideoModeFlags = SDL_SWSURFACE;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) // Initialize SDL's Video subsystem
    {
        std::cerr << "Unable to initialize SDL\n";
    }

    /* Create our window centered at RESX x RESY resolution */
    mainwindow = SDL_CreateWindow(mActiveWindowName,  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        mRESX, mRESY, mSDLVideoModeFlags );
    if (!mainwindow) /* Die if creation failed */
        std::cerr << "Unable to create window\n";

#endif
}

void SDLGLApplication::displayFrame()
{
    if (mMinimized)
    {
        SDL_Delay(1500);
    } 
    else
    {
        float time, renderTime, buildTime = 0.f;
        
        if(!mPauseAnimation)
        {
            buildTime = CUDAApplication::nextFrame();
        }

        for (int i = 0; i < mPixelSamplesPerDumpedFrame; ++i)
        {
            switch ( mRenderMode ) 
            {
            case DEFAULT:
                renderTime = CUDAApplication::generateFrame(mCamera, mNumImages, 0);
                break;
            case PATH_TRACE:
                renderTime = CUDAApplication::generateFrame(mCamera, mNumImages, 1);
                break;
            case AMBIENT_OCCLUSION:
            default:
                renderTime = CUDAApplication::generateFrame(mCamera, mNumImages, 2);
                break;
            }//switch ( mRenderMode )

            if (mPauseAnimation || !mDumpFrames)
                break;

            ++mNumImages;
        }

        if(mPauseAnimation)
        {
            ++mNumImages;
        }

        time = renderTime + buildTime;
        //display frame rate in window title
        std::string windowName(mActiveWindowName);
        const float fps = 1000.f / time;
        if (fps < 10.f)
            windowName += " ";
        windowName += ftoa(fps);

        windowName += " (render: ";
       if (renderTime < 100.f)
            windowName += " ";
        if (renderTime < 10.f)
            windowName += " ";
        windowName += ftoa(renderTime);
        
        windowName += " build: ";
        if (buildTime < 10.f)
            windowName += " ";

        windowName += ftoa(buildTime);
        windowName += ") spp: ";
        windowName += itoa(mNumImages);


        SDL_SetWindowTitle(mainwindow, windowName.c_str());
#ifdef USE_OPEN_GL        
        runGLSLShader();
#else
        drawFrameBuffer();
#endif

        if(!mPauseAnimation && mDumpFrames)
        {
            writeScreenShot();
            cameraChanged();
        }
    }
}

void SDLGLApplication::initFrameBufferTexture(GLuint *aTextureId, const int aResX, const int aResY)
{   
    // create a new texture name
    glGenTextures (1, aTextureId);
    // bind the texture name to a texture target
    glBindTexture(TEXTURE_TARGET, *aTextureId);
    // turn off filtering and set proper wrap mode 
    // (obligatory for float textures atm)
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // set texenv to replace instead of the default modulate
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    // and allocate graphics memory

    glTexImage2D(TEXTURE_TARGET, 
        0, //not to use any mipmap levels for this texture
        INTERNAL_FORMAT,
        aResX,
        aResY,
        0, //turns off borders
        TEXTURE_FORMAT,
        GL_FLOAT,
        0);
}

void SDLGLApplication::drawFrameBuffer(void)
{
    if ( SDL_MUSTLOCK(SDL_GetWindowSurface(mainwindow)) ) {
        if ( SDL_LockSurface(SDL_GetWindowSurface(mainwindow)) < 0 ) {
            return;
        }
    }

    float* frameBufferFloatPtr = CUDAApplication::getFrameBuffer();
    const float gammaRCP = 1.f/2.2f;

    switch (SDL_GetWindowSurface(mainwindow)->format->BytesPerPixel) {
    case 1: { /* Assuming 8-bpp */
        for(int y = 0; y < mRESY; ++y)
        {
            for(int x = 0; x < mRESX; ++x)
            {
                int i = 3 * (x + mRESX * y);
                uint R = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i    ], gammaRCP));
                uint G = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 1], gammaRCP));
                uint B = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 2], gammaRCP));
                uint color = SDL_MapRGB(SDL_GetWindowSurface(mainwindow)->format, R,G,B);

                Uint8 *bufp;
                bufp = (Uint8 *)SDL_GetWindowSurface(mainwindow)->pixels + y*SDL_GetWindowSurface(mainwindow)->pitch + x;
                *bufp = color;
            }
        }        
            }
            break;

    case 2: { /* Probably 15-bpp or 16-bpp */
        for(int y = 0; y < mRESY; ++y)
        {
            for(int x = 0; x < mRESX; ++x)
            {
                int i = 3 * (x + mRESX * y);
                uint R = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i    ], gammaRCP));
                uint G = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 1], gammaRCP));
                uint B = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 2], gammaRCP));
                uint color = SDL_MapRGB(SDL_GetWindowSurface(mainwindow)->format, R,G,B);

                Uint16 *bufp;

                bufp = (Uint16 *)SDL_GetWindowSurface(mainwindow)->pixels + y*SDL_GetWindowSurface(mainwindow)->pitch/2 + x;
                *bufp = color;
            }
        }        

            }
            break;

    case 3: { /* Slow 24-bpp mode, usually not used */
        for(int y = 0; y < mRESY; ++y)
        {
            for(int x = 0; x < mRESX; ++x)
            {
                int i = 3 * (x + mRESX * y);
                uint R = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i    ], gammaRCP));
                uint G = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 1], gammaRCP));
                uint B = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 2], gammaRCP));
                uint color = SDL_MapRGB(SDL_GetWindowSurface(mainwindow)->format, R,G,B);

                Uint8 *bufp;

                bufp = (Uint8 *)SDL_GetWindowSurface(mainwindow)->pixels + y*SDL_GetWindowSurface(mainwindow)->pitch + x;
                *(bufp+SDL_GetWindowSurface(mainwindow)->format->Rshift/8) = R;
                *(bufp+SDL_GetWindowSurface(mainwindow)->format->Gshift/8) = G;
                *(bufp+SDL_GetWindowSurface(mainwindow)->format->Bshift/8) = B;
            }
        }        

            }
            break;

    case 4: { /* Probably 32-bpp */
        for(int y = 0; y < mRESY; ++y)
        {
            for(int x = 0; x < mRESX; ++x)
            {
                int i = 3 * (x + mRESX * y);
                uint R = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i    ], gammaRCP));
                uint G = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 1], gammaRCP));
                uint B = static_cast<uint>(255.f * powf(frameBufferFloatPtr[i + 2], gammaRCP));
                uint color = SDL_MapRGB(SDL_GetWindowSurface(mainwindow)->format, R,G,B);

                Uint32 *bufp;

                bufp = (Uint32 *)SDL_GetWindowSurface(mainwindow)->pixels + y*SDL_GetWindowSurface(mainwindow)->pitch/4 + x;
                *bufp = color;
            }
        }        


            }
            break;
    }
    if ( SDL_MUSTLOCK(SDL_GetWindowSurface(mainwindow)) ) {
        SDL_UnlockSurface(SDL_GetWindowSurface(mainwindow));
    }
    SDL_UpdateWindowSurface(mainwindow);

}

void SDLGLApplication::initGLSL(void)
{
        int isCompiled_VS, isCompiled_FS;
        int IsLinked;
        int maxLength;
        char *vertexInfoLog;
        char *fragmentInfoLog;
        char *shaderProgramInfoLog;

        /* Allocate and assign a Vertex Array Object to our handle */
        glGenVertexArrays(1, &vao);

        /* Bind our Vertex Array Object as the current used object */
        glBindVertexArray(vao);

        /* Allocate and assign tree Vertex Buffer Objects to our handle */
        glGenBuffers(3, vbo);

        /* Create an empty vertex shader handle */
        vertexshader = glCreateShader(GL_VERTEX_SHADER);

        /* Send the vertex shader source code to GL */
        /* Note that the source code is NULL character terminated. */
        /* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
        glShaderSource(vertexshader, 1, (const GLchar**)&vertexsource, 0);

        /* Compile the vertex shader */
        glCompileShader(vertexshader);

        glGetShaderiv(vertexshader, GL_COMPILE_STATUS, &isCompiled_VS);
        if(isCompiled_VS == 0)
        {
            glGetShaderiv(vertexshader, GL_INFO_LOG_LENGTH, &maxLength);

            /* The maxLength includes the NULL character */
            vertexInfoLog = (char *)malloc(maxLength);

            glGetShaderInfoLog(vertexshader, maxLength, &maxLength, vertexInfoLog);

            /* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
            std::cerr << "Shader program compilation failed\n";
            std::cerr << vertexInfoLog << "\n";

            free(vertexInfoLog);
            return;
        }

        /* Create an empty fragment shader handle */
        fragmentshader = glCreateShader(GL_FRAGMENT_SHADER);

        /* Send the fragment shader source code to GL */
        /* Note that the source code is NULL character terminated. */
        /* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
        glShaderSource(fragmentshader, 1, (const GLchar**)&fragmentsource, 0);

        /* Compile the fragment shader */
        glCompileShader(fragmentshader);

        glGetShaderiv(fragmentshader, GL_COMPILE_STATUS, &isCompiled_FS);
        if(isCompiled_FS == 0)
        {
            glGetShaderiv(fragmentshader, GL_INFO_LOG_LENGTH, &maxLength);

            /* The maxLength includes the NULL character */
            fragmentInfoLog = (char *)malloc(maxLength);

            glGetShaderInfoLog(fragmentshader, maxLength, &maxLength, fragmentInfoLog);

            /* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
            std::cerr << "Shader program compilation failed\n";
            std::cerr << fragmentInfoLog << "\n";

            free(fragmentInfoLog);
            return;
        }

        /* If we reached this point it means the vertex and fragment shaders compiled and are syntax error free. */
        /* We must link them together to make a GL shader program */
        /* GL shader programs are monolithic. It is a single piece made of 1 vertex shader and 1 fragment shader. */
        /* Assign our program handle a "name" */
        shaderprogram = glCreateProgram();

        /* Attach our shaders to our program */
        glAttachShader(shaderprogram, vertexshader);
        glAttachShader(shaderprogram, fragmentshader);

        /* Bind attribute index 0 (coordinates) to in_Position, attribute index 1 to in_TexCoord */
        /* Attribute locations must be setup before calling glLinkProgram. */
        glBindAttribLocation(shaderprogram, 0, "in_Position");
        glBindAttribLocation(shaderprogram, 1, "in_TexCoord");

        /* Link our program */
        /* At this stage, the vertex and fragment programs are inspected, optimized and a binary code is generated for the shader. */
        /* The binary code is uploaded to the GPU, if there is no error. */
        glLinkProgram(shaderprogram);

        /* Again, we must check and make sure that it linked. If it fails, it would mean either there is a mismatch between the vertex */
        /* and fragment shaders. It might be that you have surpassed your GPU's abilities. Perhaps too many ALU operations or */
        /* too many texel fetch instructions or too many interpolators or dynamic loops. */
        glGetProgramiv(shaderprogram, GL_LINK_STATUS, (int *)&IsLinked);
        if(IsLinked == 0)
        {
            /* Noticed that glGetProgramiv is used to get the length for a shader program, not glGetShaderiv. */
            glGetProgramiv(shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);

            /* The maxLength includes the NULL character */
            shaderProgramInfoLog = (char *)malloc(maxLength);

            /* Notice that glGetProgramInfoLog, not glGetShaderInfoLog. */
            glGetProgramInfoLog(shaderprogram, maxLength, &maxLength, shaderProgramInfoLog);

            /* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
            /* In this simple program, we'll just leave */
            free(shaderProgramInfoLog);
            return;
        }

        /* Load the shader into the rendering pipeline */
        glUseProgram(shaderprogram);
}

void SDLGLApplication::runGLSLShader(void)
{
    glEnable(TEXTURE_TARGET);
    // enable texture x (read-only, not changed in the computation loop)
    glActiveTexture(GL_TEXTURE0);	
    glBindTexture(TEXTURE_TARGET, sFBTextureId);
    glUniform1i(glGetUniformLocation(shaderprogram, "uTexture"), 0); // texture unit 0

    //////////////////////////////////////////////////////////////////////////
    //NVidia
    float* frameBufferFloatPtr = CUDAApplication::getFrameBuffer();
    glTexSubImage2D(TEXTURE_TARGET,0,0,0,mRESX,mRESY,
        TEXTURE_FORMAT, GL_FLOAT, frameBufferFloatPtr);
    //////////////////////////////////////////////////////////////////////////

    const GLfloat quad[4][2] = 
    {
        {-1.f, -1.f},
        {1.f, -1.f},
        {1.f, 1.f},
        {-1.f, 1.f}
    };

    const GLfloat texCoords[4][2] =
    {
        {0.f, 1.f},
        {1.f, 1.f},
        {1.f, 0.f},
        {0.f, 0.f}
    };

    const GLuint indices[6] = {0,1,2,0,3,2};

    /////////////////////////////////////////////////////////////////////////////////////////
    //Setup OpenGL buffers

    /* Bind our first VBO as being the active buffer and storing vertex attributes (coordinates) */
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

    /* Copy the vertex positions to our buffer */
    /* 4 * 3 * sizeof(GLfloat) is the size of the positions array, since it contains 4 * 3 GLfloat values */
    glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), quad, GL_STATIC_DRAW);

    /* Specify that our coordinate data is going into attribute index 0, and contains 2 floats per vertex */
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Enable attribute index 0 as being used */
    glEnableVertexAttribArray(0);

    /* Bind our second VBO as being the active buffer and storing vertex attributes (texture coordinates) */
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

    /* Copy the color data from colors to our buffer */
    /* 4 * 3 * sizeof(GLfloat) is the size of the colors array, since it contains 4 * 3 GLfloat values */
    glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);

    /* Specify that our color data is going into attribute index 1, and contains 2 floats per vertex */
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Enable attribute index 1 as being used */
    glEnableVertexAttribArray(1);

    /* Bind our third VBO as being the active buffer and storing vertex attributes (indices) */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);

    /* Copy the index data from indices to our buffer */
    /* 2 * 3 * sizeof(GLfloat) is the size of the indices array, since it contains 2*3 GLubyte values */
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    /* Specify that our index data is going into attribute index 2, and contains three ints per vertex */
    glVertexAttribPointer(2, 3, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

    /* Enable attribute index 2 as being used */
    glEnableVertexAttribArray(2);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Invoke glDrawElements telling it to draw triangles using indicies */
    glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, 0);

    /* Disable vertex arrays */
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    //un-bind texture
    glBindTexture(TEXTURE_TARGET, 0);

    glDisable(TEXTURE_TARGET);

    SDL_GL_SwapWindow(mainwindow);
}

void SDLGLApplication::cleanup(void)
{
#ifdef USE_OPEN_GL
    //texture
    glDeleteTextures(1, &sFBTextureId);
    //shader & buffers
    glUseProgram(0);
    glDetachShader(shaderprogram, vertexshader);
    glDetachShader(shaderprogram, fragmentshader);
    glDeleteProgram(shaderprogram);
    glDeleteShader(vertexshader);
    glDeleteShader(fragmentshader);
    glDeleteBuffers(3, vbo);
    glDeleteVertexArrays(1, &vao);

    SDL_GL_DeleteContext(maincontext);
#endif

    SDL_DestroyWindow(mainwindow);
    CUDAApplication::cleanup();
}

void SDLGLApplication::changeWindowSize(void)
{
    cameraChanged();
#ifdef USE_OPEN_GL
    //texture
    glDeleteTextures(1, &sFBTextureId);
     //shader & buffers
    glUseProgram(0);
    glDetachShader(shaderprogram, vertexshader);
    glDetachShader(shaderprogram, fragmentshader);
    glDeleteProgram(shaderprogram);
    glDeleteShader(vertexshader);
    glDeleteShader(fragmentshader);
    glDeleteBuffers(3, vbo);
    glDeleteVertexArrays(1, &vao);

    SDL_GL_DeleteContext(maincontext);
#endif
    SDL_SetWindowSize(mainwindow, mRESX, mRESY);

#ifdef USE_OPEN_GL
    maincontext = SDL_GL_CreateContext(mainwindow);
    initGLSL();
    initFrameBufferTexture(&sFBTextureId, mRESX, mRESY);
#endif
}
