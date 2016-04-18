/****************************************************************************/
/* Copyright (c) 2016, Javor Kalojanov
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

#ifndef SDLGLAPPLICATION_HPP_INCLUDED_AD5D925E_13F1_4765_9E59_CDEF4B46D453
#define SDLGLAPPLICATION_HPP_INCLUDED_AD5D925E_13F1_4765_9E59_CDEF4B46D453

//////////////////////////////////////////////////////////////////////////
//OpenGL extensions
//////////////////////////////////////////////////////////////////////////
#ifdef USE_OPENGL_EXTENSIONS
#if !defined(__APPLE__)
#include <GL/glew.h>
#ifdef _WIN32
#   include <GL/wglew.h>
#else
#   include <GL/glxew.h>
#endif
#endif

/* Ensure we are using opengl's core profile only */
#define GL3_PROTOTYPES 1
#if defined(__APPLE__)
#	include <OpenGL/gl3.h>
#elif !defined(_WIN32)
#	include "glcorearb.h"
#else
#	include <GL3/gl3.h>
#endif
#else
#	include "gl_core_3_2.h"
#endif

//////////////////////////////////////////////////////////////////////////
//SDL is a cross-platform windowing...
//////////////////////////////////////////////////////////////////////////
#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__) && !defined(__NVCC__) && !defined(__CUDABE__) && !defined(__CUDANVVM__)
#if !defined(__APPLE__)
#    ifdef _WIN32
#        include "../contrib/include/SDL.h"
#    else
#        include "SDL.h"
#    endif
#else
#    include <SDL2/SDL.h>
#endif //__APPLE__
#endif //HAS CUDA

#include "Application/AnimationManager.hpp"
#include "Application/CameraManager.hpp"

class RenderBug
{
	enum GeometryMode { TRIANGLES = 0, LINES = 1, POINTS = 2 };
	GeometryMode mGeometryMode;
public:
	//data buffers:
	float *positions;
	float *normals;
	float *colors;
	size_t   numPositions;
	unsigned int   *indices;
	int   numIndices;

	static GLuint vao;//Vertex Array Object
	static GLuint vbo[4]; //Vertex Buffer Objects

	static GLuint vertexshader, fragmentshader;//shaders
	static GLuint shaderprogram; //shader program

	static GLuint vertexshader_mvp, fragmentshader_cartoon, fragmentshader_constant;//more shaders
	static GLuint shaderprogram_cartoon; //shader program
	static GLuint shaderprogram_constant; //shader program

	//////////////////////////////////////////////////////////////////////////
	//Textures & related
	//GL_TEXTURE_RECTANGLE_ARB allows non-normalized texture coordinates
	static const int TEXTURE_TARGET = GL_TEXTURE_2D;
	static const int INTERNAL_FORMAT = GL_RGB32F;//GL_LUMINANCE32F_ARB;
	static const int TEXTURE_FORMAT = GL_RGB;//GL_LUMINANCE;
	static GLuint sFBTextureId;
	static GLuint sFBOId;

	RenderBug() : mGeometryMode(TRIANGLES)
	{
		positions = NULL;
		normals = NULL;
		colors = NULL;
		numPositions = 0;
		indices = NULL;
		numIndices = 0;
	}

	~RenderBug()
	{
		if (numIndices > 0)
		{
			delete[] normals;
			delete[] indices;
			numIndices = 0;
		}

		if (numPositions > 0)
		{
			delete[] colors;
			delete[] positions;
			numPositions = 0;
		}
	}

	void renderScene(const CameraManager& aCamera);

	void setupSceneGeometry(AnimationManager& aSceneManager);

	void initCartoonShader();
	void initConstantShader();
	void renderTriangles(const CameraManager& aCamera);
	void renderLines(const CameraManager& aCamera);
	void renderPoints(const CameraManager& aCamera);

	void cleanup();

	void initFBufferShader();
	void initFrameBufferTexture(const int aResX, const int aResY);
	void renderFBuffer(float*& aFBufferFloatPtr, const int aResX, const int aResY);


};

#endif // SDLGLAPPLICATION_HPP_INCLUDED_AD5D925E_13F1_4765_9E59_CDEF4B46D453

