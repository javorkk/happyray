#ifdef _MSC_VER
#pragma once
#endif

#ifndef UTILITIES_H_INCLUDED_7BF7CC28_6F73_42D7_A8A1_950C238D7DB8
#define UTILITIES_H_INCLUDED_7BF7CC28_6F73_42D7_A8A1_950C238D7DB8

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

#define EMPTY_MATRIX4  {\
    0.0, 0.0, 0.0, 0.0,\
    0.0, 0.0, 0.0, 0.0,\
    0.0, 0.0, 0.0, 0.0,\
    0.0, 0.0, 0.0, 0.0 }

#define IDENTITY_MATRIX4 {\
    1.0, 0.0, 0.0, 0.0,\
    0.0, 1.0, 0.0, 0.0,\
    0.0, 0.0, 1.0, 0.0,\
    0.0, 0.0, 0.0, 1.0 }

typedef enum {
    X_AXIS,
    Y_AXIS,
    Z_AXIS
} AXIS;

/* Multiply 4x4 matrix m1 by 4x4 matrix m2 and store the result in m1 */
void multiply4x4(GLfloat *m1, GLfloat *m2);

/* Generate a perspective view matrix using a field of view angle fov,
* window aspect ratio, near and far clipping planes */
void perspective(GLfloat *matrix, GLfloat fov, GLfloat aspect, GLfloat nearz, GLfloat farz);

/* Perform translation operations on a matrix */
void translate(GLfloat *matrix, GLfloat x, GLfloat y, GLfloat z);

/* Rotate a matrix by an angle on a X, Y, or Z axis specified by the AXIS enum*/
void rotate(GLfloat *matrix, GLfloat angle, AXIS axis);


#endif // UTILITIES_H_INCLUDED_7BF7CC28_6F73_42D7_A8A1_950C238D7DB8
