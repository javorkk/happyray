#include "StdAfx.hpp"
#include "GLUtilities.hpp"

#ifndef _WIN32
#   include <math.h>
#endif

/* Multiply 4x4 matrix m1 by 4x4 matrix m2 and store the result in m1 */
void multiply4x4(GLfloat *m1, GLfloat *m2)
{
    GLfloat temp[16];

    int x,y;

    for (x=0; x < 4; x++)
    {
        for(y=0; y < 4; y++)
        {
            temp[y + (x*4)] = (m1[x*4] * m2[y]) +
                              (m1[(x*4)+1] * m2[y+4]) +
                              (m1[(x*4)+2] * m2[y+8]) +
                              (m1[(x*4)+3] * m2[y+12]);
        }
    }

    memcpy(m1, temp, sizeof(GLfloat) << 4);
}

/* Generate a perspective view matrix using a field of view angle fov,
 * window aspect ratio, near and far clipping planes */
void perspective(GLfloat *matrix, GLfloat fov, GLfloat aspect, GLfloat nearz, GLfloat farz)
{
    GLfloat range;

    range = tanf(fov * 0.00872664625f) * nearz; /* 0.00872664625 = PI/360 */
    memset(matrix, 0, sizeof(GLfloat) * 16);
    matrix[0] = (2.f * nearz) / ((range * aspect) - (-range * aspect));
    matrix[5] = (2.f * nearz) / (2.f * range);
    matrix[10] = -(farz + nearz) / (farz - nearz);
    matrix[11] = -1.f;
    matrix[14] = -(2.f * farz * nearz) / (farz - nearz);
}

/* Perform translation operations on a matrix */
void translate(GLfloat *matrix, GLfloat x, GLfloat y, GLfloat z)
{
    GLfloat newmatrix[16] = IDENTITY_MATRIX4;

    newmatrix[12] = x;
    newmatrix[13] = y;
    newmatrix[14] = z;

    multiply4x4(matrix, newmatrix);
}

/* Rotate a matrix by an angle on a X, Y, or Z axis */
void rotate(GLfloat *matrix, GLfloat angle, AXIS axis)
{
    const GLfloat d2r = 0.0174532925199f; /* PI / 180 */
    const int cos1[3] = { 5, 0, 0 };
    const int cos2[3] = { 10, 10, 5 };
    const int sin1[3] = { 6, 2, 1 };
    const int sin2[3] = { 9, 8, 4 };
    GLfloat newmatrix[16] = IDENTITY_MATRIX4;

    newmatrix[cos1[axis]] = cosf(d2r * angle);
    newmatrix[sin1[axis]] = -sinf(d2r * angle);
    newmatrix[sin2[axis]] = -newmatrix[sin1[axis]];
    newmatrix[cos2[axis]] = newmatrix[cos1[axis]];

    multiply4x4(matrix, newmatrix);
}


