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

#ifndef CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5
#define CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5

#include <vector_types.h>

#define _SET_MEMBER(aName, aType)					                           \
    void set##aName (aType aValue)	                                           \
    {												                           \
    m##aName = aValue;								                           \
    }

#define _GET_MEMBER(aName, aType)					                           \
    aType get##aName () const			                                       \
    {												                           \
    return m##aName ;								                           \
    }



class CameraManager
{
    static const float3 ZERO;
    float3 mPosition, mOrientation, mUp, mRotation, mRight;
    unsigned int mResX, mResY;
    float mFOV;
public:
    CameraManager(const float3& aPosition = ZERO,
         const float3& aOrientation = ZERO,
         const float3& aUp = ZERO,
         const float3& aRotation = ZERO,
         const float3& aRight = ZERO,
         unsigned int aResX = 512u,
         unsigned int aResY = 512u,
         float aFOV = 66.f):
    mPosition(aPosition),
        mOrientation(aOrientation),
        mUp(aUp),
        mRotation(aRotation),
        mRight(aRight),
        mResX(aResX),
        mResY(aResY),
        mFOV(aFOV)
    {
        //Default camera orientation
        mOrientation.z = 1.f;
        mUp.y = -1.f;
        mRight.x = 1.f;
    }

    _GET_MEMBER(Position, float3);
    _GET_MEMBER(Orientation, float3);
    _GET_MEMBER(Up, float3);
    _GET_MEMBER(Right, float3);
    _GET_MEMBER(Rotation, float3);
    _GET_MEMBER(ResX, unsigned int);
    _GET_MEMBER(ResY, unsigned int);
    _GET_MEMBER(FOV, float);

    _SET_MEMBER(Position, const float3&);
    _SET_MEMBER(Orientation, const float3&);
    _SET_MEMBER(Up, const float3&);
    _SET_MEMBER(Right, const float3&);
    _SET_MEMBER(Rotation, const float3&);
    _SET_MEMBER(ResX, unsigned int);
    _SET_MEMBER(ResY, unsigned int);
    _SET_MEMBER(FOV, float);

    //Performs rotation around all three axes with the argument angles (in degrees)
    void rotate(const float3&);
    //rotate vector around axis with argument angle (in radians)
    static float3 rotateVector( const float3& aVec , const float3& aAxis, const float aAngle);
    void moveUp(const float);
    void moveRight(const float);
    void moveForward(const float);

    void read(const char*);
    void write(const char*) const;
};

#undef _GET_MEMBER
#undef _SET_MEMBER

#endif // CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5
