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

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ALGEBRA_HPP_INCLUDED_08F20746_E9E9_452A_A9DE_8BEB2FB187AE
#define ALGEBRA_HPP_INCLUDED_08F20746_E9E9_452A_A9DE_8BEB2FB187AE


#include "CUDAStdAfx.h"
#include <vector_types.h> //float3
#include <vector_functions.h> //make_float3
//#include <math.h> //sqrtf


#define _DEF_BIN_OP3(_OP)                                                      \
DEVICE HOST float3 operator _OP (const float3& aVal1, const float3& aVal2)                   \
{                                                                              \
    float3 result;                                                             \
    result.x = aVal1.x _OP aVal2.x;                                            \
    result.y = aVal1.y _OP aVal2.y;                                            \
    result.z = aVal1.z _OP aVal2.z;                                            \
    return result;                                                             \
}                                                                              \
DEVICE HOST float3 operator _OP##= (float3& aVal1, const float3& aVal2)          \
{                                                                              \
    aVal1.x = aVal1.x _OP aVal2.x;                                             \
    aVal1.y = aVal1.y _OP aVal2.y;                                             \
    aVal1.z = aVal1.z _OP aVal2.z;                                             \
    return aVal1;                                                              \
}                                                                              \
    /*End Macro */

#define _DEF_UNARY_MINUS3                                                      \
DEVICE HOST  float3 operator- (float3 aVal)                                    \
{                                                                              \
    float3 result;                                                             \
    result.x = -aVal.x;                                                        \
    result.y = -aVal.y;                                                        \
    result.z = -aVal.z;                                                        \
    return result;                                                             \
}                                                                              \
    /*End Macro */

#define _DEF_SCALAR_OP3(_OP)                                                   \
DEVICE HOST float3 operator _OP (const float3& aVec, float aVal)                      \
{                                                                              \
    float3 result;                                                             \
    result.x = aVec.x _OP aVal;                                                \
    result.y = aVec.y _OP aVal;                                                \
    result.z = aVec.z _OP aVal;                                                \
    return result;                                                             \
}                                                                              \
DEVICE HOST float3 operator _OP##= (float3& aVec, float aV)                     \
{                                                                              \
    aVec.x _OP##= aV; aVec.y _OP##= aV; aVec.z _OP##= aV;                      \
    return aVec;                                                               \
}

#define _DEF_SCALAR_OP3_SYM(_OP)                                               \
DEVICE HOST  float3 operator _OP (float aVal, const float3& aVec)                     \
{                                                                              \
    float3 result;                                                             \
    result.x = aVal _OP aVec.x;                                                \
    result.y = aVal _OP aVec.y;                                                \
    result.z = aVal _OP aVec.z;                                                \
    return result;                                                             \
}                                                                              \
    /*End Macro*/

    _DEF_BIN_OP3(+) //Component-wise +
    _DEF_BIN_OP3(-) //Component-wise -
    _DEF_BIN_OP3(*) //Component-wise *
    _DEF_BIN_OP3(/) //Component-wise /

    _DEF_SCALAR_OP3(*) //Vector * scalar -> Vector and Vector *= scalar
    _DEF_SCALAR_OP3_SYM(*) //scalar * Vector -> Vector
    _DEF_SCALAR_OP3(/) //Vector / scalar and Vector /= scalar

    //An unary minus
    _DEF_UNARY_MINUS3

    //Make a vector with equal components
    DEVICE HOST float3 rep (float aVal)
    {
        float3 result;
        result.x = result.y = result.z = aVal;
        return result;
    }

    //Access a vector as array
    DEVICE HOST float* toPtr(float3& aVec)                       
    {                                                              
        return reinterpret_cast<float*>(&aVec);  
    }                          
    DEVICE HOST const float* toPtr(const float3& aVec)                       
    {                                                              
        return reinterpret_cast<const float*>(&aVec);  
    }  

    DEVICE HOST float3 fastDivide(const float3 aVec1, const float3 aVec2)
    {
#ifdef __CUDA_ARCH__
        float3 retval;
        retval.x = __fdividef(aVec1.x, aVec2.x);
        retval.y = __fdividef(aVec1.y, aVec2.y); 
        retval.z = __fdividef(aVec1.z, aVec2.z);
        return retval;
#else
        return aVec1 / aVec2;
#endif
    }

    DEVICE HOST float3 fastDivide(const float3& aVec1, const float& aVal)
    {
#ifdef __CUDA_ARCH__
        float3 retval;
        retval.x = __fdividef(aVec1.x, aVal);
        retval.y = __fdividef(aVec1.y, aVal); 
        retval.z = __fdividef(aVec1.z, aVal);
        return retval;
#else
        return aVec1 / aVal;
#endif
    }

    //dot product 
    DEVICE HOST float dot(const float3& aVec1, const float3& aVec2)
    {
        return aVec1.x * aVec2.x + aVec1.y * aVec2.y + aVec1.z * aVec2.z;
    }

    //dot product with the reciprocal of this
    DEVICE HOST float dotRCP(const float3& aVec1, const float3& aVec2)
    {
#ifdef __CUDA_ARCH__
        return  __fdividef(aVec1.x, aVec2.x) +
            __fdividef(aVec1.y, aVec2.y) +
            __fdividef(aVec1.z, aVec2.z);
#else
        return aVec1.x / aVec2.x + aVec1.y / aVec2.y + aVec1.z / aVec2.z;
#endif
    }

    //Cross product on the first three components
    DEVICE HOST float3 cross(const float3& aVec1, const float3& aVec2)
    {
        float3 retval;
        retval.x = aVec1.y * aVec2.z - aVec1.z * aVec2.y;
        retval.y = aVec1.z * aVec2.x - aVec1.x * aVec2.z;
        retval.z = aVec1.x * aVec2.y - aVec1.y * aVec2.x;
        return retval;
    }

    //Cross product with the reciprocal of this
    DEVICE HOST float3 crossRCP(const float3& aVec1, const float3& aVec2)
    {
        float3 retval;
#ifdef __CUDA_ARCH__
        retval.x = __fdividef(aVec2.z, aVec1.y) - __fdividef(aVec2.y, aVec1.z);
        retval.y = __fdividef(aVec2.x, aVec1.z) - __fdividef(aVec2.z, aVec1.x); 
        retval.z = __fdividef(aVec2.y, aVec1.x) - __fdividef(aVec2.x, aVec1.y);
#else
        retval.x = aVec2.z / aVec1.y - aVec2.y / aVec1.z;
        retval.y = aVec2.x / aVec1.z - aVec2.z / aVec1.x;
        retval.z = aVec2.y / aVec1.x - aVec2.x / aVec1.y;
#endif
        return retval;
    }

    //A component-wise maximum between two float3s
    DEVICE HOST float3 min(const float3 & aVec1, const float3 & aVec2)
    {
#ifdef __CUDA_ARCH__
        float3 retval;
        retval.x = fminf(aVec1.x, aVec2.x);
        retval.y = fminf(aVec1.y, aVec2.y); 
        retval.z = fminf(aVec1.z, aVec2.z);
        return retval;
#else
        float3 retval;
        retval.x = cudastd::min(aVec1.x, aVec2.x);
        retval.y = cudastd::min(aVec1.y, aVec2.y); 
        retval.z = cudastd::min(aVec1.z, aVec2.z);
        return retval;
#endif
    }

    DEVICE HOST float3 max(const float3 & aVec1, const float3 & aVec2)
    {
#ifdef __CUDA_ARCH__
        float3 retval;
        retval.x = fmaxf(aVec1.x, aVec2.x);
        retval.y = fmaxf(aVec1.y, aVec2.y); 
        retval.z = fmaxf(aVec1.z, aVec2.z);
        return retval;
#else
        float3 retval;
        retval.x = cudastd::max(aVec1.x, aVec2.x);
        retval.y = cudastd::max(aVec1.y, aVec2.y); 
        retval.z = cudastd::max(aVec1.z, aVec2.z);
        return retval;
#endif
    }


    //Length of the vector
    DEVICE HOST float len(const float3& aVec)
    {
        return sqrtf(dot(aVec,aVec));
    }

    DEVICE HOST float lenRCP(const float3& aVec)
    {
#ifdef __CUDA_ARCH__
        return rsqrtf(dot(aVec,aVec));
#else
        return 1.f / sqrtf(dot(aVec,aVec));
#endif
    }

    //computes orthogonal local coordinate system
    DEVICE HOST void getLocalCoordinates(
        const float3& aNormal,
        float3& oTangent,
        float3& oBinormal)
    {
        const int cId0  = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? 0 : 1;
        const int cId1  = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? 1 : 0;
        const float sig = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? -1.f : 1.f;

        const float invLen = 1.f / (toPtr(aNormal)[cId0] * toPtr(aNormal)[cId0] +
            aNormal.z * aNormal.z);

        toPtr(oTangent)[cId0] = aNormal.z * sig * invLen;
        toPtr(oTangent)[cId1] = 0.f;
        oTangent.z   = toPtr(aNormal)[cId0] * -1.f * sig * invLen;

        oBinormal = cross(aNormal, oTangent);
    }

    //Cross product
    DEVICE HOST float3 operator %(const float3& aVec1, const float3& aVec2)
    {
        return cross(aVec1, aVec2);
    }

    //A normalized vector: V / |V|
    DEVICE HOST float3 operator ~(const float3& aVec)
    {
        return aVec * lenRCP(aVec);
    }

#undef _DEF_BIN_OP3
#undef _DEF_UNARY_MINUS3
#undef _DEF_SCALAR_OP3
#undef _DEF_SCALAR_OP3_SYM

    struct quaternion4f
    {
        float x, y, z, w;

        DEVICE HOST quaternion4f(float aX, float aY, float aZ, float aW) :x(aX), y(aY), z(aZ), w(aW) {}

        DEVICE HOST quaternion4f(
            float aMxx, float aMyx, float aMzx,
            float aMxy, float aMyy, float aMzy,
            float aMxz, float aMyz, float aMzz
            )
        {
            const float t = aMxx + aMyy + aMzz;
            if (t > 0.f)
            {
                const float r = sqrtf(1.f + t);
                const float s = 0.5f / r;
                w = 0.5f * r;
                x = (aMzy - aMyz) * s;
                y = (aMxz - aMzx) * s;
                z = (aMyx - aMxy) * s;
            }
            else if (aMxx >= aMyy && aMxx >= aMzz)
            {
                const float r = sqrtf(1.f + aMxx - aMyy - aMzz);
                const float s = 0.5f / r;
                w = (aMzy - aMyz) * s;
                x = 0.5f * r; 
                y = (aMxz - aMzx) * s;
                z = (aMyx - aMxy) * s;

            }
            else if (aMyy >= aMxx && aMyy >= aMzz)
            {
                const float r = sqrtf(1.f + aMyy - aMxx - aMzz);
                const float s = 0.5f / r;
                w = (aMxz - aMzx) * s;
                x = (aMzy - aMyz) * s;
                y = 0.5f * r; 
                z = (aMyx - aMxy) * s;
            }
            else if (aMzz >= aMxx && aMzz >= aMyy)
            {
                const float r = sqrtf(1.f + aMzz - aMxx - aMyy);
                const float s = 0.5f / r;
                w = (aMyx - aMxy) * s; 
                x = (aMzy - aMyz) * s;
                y = (aMxz - aMzx) * s;
                z = 0.5f * r; 
            }
        }

        DEVICE HOST float3 operator ()(const float3& aVec) const
        {
            //const float ww = w * w;
            const float wx = w * x;
            const float wy = w * y;
            const float wz = w * z;

            const float xx = x * x;
            const float xy = x * y;
            const float xz = x * z;
            

            const float yy = y * y;
            const float yz = y * z;
            

            const float zz = z * z;


            return make_float3(
                aVec.x - 2.f * (yy + xx) * aVec.x + 2.f * (xy - wz) * aVec.y + 2.f * (xz + wy) * aVec.z,
                aVec.y + 2.f * (xy + wz) * aVec.x - 2.f * (xx + zz) * aVec.y + 2.f * (yz - wx) * aVec.z,
                aVec.z + 2.f * (xz - wy) * aVec.x + 2.f * (yz + wx) * aVec.y - 2.f * (xx + yy) * aVec.z
                );

        }

        DEVICE HOST void toMatrix3f(
            float& oMxx, float& oMyx, float& oMzx,
            float& oMxy, float& oMyy, float& oMzy,
            float& oMxz, float& oMyz, float& oMzz) const
        {
            //const float ww = w * w;
            const float wx = w * x;
            const float wy = w * y;
            const float wz = w * z;

            const float xx = x * x;
            const float xy = x * y;
            const float xz = x * z;


            const float yy = y * y;
            const float yz = y * z;


            const float zz = z * z;

            oMxx = 2.f * (yy + xx); oMyx = 2.f * (xy - wz); oMzx = 2.f * (xz + wy);
            oMxy = 2.f * (xy + wz); oMyy = 2.f * (xx + zz); oMzy = 2.f * (yz - wx);
            oMxz = 2.f * (xz - wy); oMyz = 2.f * (yz + wx); oMzz = 2.f * (xx + yy);
            oMxx = 1 - oMxx;
            oMyy = 1 - oMyy;
            oMzz = 1 - oMzz;
        }

    };

    DEVICE HOST quaternion4f make_quaternion4f(float aX, float aY, float aZ, float aW) { return quaternion4f(aX, aY, aZ, aW); }

    DEVICE HOST float magnitudeSQR(const quaternion4f& aQ){ return aQ.x * aQ.x + aQ.y * aQ.y + aQ.z * aQ.z + aQ.w * aQ.w; }

    DEVICE HOST float magnitude(const quaternion4f& aQ){ return sqrtf(magnitudeSQR(aQ)); }    

    DEVICE HOST quaternion4f operator /(const quaternion4f& aQ, float aS) { return quaternion4f(aQ.x / aS, aQ.y / aS, aQ.z / aS, aQ.w / aS); }
    DEVICE HOST quaternion4f operator *(const quaternion4f& aQ, float aS) { return quaternion4f(aQ.x * aS, aQ.y * aS, aQ.z * aS, aQ.w * aS); }

    DEVICE HOST quaternion4f operator ~(const quaternion4f& aQ) { return aQ / magnitude(aQ); }

    DEVICE HOST quaternion4f operator *(const quaternion4f& aQ1, const quaternion4f& aQ2)
    { 
        return quaternion4f(
            aQ1.w * aQ2.x + aQ1.x * aQ2.w + aQ1.y * aQ2.z - aQ1.z * aQ2.y,
            aQ1.w * aQ2.y + aQ1.y * aQ2.w + aQ1.z * aQ2.x - aQ1.x * aQ2.z,
            aQ1.w * aQ2.z + aQ1.z * aQ2.w + aQ1.x * aQ2.y - aQ1.y * aQ2.x,
            aQ1.w * aQ2.w - aQ1.x * aQ2.x - aQ1.y * aQ2.y - aQ1.z * aQ2.z
            );
    }

    DEVICE HOST bool isIdentity(const quaternion4f& aQ, float aEPS)
    {
        return fabsf(aQ.x) + fabsf(aQ.y) + fabsf(aQ.z) < aEPS && fabsf(aQ.w - 1.f) < aEPS;
    }


#endif // ALGEBRA_HPP_INCLUDED_08F20746_E9E9_452A_A9DE_8BEB2FB187AE
