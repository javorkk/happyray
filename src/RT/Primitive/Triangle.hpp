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

#ifndef TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035
#define TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035


#include "CUDAStdAfx.h"
#include "RT/Primitive/Primitive.hpp"

typedef Primitive<3> Triangle;

struct ShevtsovTriAccel
{
    float   nu;
    float   nv;
    float   np;
    float   pu;
    float   pv;
    float   e0u;
    float   e0v;
    float   e1u;
    float   e1v;
    int    dimW;
    int    dimU;
    int    dimV;

    ShevtsovTriAccel(){}

    ShevtsovTriAccel(Triangle aTriangle);
};

struct WoopTriAccel
{
    float data[12];

    WoopTriAccel(){}

    WoopTriAccel(const Triangle& aTriangle);
};

#endif // TRIANGLE_HPP_INCLUDED_C96DCCD7_B8E0_4CF2_8AD0_AB9362B10035
