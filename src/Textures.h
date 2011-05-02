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

#ifndef TEXTURES_H_INCLUDED_8C40E109_E13D_496F_92E1_D0C6F600C00C
#define TEXTURES_H_INCLUDED_8C40E109_E13D_496F_92E1_D0C6F600C00C

#include <cuda_runtime_api.h>

//Declare textures here and include this in all files that use CUDA textures

//////////////////////////////////////////////////////////////////////////////
//Used in RT/Structure/PrimitiveArray
//////////////////////////////////////////////////////////////////////////////

//Vertex buffers
texture< float4, 1, cudaReadModeElementType >   texVertices; //stream 0
//texture< float3, 1, cudaReadModeElementType >   texVertices1; //stream 1
//texture< float3, 1, cudaReadModeElementType >   texVertices2; //stream 2
//Index buffers
texture< uint, 1, cudaReadModeElementType >   texVertexIndices; //stream 0
//texture< uint, 1, cudaReadModeElementType >   texVertexIndices1; //stream 1
//texture< uint, 1, cudaReadModeElementType >   texVertexIndices2; //stream 2
//Pre-computed Structures
texture< float4, 1, cudaReadModeElementType >   texFaceAccel;

/////////////////////////////////////////////////////////////////////////////
//Used in RT/Structure/UniformGrid
/////////////////////////////////////////////////////////////////////////////
texture< uint2, 3, cudaReadModeElementType >        texGridCells;

#endif // TEXTURES_H_INCLUDED_8C40E109_E13D_496F_92E1_D0C6F600C00C
