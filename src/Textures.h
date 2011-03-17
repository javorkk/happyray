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
