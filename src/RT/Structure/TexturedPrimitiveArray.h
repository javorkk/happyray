#ifdef _MSC_VER
#pragma once
#endif

#ifndef TEXTUREDPRIMITIVEARRAY_H_INCLUDED_DE8E9B70_22C4_45CF_8E28_DEABAAD7C1EB
#define TEXTUREDPRIMITIVEARRAY_H_INCLUDED_DE8E9B70_22C4_45CF_8E28_DEABAAD7C1EB

#include "CUDAStdAfx.h"
#include "PrimitiveArray.h"
#include "Textures.h"

template<class tPrimitive>
class TexturedPrimitiveArray: public PrimitiveArray<tPrimitive>
{

public:


    HOST TexturedPrimitiveArray():PrimitiveArray<tPrimitive>()
    {}

    HOST void unbindVerticesTexture()
    {
        MY_CUDA_SAFE_CALL( cudaUnbindTexture( &texVertices) );
    }

    HOST void unbindIndicesTexture()
    {
        MY_CUDA_SAFE_CALL( cudaUnbindTexture( &texVertexIndices) );
    }

    HOST void bindVerticesTexture(float4*& aVertexData, const size_t aDataSize)
    {
        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        MY_CUDA_SAFE_CALL( cudaBindTexture(NULL, texVertices,
            (void*) aVertexData, chanelFormatDesc, aDataSize) );

    }

    HOST void bindIndicesTexture(uint*& aIndicesData, const size_t aDataSize)
    {
        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<uint>();

        MY_CUDA_SAFE_CALL( cudaBindTexture(NULL, texVertexIndices,
            (void*) aIndicesData, chanelFormatDesc, aDataSize) );

    }

    ////////////////////////////////////////////////////////////
    //Device Functions
    ////////////////////////////////////////////////////////////

    DEVICE tPrimitive operator[](uint aIndex) const
    {
        uint indices[tPrimitive::NUM_VERTICES];

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            indices[i] = tex1Dfetch(texVertexIndices, aIndex * tPrimitive::NUM_VERTICES + i);
            //indices[i] = indicesBufferDevicePtr[aIndex * tPrimitive::NUM_VERTICES + i];
        }

        tPrimitive result;

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            float4 tmp = tex1Dfetch(texVertices, indices[i]);
            //float4 tmp = vertexBufferDevicePtr[indices[i]];
            result.vtx[i].x = tmp.x;
            result.vtx[i].y = tmp.y;
            result.vtx[i].z = tmp.z;
        }

        return result;
    }

};//class Textured Primitive Array


#endif // TEXTUREDPRIMITIVEARRAY_H_INCLUDED_DE8E9B70_22C4_45CF_8E28_DEABAAD7C1EB
