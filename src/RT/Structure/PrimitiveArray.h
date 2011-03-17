#ifdef _MSC_VER
#pragma once
#endif

#ifndef PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0
#define PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0

#include "CUDAStdAfx.h"
#include "Textures.h"

template<class tPrimitive>
class PrimitiveArray
{

public:
    /////////////////////////////////////////////////////////////
    //Memory Managment
    /////////////////////////////////////////////////////////////
    float4* vertexBufferDevicePtr;
    float4* vertexBufferHostPtr;
    size_t  vertexBufferSize;

    uint*  indicesBufferDevicePtr;
    uint*  indicesBufferHostPtr;
    size_t indicesBufferSize;

    float3* normalBufferDevicePtr;
    float3* normalBufferHostPtr;
    size_t  normalBufferSize;

    uint*  normalIndicesBufferDevicePtr;
    uint*  normalIndicesBufferHostPtr;
    size_t normalIndicesBufferSize;

    size_t numPrimitives;

    HOST PrimitiveArray():
        vertexBufferDevicePtr(NULL),
        vertexBufferHostPtr  (NULL),
        vertexBufferSize(0u),
        indicesBufferDevicePtr(NULL),
        indicesBufferHostPtr  (NULL),
        indicesBufferSize(0u),
        normalBufferDevicePtr(NULL),
        normalBufferHostPtr  (NULL),
        normalBufferSize(0u),
        normalIndicesBufferDevicePtr(NULL),
        normalIndicesBufferHostPtr  (NULL),
        normalIndicesBufferSize(0u),
        numPrimitives(0u)
    {}

    HOST ~PrimitiveArray()
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120
        if(vertexBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
        if(normalBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(normalBufferHostPtr) );
        if(normalIndicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(normalIndicesBufferHostPtr) );
#else
        if(vertexBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
        if(normalBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(normalBufferHostPtr) );
        if(normalIndicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(normalIndicesBufferHostPtr) );

        if(vertexBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(vertexBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );
        if(normalBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(normalBufferDevicePtr) );
        if(normalIndicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(normalIndicesBufferDevicePtr) );
#endif
    }

    HOST DEVICE size_t getMemorySize()
    {
        return vertexBufferSize + indicesBufferSize + normalBufferSize;
    }

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

//#pragma unroll tPrimitive::NUM_VERTICES
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            indices[i] = tex1Dfetch(texVertexIndices, aIndex * tPrimitive::NUM_VERTICES + i);
        }

        tPrimitive result;

//#pragma unroll tPrimitive::NUM_VERTICES
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
             float4 tmp = tex1Dfetch(texVertices, indices[i]);
             result.vtx[i].x = tmp.x;
             result.vtx[i].y = tmp.y;
             result.vtx[i].z = tmp.z;
        }

        return result;
    }

    //The normals are stored in the vertex array of the returned primitive
    DEVICE tPrimitive getVertexNormals(uint aIndex)
    {
        uint indices[tPrimitive::NUM_VERTICES];

//#pragma unroll tPrimitive::NUM_VERTICES
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            indices[i] = normalIndicesBufferDevicePtr[aIndex * tPrimitive::NUM_VERTICES + i]; 
        }

        tPrimitive result;

//#pragma unroll tPrimitive::NUM_VERTICES
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            result.vtx[i] = normalBufferDevicePtr[indices[i]];
        }

        return result;
    }

};//class Primitive Array


#include "RT/Primitive/Primitive.hpp"
#include "Application/WFObject.hpp"

class ObjUploader
{
public:
    //Vertex coordinates
    HOST void uploadObjFrameVertexData(
        const WFObject& aKeyFrame1,
        const WFObject& aKeyFrame2,
        const float     aCoeff,
        float3&         oMinBound,
        float3&         oMaxBound,
        PrimitiveArray<Primitive<3> >& aArray
        );
    //Normal coordinates
    HOST void uploadObjFrameNormalData(
        const WFObject& aKeyFrame1,
        const WFObject& aKeyFrame2,
        const float     aCoeff,
        PrimitiveArray<Primitive<3> >& aArray
        );

    //Vertex and normal indices
    HOST void uploadObjFrameIndexData(
        const WFObject& aKeyFrame1,
        const WFObject& aKeyFrame2,
        PrimitiveArray<Primitive<3> >& aArray
        );
};//class ObjUploader

#endif // PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0
