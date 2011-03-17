#ifdef _MSC_VER
#pragma once
#endif

#ifndef PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0
#define PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0

#include "CUDAStdAfx.h"
#include "Textures.h"

template<class Primitive>
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
        normalIndicesBufferSize(0u)
    {}

    HOST ~PrimitiveArray()
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120
        MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(normalBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(normalIndicesBufferHostPtr) );
#else
        MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(normalBufferHostPtr) );
        MY_CUDA_SAFE_CALL( cudaFreeHost(normalIndicesBufferHostPtr) );

        MY_CUDA_SAFE_CALL( cudaFree(vertexBufferDevicePtr) );
        MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );
        MY_CUDA_SAFE_CALL( cudaFree(normalBufferDevicePtr) );
        MY_CUDA_SAFE_CALL( cudaFree(normalIndicesBufferDevicePtr) );
#endif
    }

    HOST DEVICE size_t getMemorySize()
    {
        return vertexBufferSize + indicesBufferSize + normalBufferSize;
    }

    HOST void allocateMappedDeviceArray(void** aDevicePtr, void** aHostPtr, size_t aSize,
        void** aOldDevicePtr, void** aOldHostPtr, size_t& aOldSize)
    {
        if (aOldSize < aSize)
        {
            aOldSize = aSize;
            MY_CUDA_SAFE_CALL( cudaHostAlloc(aOldHostPtr,aSize, cudaHostAllocMapped) );
        }

        MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer(aOldDevicePtr, *aOldHostPtr, 0));
        *aDevicePtr = *aOldDevicePtr;
        *aHostPtr = *aOldHostPtr;
    }

    HOST void allocateDeviceArray(void** aPtr, size_t aSize,
        void** aOldPtr, size_t& aOldSize)
    {
        if (aOldSize < aSize)
        {
            aOldSize = aSize;
            CUDA_SAFE_CALL( cudaMalloc(aOldPtr, aSize));
        }

        *aPtr = *aOldPtr;
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

    DEVICE Primitive operator[](uint aIndex) const
    {
        uint indices[Primitive::NUM_VERTICES];

#pragma unroll Primitive::NUM_VERTICES
        for(uint i = 0; i < Primitive::NUM_VERTICES; ++i)
        {
            float4 tmp = tex1Dfetch(texVertexIndices, aIndex * Primitive::NUM_VERTICES + i);
            indices[i].x = tmp.x;
            indices[i].y = tmp.y;
            indices[i].z = tmp.z;
        }

        Primitive result;

#pragma unroll Primitive::NUM_VERTICES
        for(uint i = 0; i < Primitive::NUM_VERTICES; ++i)
        {
            result.vtx[i] = tex1Dfetch(texVertices, indices[i]);
        }

        return result;
    }

    //The normals are stored in the vertex array of the returned primitive
    DEVICE Primitive getVertexNormals(uint aIndex)
    {
        uint indices[Primitive::NUM_VERTICES];

#pragma unroll Primitive::NUM_VERTICES
        for(uint i = 0; i < Primitive::NUM_VERTICES; ++i)
        {
            indices[i] = normalIndicesBufferDevicePtr[aIndex * Primitive::NUM_VERTICES + i]; 
        }

        Primitive result;

#pragma unroll Primitive::NUM_VERTICES
        for(uint i = 0; i < Primitive::NUM_VERTICES; ++i)
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
