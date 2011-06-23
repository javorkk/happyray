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

    size_t numPrimitives;

    HOST PrimitiveArray():
        vertexBufferDevicePtr(NULL),
        vertexBufferHostPtr  (NULL),
        vertexBufferSize(0u),
        indicesBufferDevicePtr(NULL),
        indicesBufferHostPtr  (NULL),
        indicesBufferSize(0u),
        numPrimitives(0u)
    {}

    HOST void cleanup()
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120
        if(vertexBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
#else
        if(vertexBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );

        if(vertexBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(vertexBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );
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

};//class Primitive Array

template<class tPrimitive, class tType>
class AttribStruct
{
public:
    tType data[tPrimitive::NUM_VERTICES];
};

template<class tPrimitive, class tType>
class PrimitiveAttributeArray
{
public:
    typedef AttribStruct<tPrimitive, tType> t_AttribStruct;

    /////////////////////////////////////////////////////////////
    //Memory Managment
    /////////////////////////////////////////////////////////////

    float3* dataBufferDevicePtr;
    float3* dataBufferHostPtr;
    size_t  dataBufferSize;

    uint*  indicesBufferDevicePtr;
    uint*  indicesBufferHostPtr;
    size_t indicesBufferSize;

    size_t numPrimitives;

    HOST PrimitiveAttributeArray():
        dataBufferDevicePtr(NULL),
        dataBufferHostPtr  (NULL),
        dataBufferSize(0u),
        indicesBufferDevicePtr(NULL),
        indicesBufferHostPtr  (NULL),
        indicesBufferSize(0u),
        numPrimitives(0u)
    {}

    HOST void cleanup()
    {
#if HAPPYRAY__CUDA_ARCH__ >= 120
        if(dataBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(dataBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );
#else
        if(dataBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(dataBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );

        if(dataBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(dataBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );
#endif
    }

    HOST DEVICE size_t getMemorySize()
    {
        return dataBufferSize + indicesBufferSize;
    }


    ////////////////////////////////////////////////////////////
    //Device Functions
    ////////////////////////////////////////////////////////////
 
    DEVICE t_AttribStruct  operator[](uint aIndex) const
    {
        uint indices[tPrimitive::NUM_VERTICES];

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            indices[i] = indicesBufferDevicePtr[aIndex * tPrimitive::NUM_VERTICES + i]; 
        }

        t_AttribStruct result;

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
            result.data[i] = dataBufferDevicePtr[indices[i]];
        }

        return result;
    }

};//class Primitive Attribute Array

#include "RT/Primitive/Primitive.hpp"
#include "Application/WFObject.hpp"
#include "RT/Structure/MemoryManager.h"

class ObjUploader
{
public:
    ////Vertex coordinates
    //HOST void uploadObjFrameVertexData(
    //    const WFObject& aKeyFrame1,
    //    const WFObject& aKeyFrame2,
    //    const float     aCoeff,
    //    float3&         oMinBound,
    //    float3&         oMaxBound,
    //    PrimitiveArray<Primitive<3> >& aArray
    //    );
    ////Normal coordinates
    //HOST void uploadObjFrameNormalData(
    //    const WFObject& aKeyFrame1,
    //    const WFObject& aKeyFrame2,
    //    const float     aCoeff,
    //    PrimitiveAttributeArray<Primitive<3>, float3 >& aArray
    //    );

    ////Vertex indices
    //HOST void uploadObjFrameVertexIndexData(
    //    const WFObject& aKeyFrame1,
    //    const WFObject& aKeyFrame2,
    //    PrimitiveArray<Primitive<3> >& aArray
    //    );

    ////Normal indices
    //HOST void uploadObjFrameNormalIndexData(
    //    const WFObject& aKeyFrame1,
    //    const WFObject& aKeyFrame2,
    //    PrimitiveAttributeArray<Primitive<3>, float3 >& aArray
    //    );
    HOST void ObjUploader::uploadObjFrameVertexData(
    const WFObject& aKeyFrame1,
    const WFObject& aKeyFrame2,
    const float     aCoeff,
    float3&         oMinBound,
    float3&         oMaxBound,
    PrimitiveArray<Primitive<3> >& aArray)
{
    const size_t numVertices1 = aKeyFrame1.getNumVertices();
    const size_t numVertices2 = aKeyFrame2.getNumVertices();
    const size_t numVertices  = numVertices2;
    const size_t verticesNewSize = numVertices * sizeof(float4);

    ////////////////////////////////////////////////////////////
    //cleanup
    ////////////////////////////////////////////////////////////
    aArray.unbindVerticesTexture();
    MemoryManager::allocateHostDeviceArrayPair(
        (void**)&aArray.vertexBufferDevicePtr,
        (void**)&aArray.vertexBufferHostPtr,
        verticesNewSize,
        (void**)&aArray.vertexBufferDevicePtr,
        (void**)&aArray.vertexBufferHostPtr,
        aArray.vertexBufferSize);

    float4* verticesHost = aArray.vertexBufferHostPtr;
    float4* verticesDevice = aArray.vertexBufferDevicePtr;
    
    //////////////////////////////////////////////////////////////////////////
    //Copy and transfer vertex data
    //////////////////////////////////////////////////////////////////////////
    size_t it = 0;
    for (; it < cudastd::min(numVertices1, numVertices2); ++it)
    {
        verticesHost[it].x = aKeyFrame1.getVertex(it).x * (1.f - aCoeff) + aKeyFrame2.getVertex(it).x * aCoeff;
        verticesHost[it].y = aKeyFrame1.getVertex(it).y * (1.f - aCoeff) + aKeyFrame2.getVertex(it).y * aCoeff;
        verticesHost[it].z = aKeyFrame1.getVertex(it).z * (1.f - aCoeff) + aKeyFrame2.getVertex(it).z * aCoeff;
        verticesHost[it].w = 0.f;

        oMinBound.x = cudastd::min(verticesHost[it].x, oMinBound.x);
        oMinBound.y = cudastd::min(verticesHost[it].y, oMinBound.y);
        oMinBound.z = cudastd::min(verticesHost[it].z, oMinBound.z);
        oMaxBound.x = cudastd::max(verticesHost[it].x, oMaxBound.x);
        oMaxBound.y = cudastd::max(verticesHost[it].y, oMaxBound.y);
        oMaxBound.z = cudastd::max(verticesHost[it].z, oMaxBound.z);
    }

    for (; it < numVertices2 ; ++it)
    {
        verticesHost[it].x = aKeyFrame2.getVertex(it).x;
        verticesHost[it].y = aKeyFrame2.getVertex(it).y;
        verticesHost[it].z = aKeyFrame2.getVertex(it).z;
        verticesHost[it].w = 0.f;

        oMinBound.x = cudastd::min(verticesHost[it].x, oMinBound.x);
        oMinBound.y = cudastd::min(verticesHost[it].y, oMinBound.y);
        oMinBound.z = cudastd::min(verticesHost[it].z, oMinBound.z);
        oMaxBound.x = cudastd::max(verticesHost[it].x, oMaxBound.x);
        oMaxBound.y = cudastd::max(verticesHost[it].y, oMaxBound.y);
        oMaxBound.z = cudastd::max(verticesHost[it].z, oMaxBound.z);
    }
    
    MY_CUDA_SAFE_CALL( cudaMemcpy( verticesDevice, verticesHost, verticesNewSize, cudaMemcpyHostToDevice) );


    aArray.bindVerticesTexture(aArray.vertexBufferDevicePtr, aArray.vertexBufferSize);
}

HOST void ObjUploader::uploadObjFrameNormalData(
    const WFObject& aKeyFrame1,
    const WFObject& aKeyFrame2,
    const float     aCoeff,
    PrimitiveAttributeArray<Primitive<3>, float3 >& aArray)
{
    const size_t numNormals1 = aKeyFrame1.getNumNormals();
    const size_t numNormals2 = aKeyFrame2.getNumNormals();
    const size_t numNormals  = numNormals2;
    const size_t normalsNewSize = numNormals * sizeof(float3);

    ////////////////////////////////////////////////////////////
    //cleanup
    ////////////////////////////////////////////////////////////
    MemoryManager::allocateHostDeviceArrayPair(
        (void**)&aArray.dataBufferDevicePtr,
        (void**)&aArray.dataBufferHostPtr,
        normalsNewSize,
        (void**)&aArray.dataBufferDevicePtr,
        (void**)&aArray.dataBufferHostPtr,
        aArray.dataBufferSize);

    float3* normalsHost = aArray.dataBufferHostPtr;
    float3* normalsDevice = aArray.dataBufferDevicePtr;
    
    //////////////////////////////////////////////////////////////////////////
    //Copy and transfer normal data
    //////////////////////////////////////////////////////////////////////////
    size_t it = 0;
    for (; it < cudastd::min(numNormals1, numNormals2); ++it)
    {
        normalsHost[it].x = aKeyFrame1.getNormal(it).x * (1.f - aCoeff) + aKeyFrame2.getNormal(it).x * aCoeff;
        normalsHost[it].y = aKeyFrame1.getNormal(it).y * (1.f - aCoeff) + aKeyFrame2.getNormal(it).y * aCoeff;
        normalsHost[it].z = aKeyFrame1.getNormal(it).z * (1.f - aCoeff) + aKeyFrame2.getNormal(it).z * aCoeff;
    }

    for (; it < numNormals2 ; ++it)
    {
        normalsHost[it].x = aKeyFrame2.getNormal(it).x;
        normalsHost[it].y = aKeyFrame2.getNormal(it).y;
        normalsHost[it].z = aKeyFrame2.getNormal(it).z;
    }

    MY_CUDA_SAFE_CALL( cudaMemcpy( normalsDevice, normalsHost, normalsNewSize, cudaMemcpyHostToDevice) );

}

HOST void ObjUploader::uploadObjFrameVertexIndexData(
    const WFObject& aKeyFrame1,
    const WFObject& aKeyFrame2,
    PrimitiveArray<Primitive<3> >& aArray)
{
    aArray.numPrimitives = aKeyFrame1.getNumFaces();
    const size_t numIndices1 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices2 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices  = numIndices2;
    const size_t indicesNewSize = numIndices * sizeof(uint);

    ////////////////////////////////////////////////////////////
    //cleanup
    ////////////////////////////////////////////////////////////
    aArray.unbindIndicesTexture();
    MemoryManager::allocateHostDeviceArrayPair(
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        indicesNewSize,
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        aArray.indicesBufferSize);

    uint* indicesHost = aArray.indicesBufferHostPtr;
    uint* indicesDevice = aArray.indicesBufferDevicePtr;

    
    //////////////////////////////////////////////////////////////////////////
    //Copy and transfer indices
    //////////////////////////////////////////////////////////////////////////
    size_t it = 0;
    for (; it < cudastd::min(numIndices1, numIndices2); ++it)
    {
        indicesHost[it]       = aKeyFrame1.getVertexIndex(it);
    }
    for (; it < numIndices2 ; ++it)
    {
        indicesHost[it]       = aKeyFrame2.getVertexIndex(it);
    }

    MY_CUDA_SAFE_CALL( cudaMemcpy( indicesDevice, indicesHost, indicesNewSize, cudaMemcpyHostToDevice) );

    aArray.bindIndicesTexture(aArray.indicesBufferDevicePtr, aArray.indicesBufferSize);
}

HOST void ObjUploader::uploadObjFrameNormalIndexData(
    const WFObject& aKeyFrame1,
    const WFObject& aKeyFrame2,
    PrimitiveAttributeArray<Primitive<3>, float3 >& aArray)
{
    aArray.numPrimitives = aKeyFrame1.getNumFaces();
    const size_t numIndices1 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices2 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices  = numIndices2;
    const size_t indicesNewSize = numIndices * sizeof(uint);

    ////////////////////////////////////////////////////////////
    //cleanup
    ////////////////////////////////////////////////////////////
    MemoryManager::allocateHostDeviceArrayPair(
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        indicesNewSize,
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        aArray.indicesBufferSize);

    uint* normalIndicesHost = aArray.indicesBufferHostPtr;
    uint* normalIndicesDevice = aArray.indicesBufferDevicePtr;

    
    //////////////////////////////////////////////////////////////////////////
    //Copy and transfer indices
    //////////////////////////////////////////////////////////////////////////
    size_t it = 0;
    for (; it < cudastd::min(numIndices1, numIndices2); ++it)
    {
        normalIndicesHost[it] = aKeyFrame1.getNormalIndex(it);
    }
    for (; it < numIndices2 ; ++it)
    {
        normalIndicesHost[it] = aKeyFrame2.getNormalIndex(it);
    }

    MY_CUDA_SAFE_CALL( cudaMemcpy( normalIndicesDevice, normalIndicesHost, indicesNewSize, cudaMemcpyHostToDevice) );
}

};//class ObjUploader


#endif // PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0
