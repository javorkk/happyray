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

template<class tPrimitive>
class PrimitiveArray
{

public:
    /////////////////////////////////////////////////////////////
    //Memory Management
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
        if(vertexBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(vertexBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );

        if(vertexBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(vertexBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );

        vertexBufferDevicePtr = NULL;
        vertexBufferHostPtr = NULL;
        vertexBufferSize = 0u;
        indicesBufferDevicePtr= NULL;
        indicesBufferHostPtr= NULL;
        indicesBufferSize = 0u;
        numPrimitives = 0u;

    }

    HOST DEVICE size_t getMemorySize()
    {
        return vertexBufferSize + indicesBufferSize;
    }

    ////////////////////////////////////////////////////////////
    //Accessors Functions
    ////////////////////////////////////////////////////////////

    HOST DEVICE tPrimitive operator[](uint aIndex) const
    {

        uint indices[tPrimitive::NUM_VERTICES];


#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
#ifdef __CUDA_ARCH__
            indices[i] = indicesBufferDevicePtr[aIndex * tPrimitive::NUM_VERTICES + i];
#else
            indices[i] = indicesBufferHostPtr[aIndex * tPrimitive::NUM_VERTICES + i];
#endif // __CUDA_ARCH__
        }

        tPrimitive result;

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
#ifdef __CUDA_ARCH__
            float4 tmp = vertexBufferDevicePtr[indices[i]];
#else
            float4 tmp = vertexBufferHostPtr[indices[i]];
#endif // __CUDA_ARCH__
            result.vtx[i].x = tmp.x;
            result.vtx[i].y = tmp.y;
            result.vtx[i].z = tmp.z;
        }

        return result;
    }

};//class Primitive Array



template<class tPrimitive, class tType>
class VtxAttribStruct
{
public:
    tType data[tPrimitive::NUM_VERTICES];
};

template<class tPrimitive, class tType>
class VtxAttributeArray
{
public:
    typedef VtxAttribStruct<tPrimitive, tType> t_AttribStruct;

    /////////////////////////////////////////////////////////////
    //Memory Management
    /////////////////////////////////////////////////////////////

    tType* dataBufferDevicePtr;
    tType* dataBufferHostPtr;
    size_t  dataBufferSize;

    uint*  indicesBufferDevicePtr;
    uint*  indicesBufferHostPtr;
    size_t indicesBufferSize;

    HOST VtxAttributeArray():
        dataBufferDevicePtr(NULL),
        dataBufferHostPtr  (NULL),
        dataBufferSize(0u),
        indicesBufferDevicePtr(NULL),
        indicesBufferHostPtr  (NULL),
        indicesBufferSize(0u)
    {}

    HOST void cleanup()
    {

        if(dataBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(dataBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );

        if(dataBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(dataBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );

        dataBufferDevicePtr = NULL;
        dataBufferHostPtr   = NULL;
        dataBufferSize      = 0u;
        indicesBufferDevicePtr = NULL;
        indicesBufferHostPtr   = NULL;
        indicesBufferSize   = 0u;
    }

    HOST DEVICE size_t getMemorySize()
    {
        return dataBufferSize + indicesBufferSize;
    }


    ////////////////////////////////////////////////////////////
    //Accessors Functions
    ////////////////////////////////////////////////////////////
 
    HOST DEVICE t_AttribStruct  operator[](uint aIndex) const
    {
        uint indices[tPrimitive::NUM_VERTICES];

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
#ifdef __CUDA_ARCH__
            indices[i] = indicesBufferDevicePtr[aIndex * tPrimitive::NUM_VERTICES + i]; 
#else
            indices[i] = indicesBufferHostPtr[aIndex * tPrimitive::NUM_VERTICES + i];
#endif // __CUDA_ARCH__
        }

        t_AttribStruct result;

#pragma unroll 3
        for(uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
        {
#ifdef __CUDA_ARCH__
            result.data[i] = dataBufferDevicePtr[indices[i]];
#else
            result.data[i] = dataBufferHostPtr[indices[i]];
#endif // __CUDA_ARCH__
        }

        return result;
    }

};//class Vertex Attribute Array

template<class tType>
class PrimitiveAttributeArray
{
public:
    /////////////////////////////////////////////////////////////
    //Memory Management
    /////////////////////////////////////////////////////////////

    tType* dataBufferDevicePtr;
    tType* dataBufferHostPtr;
    size_t  dataBufferSize;

    uint*  indicesBufferDevicePtr;
    uint*  indicesBufferHostPtr;
    size_t indicesBufferSize;

#ifndef __CUDA_ARCH__
    HOST PrimitiveAttributeArray():
        dataBufferDevicePtr(NULL),
        dataBufferHostPtr  (NULL),
        dataBufferSize(0u),
        indicesBufferDevicePtr(NULL),
        indicesBufferHostPtr  (NULL),
        indicesBufferSize(0u)
    {}
#endif

    HOST void cleanup()
    {

        if(dataBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(dataBufferHostPtr) );
        if(indicesBufferHostPtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFreeHost(indicesBufferHostPtr) );

        if(dataBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(dataBufferDevicePtr) );
        if(indicesBufferDevicePtr != NULL)
            MY_CUDA_SAFE_CALL( cudaFree(indicesBufferDevicePtr) );

        dataBufferDevicePtr = NULL;
        dataBufferHostPtr   = NULL;
        dataBufferSize      = 0u;
        indicesBufferDevicePtr = NULL;
        indicesBufferHostPtr   = NULL;
        indicesBufferSize   = 0u;
    }

    HOST DEVICE size_t getMemorySize()
    {
        return dataBufferSize + indicesBufferSize;
    }


    ////////////////////////////////////////////////////////////
    //Accessors Functions
    ////////////////////////////////////////////////////////////
 
    DEVICE tType  operator[](uint aIndex) const
    {
#ifdef __CUDA_ARCH__
        return dataBufferDevicePtr[indicesBufferDevicePtr[aIndex]];
#else
        return dataBufferHostPtr[indicesBufferHostPtr[aIndex]];
#endif // __CUDA_ARCH__
        
    }

};//class Primitive Attribute Array


#endif // PRIMITIVEARRAY_H_INCLUDED_4D30F730_121F_4E9F_915B_1814410DAEB0
