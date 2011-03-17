#include "CUDAStdAfx.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/MemoryManager.h"

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
    MemoryManager::allocateMappedDeviceArray(
        (void**)&aArray.vertexBufferDevicePtr,
        (void**)&aArray.vertexBufferHostPtr,
        verticesNewSize,
        (void**)&aArray.vertexBufferDevicePtr,
        (void**)&aArray.vertexBufferHostPtr,
        aArray.vertexBufferSize);

    float4* verticesHost = aArray.vertexBufferHostPtr;
    
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

    aArray.bindVerticesTexture(aArray.vertexBufferDevicePtr, aArray.vertexBufferSize);
}

HOST void ObjUploader::uploadObjFrameNormalData(
    const WFObject& aKeyFrame1,
    const WFObject& aKeyFrame2,
    const float     aCoeff,
    PrimitiveArray<Primitive<3> >& aArray)
{
    const size_t numNormals1 = aKeyFrame1.getNumNormals();
    const size_t numNormals2 = aKeyFrame2.getNumNormals();
    const size_t numNormals  = numNormals2;
    const size_t normalsNewSize = numNormals * sizeof(float3);

    ////////////////////////////////////////////////////////////
    //cleanup
    ////////////////////////////////////////////////////////////
    MemoryManager::allocateMappedDeviceArray(
        (void**)&aArray.normalBufferDevicePtr,
        (void**)&aArray.normalBufferHostPtr,
        normalsNewSize,
        (void**)&aArray.normalBufferDevicePtr,
        (void**)&aArray.normalBufferHostPtr,
        aArray.normalBufferSize);

    float3* normalsHost = aArray.normalBufferHostPtr;
    
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
}

HOST void ObjUploader::uploadObjFrameIndexData(
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
    MemoryManager::allocateMappedDeviceArray(
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        indicesNewSize,
        (void**)&aArray.indicesBufferDevicePtr,
        (void**)&aArray.indicesBufferHostPtr,
        aArray.indicesBufferSize);

    MemoryManager::allocateMappedDeviceArray(
        (void**)&aArray.normalIndicesBufferDevicePtr,
        (void**)&aArray.normalIndicesBufferHostPtr,
        indicesNewSize,
        (void**)&aArray.normalIndicesBufferDevicePtr,
        (void**)&aArray.normalIndicesBufferHostPtr,
        aArray.normalIndicesBufferSize);


    uint* indicesHost = aArray.indicesBufferHostPtr;
    uint* normalIndicesHost = aArray.normalIndicesBufferHostPtr;

    
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

    aArray.bindIndicesTexture(aArray.indicesBufferDevicePtr, aArray.indicesBufferSize);

    for (; it < cudastd::min(numIndices1, numIndices2); ++it)
    {
        normalIndicesHost[it] = aKeyFrame1.getNormalIndex(it);
    }
    for (; it < numIndices2 ; ++it)
    {
        normalIndicesHost[it] = aKeyFrame2.getNormalIndex(it);
    }

}

