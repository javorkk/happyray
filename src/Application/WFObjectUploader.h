#ifdef _MSC_VER
#pragma once
#endif

#ifndef WFOBJECTUPLOADER_H_BF3C6B01_FFA4_461A_970C_5301D0C8ED13
#define WFOBJECTUPLOADER_H_BF3C6B01_FFA4_461A_970C_5301D0C8ED13

#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/Material.hpp"
#include "RT/Structure/3DTextureMemoryManager.h"
#include "Application/WFObject.hpp"
#include "RT/Structure/TexturedPrimitiveArray.h"
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
        TexturedPrimitiveArray<Primitive<3> >& aArray)
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
        VtxAttributeArray<Primitive<3>, float3 >& aArray)
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
        TexturedPrimitiveArray<Primitive<3> >& aArray)
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
        VtxAttributeArray<Primitive<3>, float3 >& aArray)
    {
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

    HOST void ObjUploader::uploadObjFrameMaterialData(
        const WFObject& aKeyFrame2,
        PrimitiveAttributeArray< PhongMaterial >& aArray)
    {
        const size_t numMaterials = aKeyFrame2.getNumMaterials();
        const size_t materialsNewSize = numMaterials * sizeof(PhongMaterial);

        ////////////////////////////////////////////////////////////
        //cleanup
        ////////////////////////////////////////////////////////////
        MemoryManager::allocateHostDeviceArrayPair(
            (void**)&aArray.dataBufferDevicePtr,
            (void**)&aArray.dataBufferHostPtr,
            materialsNewSize,
            (void**)&aArray.dataBufferDevicePtr,
            (void**)&aArray.dataBufferHostPtr,
            aArray.dataBufferSize);

        PhongMaterial* matHost = aArray.dataBufferHostPtr;
        PhongMaterial* matDevice = aArray.dataBufferDevicePtr;

        //////////////////////////////////////////////////////////////////////////
        //Copy and transfer data
        //////////////////////////////////////////////////////////////////////////        
        PhongMaterial current;
        for (size_t it = 0; it < numMaterials; ++it)
        {
            current.diffuseReflectance.x = aKeyFrame2.getMaterial(it).diffuseCoeff.x;
            current.diffuseReflectance.y = aKeyFrame2.getMaterial(it).diffuseCoeff.y;
            current.diffuseReflectance.z = aKeyFrame2.getMaterial(it).diffuseCoeff.z;
            current.diffuseReflectance.w = aKeyFrame2.getMaterial(it).indexOfRefraction;
            if(aKeyFrame2.getMaterial(it).isRefractive)
                current.diffuseReflectance.w += 10.f;
            current.specularReflectance.x = aKeyFrame2.getMaterial(it).specularCoeff.x;
            current.specularReflectance.y = aKeyFrame2.getMaterial(it).specularCoeff.y;
            current.specularReflectance.z = aKeyFrame2.getMaterial(it).specularCoeff.z;
            current.specularReflectance.w = aKeyFrame2.getMaterial(it).specularExp;

            //current.emission.x = aKeyFrame2.getMaterial(it).emission.x;
            //current.emission.y = aKeyFrame2.getMaterial(it).emission.y;
            //current.emission.z = aKeyFrame2.getMaterial(it).emission.z;
            //current.emission.w = 0.f;

            matHost[it] = current;
        }

        MY_CUDA_SAFE_CALL( cudaMemcpy( matDevice, matHost, materialsNewSize, cudaMemcpyHostToDevice) );

        const size_t numIndices = aKeyFrame2.getNumFaces();
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

        uint* materialIndicesHost = aArray.indicesBufferHostPtr;
        uint* materialIndicesDevice = aArray.indicesBufferDevicePtr;


        //////////////////////////////////////////////////////////////////////////
        //Copy and transfer indices
        //////////////////////////////////////////////////////////////////////////
        for (size_t it = 0; it < numIndices; ++it)
        {
            materialIndicesHost[it] = aKeyFrame2.getFace(it).material;
        }

        MY_CUDA_SAFE_CALL( cudaMemcpy( materialIndicesDevice, materialIndicesHost, indicesNewSize, cudaMemcpyHostToDevice) );

    }

    HOST void ObjUploader::uploadObjFrameTextureData(
        const WFObject& aKeyFrame2,
        PrimitiveAttributeArray< TexturedPhongMaterial >& aArray,
        TextureMemoryManager& aTexMemoryManager)
    {
        const size_t numMaterials = aKeyFrame2.getNumMaterials();
        const size_t materialsNewSize = numMaterials * sizeof(TexturedPhongMaterial);

        ////////////////////////////////////////////////////////////
        //cleanup
        ////////////////////////////////////////////////////////////
        MemoryManager::allocateHostDeviceArrayPair(
            (void**)&aArray.dataBufferDevicePtr,
            (void**)&aArray.dataBufferHostPtr,
            materialsNewSize,
            (void**)&aArray.dataBufferDevicePtr,
            (void**)&aArray.dataBufferHostPtr,
            aArray.dataBufferSize);

        TexturedPhongMaterial* matHost = aArray.dataBufferHostPtr;
        TexturedPhongMaterial* matDevice = aArray.dataBufferDevicePtr;

        //////////////////////////////////////////////////////////////////////////
        //Copy and transfer data
        //////////////////////////////////////////////////////////////////////////        
        TexturedPhongMaterial current = aTexMemoryManager.getParameters();
        for (size_t it = 0; it < numMaterials; ++it)
        {
            current.diffuseReflectance.x = aKeyFrame2.getMaterial(it).diffuseCoeff.x;
            current.diffuseReflectance.y = aKeyFrame2.getMaterial(it).diffuseCoeff.y;
            current.diffuseReflectance.z = aKeyFrame2.getMaterial(it).diffuseCoeff.z;
            current.diffuseReflectance.w = aKeyFrame2.getMaterial(it).indexOfRefraction;
            if(aKeyFrame2.getMaterial(it).isRefractive)
                current.diffuseReflectance.w += 10.f;
            current.specularReflectance.x = aKeyFrame2.getMaterial(it).specularCoeff.x;
            current.specularReflectance.y = aKeyFrame2.getMaterial(it).specularCoeff.y;
            current.specularReflectance.z = aKeyFrame2.getMaterial(it).specularCoeff.z;
            current.specularReflectance.w = aKeyFrame2.getMaterial(it).specularExp;

            //current.emission.x = aKeyFrame2.getMaterial(it).emission.x;
            //current.emission.y = aKeyFrame2.getMaterial(it).emission.y;
            //current.emission.z = aKeyFrame2.getMaterial(it).emission.z;
            //current.emission.w = 0.f;

            matHost[it] = current;
        }

        MY_CUDA_SAFE_CALL( cudaMemcpy( matDevice, matHost, materialsNewSize, cudaMemcpyHostToDevice) );

        const size_t numIndices = aKeyFrame2.getNumFaces();
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

        uint* materialIndicesHost = aArray.indicesBufferHostPtr;
        uint* materialIndicesDevice = aArray.indicesBufferDevicePtr;


        //////////////////////////////////////////////////////////////////////////
        //Copy and transfer indices
        //////////////////////////////////////////////////////////////////////////
        for (size_t it = 0; it < numIndices; ++it)
        {
            materialIndicesHost[it] = aKeyFrame2.getFace(it).material;
        }

        MY_CUDA_SAFE_CALL( cudaMemcpy( materialIndicesDevice, materialIndicesHost, indicesNewSize, cudaMemcpyHostToDevice) );

    }



};//class ObjUploader

#endif // WFOBJECTUPLOADER_H_BF3C6B01_FFA4_461A_970C_5301D0C8ED13
