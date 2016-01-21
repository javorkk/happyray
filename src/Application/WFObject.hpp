/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
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

#ifndef FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
#define FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223

#include "Core/Algebra.hpp"

//A Wavefront3D object. It provides the functionality of 
//	an indexed triangle mesh. The primitive consists
//	of 5 buffers holding vertex positions, vertex normals,
//	vertex texture coordinates, face materials, and the faces
//	them selfs. Each face is a triangle and only contains indices
//	into the position, normal, texture coordinate, and material
//	buffers.

class WFObject
{
public:
    class Face;
    static const float3 C0;
    static const float3 C1;
    static const float3 C099;

public:
    WFObject():
        mNumVertices(0u), mNumNormals(0u), mNumFaces(0u), mNumMaterials(0u),
            mNumTexCoords(0u),
        mVerticesBufferSize(0u), mNormalsBufferSize(0u), mFacesBufferSize(0u),
            mMaterialsBufferSize(0u), mTexCoordsBufferSize(0u),
        mVertices(NULL), mNormals(NULL), mVertexIndices(NULL),
        mNormalIndices(NULL),mFaces(NULL), mMaterials(NULL),
            mTexCoords(NULL)
    {}

    ~WFObject()
    {
        //TODO : fix destructor
        if (mVertices != NULL)
        {
            delete[] mVertices;
        }
        if (mNormals != NULL)
        {
            delete[] mNormals;
        }
        if(mVertexIndices != NULL)
        {
            delete mVertexIndices;
        }
        if(mNormalIndices != NULL)
        {
            delete mNormalIndices;
        }

        if (mFaces != NULL)
        {
            delete[] mFaces;
        }
        if (mMaterials != NULL)
        {
            delete[] mMaterials;
        }
        if (mTexCoords != NULL)
        {
            delete[] mTexCoords;
        }
    }

    //Represents a material
    struct Material
    {
        //The unique name of the material
        const char* name;
        //diffuse reflectance
        float3 diffuseCoeff;
        //specular reflectance
        float3 specularCoeff;
        //ambient color
        float3 ambientCoeff;
        //emission (light source only)
        float3 emission;

        float specularExp;
        float indexOfRefraction;
        bool isRefractive;


        Material():
            diffuseCoeff(C0),
            specularCoeff(C0),
            ambientCoeff(C0),
            emission(C0),
            specularExp(1.f),
            indexOfRefraction(1.f),
            isRefractive(false) //non-transparent
        {}
        Material(const char* _name) : name(_name),
            diffuseCoeff(C0),
            specularCoeff(C0),
            ambientCoeff(C0),
            specularExp(1.f),
            indexOfRefraction(1.f),
            isRefractive(false)
        {}

        void setupPhongCoefficients();

    };


    //A face definition (triangle)
    class Face
    {
        WFObject *m_lwObject;
    public:
        size_t material;
        size_t vert1, tex1, norm1;
        size_t vert2, tex2, norm2;
        size_t vert3, tex3, norm3;

        Face() : m_lwObject(NULL)
        {}

        Face(WFObject * _obj) : m_lwObject(_obj) {}
    };


    size_t getNumVertices() const
    {
        return mNumVertices;
    }
    size_t getNumNormals() const
    {
        return mNumNormals;
    }
    size_t getNumFaces() const
    {
        return mNumFaces;
    }
    size_t getNumMaterials() const
    {
        return mNumMaterials;
    }
    size_t getNumTexCoords() const
    {
        return mNumTexCoords;
    }
    
    float3 getVertex(size_t aVtxId) const
    {
        return *reinterpret_cast<float3*>(mVertices + aVtxId);
    }

    float3 getNormal(size_t aNormalId) const
    {
        return *reinterpret_cast<float3*>(mNormals + aNormalId);
    }

    Face getFace(size_t aFaceId) const
    {
        return mFaces[aFaceId];
    }

    uint getVertexIndex(size_t aId) const
    {
        return mVertexIndices[aId];
    }

    uint getNormalIndex(size_t aId) const
    {
        return mNormalIndices[aId];
    }
    Material getMaterial(size_t aMatId) const
    {
        return mMaterials[aMatId];
    }

    float2 getTexCoords(size_t aCoordId) const
    {
        return mTexCoords[aCoordId];
    }

    float3& getVertex(size_t aVtxId)
    {
        return *reinterpret_cast<float3*>(mVertices + aVtxId);
    }

    float3& getNormal(size_t aNormalId)
    {
        return *reinterpret_cast<float3*>(mNormals + aNormalId);
    }

    Face& getFace(size_t aFaceId)
    {
        return mFaces[aFaceId];
    }

    Material& getMaterial(size_t aMatId)
    {
        return mMaterials[aMatId];
    }

    float2& getTexCoords(size_t aCoordId)
    {
        return mTexCoords[aCoordId];
    }

    size_t insertVertex(const float3& aVertex);

    size_t insertNormal(const float3& aNormal);

    size_t insertFace(const Face& aFace);

    size_t insertMaterial(const Material& aMaterial);


    void allocateVertices(const size_t aSize)
    {
        if (mVertices != NULL)
        {
            delete[] mVertices;
        }

        mVerticesBufferSize = aSize + cudastd::max((size_t)1,aSize / 4u);
        mVertices = new float3[mVerticesBufferSize];
        mNumVertices = aSize;
    }


    void allocateNormals(const size_t aSize)
    {
        if (mNormals != NULL)
        {
            delete[] mNormals;
        }

        mNormalsBufferSize = aSize + cudastd::max((size_t)1,aSize / 4u);
        mNormals = new float3[mNormalsBufferSize];
        mNumNormals = aSize;
    }

    void allocateFaces(const size_t aSize)
    {
        if (mFaces != NULL)
        {
            delete[] mFaces;
        }

        if (mVertexIndices != NULL)
        {
            delete mVertexIndices;
        }

        if (mNormalIndices != NULL)
        {
            delete mNormalIndices;
        }

        mFacesBufferSize = aSize + cudastd::max((size_t)1, aSize / 4u);
        mFaces = new Face[mFacesBufferSize];
        mVertexIndices = new uint[mFacesBufferSize * 3];
        mNormalIndices = new uint[mFacesBufferSize * 3];
        mNumFaces = aSize;
    }

    void allocateMaterials(const size_t aSize)
    {
        if (mMaterials != NULL)
        {
            delete[] mMaterials;
        }

        mMaterialsBufferSize = aSize + cudastd::max((size_t)1,aSize / 4u);
        mMaterials = new Material[mMaterialsBufferSize];
        mNumMaterials = aSize;

    }

    void allocateTexCoords(const size_t aSize)
    {
        if (mTexCoords != NULL)
        {
            delete[] mTexCoords;
        }

        mTexCoordsBufferSize = aSize + cudastd::max((size_t)1, aSize / 4u);
        mTexCoords = new float2[mTexCoordsBufferSize];
        mNumTexCoords = aSize;

    }

    //Reads the FrontWave3D object from a file
    void read(const char* aFileName);
    void loadWFObj(const char* aFileName);
    void copyVectorsToArrays(); //hack for using std constructs without nvcc knowing about them

    typedef const float3* t_VertexIterator;
    typedef const uint* t_IndexIterator;
    typedef const Face* t_FaceIterator;
    typedef const Material* t_MaterialIterator;
    typedef Material* t_MaterialIteratorNonConst;


    t_VertexIterator verticesBegin() const
    {
        return mVertices;
    }

    t_VertexIterator verticesEnd() const
    {
        return mVertices + mNumVertices;
    }

    t_FaceIterator facesBegin() const
    {
        return mFaces;
    }

    t_FaceIterator facesEnd() const
    {
        return mFaces + mNumFaces;
    }

    t_MaterialIterator materialsBegin() const
    {
        return mMaterials;
    }

    t_MaterialIterator materialsEnd() const
    {
        return mMaterials + mNumMaterials;
    }

    t_MaterialIteratorNonConst materialsBegin()
    {
        return mMaterials;
    }

    t_MaterialIteratorNonConst materialsEnd()
    {
        return mMaterials + mNumMaterials;
    }

    t_IndexIterator vertexIndicesBegin()
    {
        return mVertexIndices;
    }

    t_IndexIterator vertexIndicesEnd()
    {
        return mVertexIndices + mNumFaces * 3;
    }

    t_IndexIterator NormalIndicesBegin()
    {
        return mNormalIndices;
    }

    t_IndexIterator NormalIndicesEnd()
    {
        return mNormalIndices + mNumFaces * 3;
    }

    t_VertexIterator normalsBegin() const
    {
        return mNormals;
    }

    t_VertexIterator normalsEnd() const
    {
        return mNormals + mNumNormals;
    }

private:
    size_t mNumVertices, mNumNormals, mNumFaces, mNumMaterials, mNumTexCoords;
    size_t mVerticesBufferSize, mNormalsBufferSize, mFacesBufferSize, mMaterialsBufferSize, mTexCoordsBufferSize;
    float3* mVertices;
    float3* mNormals;
    uint* mVertexIndices;
    uint* mNormalIndices;
    Face*  mFaces;
    Material* mMaterials;
    float2* mTexCoords;
};

#endif // FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
