#ifdef _MSC_VER
#pragma once
#endif

#ifndef TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683
#define TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"
#include "RT/Primitive/Primitive.hpp"
#include "RT/Primitive/BBox.hpp"
#include "RT/Structure/UniformGrid.h"

#define COMPACT_INSTANCES
#ifdef COMPACT_INSTANCES
#include "DeviceConstants.h"
class  UnboundedGeometryInstanceQuaternion
{
    //28 byte instance class
    static const uint FLAG_SHIFT = 0u;
    static const uint FLAG_MASK = 0x1u;
    static const uint FLAG_MASKNEG = 0xFFFFFFFEu;

    //Transformation    
    uint indexAndSign;//last bit -> reflection flag, 0:31 bit -> index

    quaternion3f irotation;
    float3 itranslation;

public:

    DEVICE HOST bool isReflection() const
    {
        return indexAndSign & FLAG_MASK;
    }

    DEVICE HOST void setNoReflection()
    {
        indexAndSign = indexAndSign & FLAG_MASKNEG;
    }

    DEVICE HOST void setReflection()
    {
        indexAndSign = indexAndSign | FLAG_MASK;
    }

    DEVICE HOST uint getIndex() const
    {
        return indexAndSign >> 1;
    }

    DEVICE HOST void setIndex(uint aIndex)
    {
        indexAndSign = (indexAndSign & FLAG_MASK) | (aIndex << 1);
    }

    DEVICE HOST void setIdentityTransormation()
    {
        setNoReflection();
        irotation = make_quaternion4f(0.f, 0.f, 0.f, 1.f);
        itranslation = make_float3(0.f, 0.f, 0.f);
    }

    DEVICE HOST bool isIdentityTransformation() const
    {
        return fabsf(itranslation.x) + fabsf(itranslation.y) + fabsf(itranslation.z) < EPS
            && isIdentity(irotation, EPS)
            && !isReflection();
    }


    DEVICE HOST void setTransformation(
        float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        )
    {
        itranslation.x = m30;
        itranslation.y = m31;
        itranslation.z = m32;

        const float det = m00 * m11 * m22 + m10 * m21 * m02 + m20 * m01 * m12 -
            m20 * m11 * m02 - m10 * m01 * m22 - m00 * m21 * m12;
        if (det > 0.f)
        {
            setNoReflection();
            irotation = quaternion3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
        }
        else
        {
            setReflection();
            irotation = quaternion3f(
                -m00, m10, m20,
                -m01, m11, m21,
                -m02, m12, m22);
        }
    }

    DEVICE HOST void getTransformation(
        float& m00, float& m10, float& m20, float& m30,
        float& m01, float& m11, float& m21, float& m31,
        float& m02, float& m12, float& m22, float& m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        ) const
    {
        if (isReflection())
        {
            irotation.toMatrix3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
            m00 = -m00;
            m01 = -m01;
            m02 = -m02;
        }
        else
        {
            irotation.toMatrix3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
        }

        m30 = itranslation.x;
        m31 = itranslation.y;
        m32 = itranslation.z;
    }

    DEVICE HOST float3 transformRay(const float3 aRayOrg, float3& oRayDirRCP) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            irotation.toMatrix3f(
                col0.x, col1.x, col2.x,
                col0.y, col1.y, col2.y,
                col0.z, col1.z, col2.z
                );
            col0 = -col0;

            float3 rayDirT = col0 / oRayDirRCP.x + col1 / oRayDirRCP.y +
                col2 / oRayDirRCP.z;

            oRayDirRCP.x = 1.f / rayDirT.x;
            oRayDirRCP.y = 1.f / rayDirT.y;
            oRayDirRCP.z = 1.f / rayDirT.z;

            float3 rayOrgT = col0 * aRayOrg.x + col1 * aRayOrg.y + col2 * aRayOrg.z + itranslation;

            return rayOrgT;

        }
        else
        {
            oRayDirRCP = transformVecRCP(irotation, oRayDirRCP);

            oRayDirRCP.x = 1.f / oRayDirRCP.x;
            oRayDirRCP.y = 1.f / oRayDirRCP.y;
            oRayDirRCP.z = 1.f / oRayDirRCP.z;

            float3 rayOrgT = transformVec(irotation, aRayOrg) + itranslation;

            return rayOrgT;
        }
    }

    DEVICE HOST float3 transformPointToLocal(const float3 aPoint) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            irotation.toMatrix3f(
                col0.x, col1.x, col2.x,
                col0.y, col1.y, col2.y,
                col0.z, col1.z, col2.z
                );
            col0 = -col0;

            float3 rayOrgT = col0 * aPoint.x + col1 * aPoint.y + col2 * aPoint.z + itranslation;

            return rayOrgT;
        }
        else
        {
            float3 rayOrgT = transformVec(irotation, aPoint) + itranslation;

            return rayOrgT;
        }
    }

    DEVICE HOST float3 transformNormalToGlobal(const float3 aNormal) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            //Assume M inverse same as M transpose
            irotation.toMatrix3f(
                col0.x, col0.y, col0.z,
                col1.x, col1.y, col1.z,
                col2.x, col2.y, col2.z
                );
            ////Apply reflection
            //col0.x = -col0.x;
            //col1.x = -col1.x;
            //col2.x = -col2.x;

            float3 normalT = col0 * aNormal.x + col1 * aNormal.y + col2 * aNormal.z;

            return normalT;
        }
        else
        {
            quaternion3f rotation = irotation.conjugate();
            float3 normalT = transformVec(rotation, aNormal);
            return normalT;
        }
    }

    DEVICE HOST BBox transformedBBox(const BBox& aBounds) const
    {
        BBox result;
        if (isIdentityTransformation())
        {
            result.vtx[0] = aBounds.vtx[0];
            result.vtx[1] = aBounds.vtx[1];

        }
        else if (isReflection())
        {
            float3 col0, col1, col2;
            //Assume M inverse same as M transpose
            irotation.toMatrix3f(
                col0.x, col0.y, col0.z,
                col1.x, col1.y, col1.z,
                col2.x, col2.y, col2.z
                );
            //Apply reflection
            col0.x = -col0.x;
            col1.x = -col1.x;
            col2.x = -col2.x;

            float3 minBound = aBounds.vtx[0];
            minBound = minBound - itranslation;
            minBound = col0 * minBound.x + col1 * minBound.y + col2 * minBound.z;

            float3 maxBound = aBounds.vtx[1];
            maxBound = maxBound - itranslation;
            maxBound = col0 * maxBound.x + col1 * maxBound.y + col2 * maxBound.z;

            result.vtx[0] = min(minBound, maxBound);
            result.vtx[1] = max(minBound, maxBound);

            float3 pt1 = make_float3(aBounds.vtx[0].x, aBounds.vtx[0].y, aBounds.vtx[1].z);
            pt1 = pt1 -itranslation;
            pt1 = col0 * pt1.x + col1 * pt1.y + col2 * pt1.z;
            result.extend(pt1);

            float3 pt2 = make_float3(aBounds.vtx[0].x, aBounds.vtx[1].y, aBounds.vtx[0].z);
            pt2 = pt2 -itranslation;
            pt2 = col0 * pt2.x + col1 * pt2.y + col2 * pt2.z;
            result.extend(pt2);

            float3 pt3 = make_float3(aBounds.vtx[0].x, aBounds.vtx[1].y, aBounds.vtx[1].z);
            pt3 = pt3 -itranslation;
            pt3 = col0 * pt3.x + col1 * pt3.y + col2 * pt3.z;
            result.extend(pt3);

            float3 pt4 = make_float3(aBounds.vtx[1].x, aBounds.vtx[0].y, aBounds.vtx[0].z);
            pt4 = pt4 -itranslation;
            pt4 = col0 * pt4.x + col1 * pt4.y + col2 * pt4.z;
            result.extend(pt4);

            float3 pt5 = make_float3(aBounds.vtx[1].x, aBounds.vtx[0].y, aBounds.vtx[1].z);
            pt5 = pt5 -itranslation;
            pt5 = col0 * pt5.x + col1 * pt5.y + col2 * pt5.z;
            result.extend(pt5);

            float3 pt6 = make_float3(aBounds.vtx[1].x, aBounds.vtx[1].y, aBounds.vtx[0].z);
            pt6 = pt6 -itranslation;
            pt6 = col0 * pt6.x + col1 * pt6.y + col2 * pt6.z;
            result.extend(pt6);
        }
        else
        {
            quaternion3f rotation = irotation.conjugate();
            float3 minBound = transformVec(rotation, aBounds.vtx[0] - itranslation);
            float3 maxBound = transformVec(rotation, aBounds.vtx[1] - itranslation);

            result.vtx[0] = min(minBound, maxBound);
            result.vtx[1] = max(minBound, maxBound);

            float3 pt1 = make_float3(aBounds.vtx[0].x, aBounds.vtx[0].y, aBounds.vtx[1].z);
            result.extend( transformVec(rotation, pt1 - itranslation) );

            float3 pt2 = make_float3(aBounds.vtx[0].x, aBounds.vtx[1].y, aBounds.vtx[0].z);
            result.extend( transformVec(rotation, pt2 - itranslation) );

            float3 pt3 = make_float3(aBounds.vtx[0].x, aBounds.vtx[1].y, aBounds.vtx[1].z);
            result.extend( transformVec(rotation, pt3 - itranslation) );

            float3 pt4 = make_float3(aBounds.vtx[1].x, aBounds.vtx[0].y, aBounds.vtx[0].z);
            result.extend( transformVec(rotation, pt4 - itranslation) );

            float3 pt5 = make_float3(aBounds.vtx[1].x, aBounds.vtx[0].y, aBounds.vtx[1].z);
            result.extend( transformVec(rotation, pt5 - itranslation) );

            float3 pt6 = make_float3(aBounds.vtx[1].x, aBounds.vtx[1].y, aBounds.vtx[0].z);
            result.extend( transformVec(rotation, pt6 - itranslation) );

        }

        return result;
    }
};
#endif

class  GeometryInstanceQuaternion : public Primitive<2>
{
    //52 byte instance class
    static const uint FLAG_SHIFT = 0u;
    static const uint FLAG_MASK = 0x1u;
    static const uint FLAG_MASKNEG = 0xFFFFFFFEu;

    //float3 vtx[2]; //inherited -> bounding box

    //Transformation    
    uint indexAndSign;//last bit -> reflection flag, 0:31 bit -> index

    quaternion3f irotation;
    float3 itranslation;

public:


    DEVICE HOST bool isReflection() const
    {
        return indexAndSign & FLAG_MASK;
    }

    DEVICE HOST void setNoReflection()
    {
        indexAndSign = indexAndSign & FLAG_MASKNEG;
    }

    DEVICE HOST void setReflection()
    {
        indexAndSign = indexAndSign | FLAG_MASK;
    }

    DEVICE HOST uint getIndex() const
    {
        return indexAndSign >> 1;
    }

    DEVICE HOST void setIndex(uint aIndex)
    {
        indexAndSign = (indexAndSign & FLAG_MASK) | (aIndex << 1);
    }

    DEVICE HOST void setIdentityTransormation()
    {
        setNoReflection();
        irotation = make_quaternion4f(0.f, 0.f, 0.f, 1.f);
        itranslation = make_float3(0.f, 0.f, 0.f);
    }

    DEVICE HOST bool isIdentityTransformation() const
    {
        return fabsf(itranslation.x) + fabsf(itranslation.y) + fabsf(itranslation.z) < EPS
            && isIdentity(irotation, EPS)
            && !isReflection();
    }


    DEVICE HOST void setTransformation(
        float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        )
    {
        itranslation.x = m30;
        itranslation.y = m31;
        itranslation.z = m32;

        const float det = m00 * m11 * m22 + m10 * m21 * m02 + m20 * m01 * m12 -
            m20 * m11 * m02 - m10 * m01 * m22 - m00 * m21 * m12;
        if (det > 0.f)
        {
            setNoReflection();
            irotation = quaternion3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
        }
        else
        {
            setReflection();
            irotation = quaternion3f(
                -m00, m10, m20,
                -m01, m11, m21,
                -m02, m12, m22);
        }
    }

    DEVICE HOST void getTransformation(
        float& m00, float& m10, float& m20, float& m30,
        float& m01, float& m11, float& m21, float& m31,
        float& m02, float& m12, float& m22, float& m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        ) const
    {
        if (isReflection())
        {
            irotation.toMatrix3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
            m00 = -m00;
            m01 = -m01;
            m02 = -m02;
        }
        else
        {
            irotation.toMatrix3f(
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22);
        }

        m30 = itranslation.x;
        m31 = itranslation.y;
        m32 = itranslation.z;
    }

    DEVICE HOST float3 transformRay(const float3 aRayOrg, float3& oRayDirRCP) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            irotation.toMatrix3f(
                col0.x, col1.x, col2.x,
                col0.y, col1.y, col2.y,
                col0.z, col1.z, col2.z
                );
            col0 = -col0;

            float3 rayDirT = col0 / oRayDirRCP.x + col1 / oRayDirRCP.y +
                col2 / oRayDirRCP.z;

            oRayDirRCP.x = 1.f / rayDirT.x;
            oRayDirRCP.y = 1.f / rayDirT.y;
            oRayDirRCP.z = 1.f / rayDirT.z;

            float3 rayOrgT = col0 * aRayOrg.x + col1 * aRayOrg.y + col2 * aRayOrg.z + itranslation;

            return rayOrgT;

        }
        else
        {
            oRayDirRCP = transformVecRCP(irotation, oRayDirRCP);

            oRayDirRCP.x = 1.f / oRayDirRCP.x;
            oRayDirRCP.y = 1.f / oRayDirRCP.y;
            oRayDirRCP.z = 1.f / oRayDirRCP.z;

            float3 rayOrgT = transformVec(irotation, aRayOrg) + itranslation;

            return rayOrgT;
        }
    }

    DEVICE HOST float3 transformPointToLocal(const float3 aPoint) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            irotation.toMatrix3f(
                col0.x, col1.x, col2.x,
                col0.y, col1.y, col2.y,
                col0.z, col1.z, col2.z
                );
            col0 = -col0;

            float3 rayOrgT = col0 * aPoint.x + col1 * aPoint.y + col2 * aPoint.z + itranslation;

            return rayOrgT;
        }
        else
        {
            float3 rayOrgT = transformVec(irotation, aPoint) + itranslation;

            return rayOrgT;
        }
    }

    DEVICE HOST float3 transformNormalToGlobal(const float3& aNormal) const
    {

        if (isReflection())
        {
            float3 col0, col1, col2;
            //Assume M inverse same as M transpose
            irotation.toMatrix3f(
                col0.x, col0.y, col0.z,
                col1.x, col1.y, col1.z,
                col2.x, col2.y, col2.z
                );
            //Apply reflection
            col0.x = -col0.x;
            col1.x = -col1.x;
            col2.x = -col2.x;

            float3 normalT = col0 * aNormal.x + col1 * aNormal.y + col2 * aNormal.z;

            return normalT;
        }
        else
        {
            quaternion3f rotation = irotation.conjugate();
            float3 normalT = transformVec(rotation, aNormal);
            return normalT;
        }
    }
};

class  GeometryInstanceMatrix : public Primitive<2>
{
    //76 byte instance class
public:
    //float3 vtx[2]; //inherited -> bounding box
    uint index;
    //Transformation
    //float3 rotation0, rotation1, rotation2, translation;
    float3 irotation0, irotation1, irotation2;    
    float3 itranslation;

    DEVICE HOST uint getIndex() const
    {
        return index;
    }

    DEVICE HOST void setIndex(uint aIndex)
    {
        index = aIndex;
    }


    DEVICE HOST void setIdentityTransormation()
    {

        irotation0 = make_float3(1.f, 0.f, 0.f);
        irotation1 = make_float3(0.f, 1.f, 0.f);
        irotation2 = make_float3(0.f, 0.f, 1.f);
        itranslation = make_float3(0.f, 0.f, 0.f);
    }

    DEVICE HOST bool isIdentityTransformation()
    {

        float3 vec0 = irotation0 - make_float3(1.f, 0.f, 0.f);
        float3 vec1 = irotation1 - make_float3(0.f, 1.f, 0.f);
        float3 vec2 = irotation2 - make_float3(0.f, 0.f, 1.f);
        float3 vec3 = itranslation;
        return dot(vec0, vec0) <= EPS &&
            dot(vec1, vec1) <= EPS &&
            dot(vec2, vec2) <= EPS &&
            dot(vec3, vec3) <= EPS;

    }


    DEVICE HOST void setTransformation(
        float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        )
    {
        itranslation.x = m30;
        itranslation.y = m31;
        itranslation.z = m32;

        irotation0 = make_float3(m00, m01, m02);
        irotation1 = make_float3(m10, m11, m12);
        irotation2 = make_float3(m20, m21, m22);
    }

    DEVICE HOST void getTransformation(
        float& m00, float& m10, float& m20, float& m30,
        float& m01, float& m11, float& m21, float& m31,
        float& m02, float& m12, float& m22, float& m32
        //float m03, float m13, float m23, float m33 -> assumed last row: 0 0 0 1
        ) const
    {

        m00 = irotation0.x;
        m01 = irotation0.y;
        m02 = irotation0.z;

        m10 = irotation1.x;
        m11 = irotation1.y;
        m12 = irotation1.z;

        m20 = irotation2.x;
        m21 = irotation2.y;
        m22 = irotation2.z;

        m30 = itranslation.x;
        m31 = itranslation.y;
        m32 = itranslation.z;
    }

    //returns the new origin, overwrites the old direction
    DEVICE HOST float3 transformRay(const float3& aRayOrg, float3& oRayDirRCP) const
    {
        float3 rayOrgT = aRayOrg + itranslation;
        rayOrgT = irotation0 * aRayOrg.x + irotation1 * aRayOrg.y + irotation2 * aRayOrg.z + itranslation;

        float3 rayDirT = irotation0 / oRayDirRCP.x + irotation1 / oRayDirRCP.y +
            irotation2 / oRayDirRCP.z;

        oRayDirRCP.x = 1.f / rayDirT.x;
        oRayDirRCP.y = 1.f / rayDirT.y;
        oRayDirRCP.z = 1.f / rayDirT.z;

        return rayOrgT;
    }

    DEVICE HOST float3 transformPointToLocal(const float3& aPoint) const
    {
        return irotation0 * aPoint.x + irotation1 * aPoint.y + irotation2 * aPoint.z + itranslation;
    }

    DEVICE HOST float3 transformNormalToGlobal(const float3& aNormal) const
    {
        float m00; float m10; float m20; float m30;
        float m01; float m11; float m21; float m31;
        float m02; float m12; float m22; float m32;

        getTransformation(
            m00, m01, m02, m30,
            m10, m11, m12, m31,
            m20, m21, m22, m32);

        float3 normalT =
            make_float3(m00, m01, m02) * aNormal.x +
            make_float3(m10, m11, m12) * aNormal.y +
            make_float3(m20, m21, m22) * aNormal.z;

        return normalT;
    }

};

template<>
class BBoxExtractor< GeometryInstanceQuaternion >
{
public:
    DEVICE HOST static BBox get(const GeometryInstanceQuaternion& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

template<>
class BBoxExtractor< GeometryInstanceMatrix >
{
public:
    DEVICE HOST static BBox get(const GeometryInstanceMatrix& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

#ifdef COMPACT_INSTANCES

template<>
class BBoxExtractor< UnboundedGeometryInstanceQuaternion >
{
public:
    DEVICE HOST static BBox get(const UnboundedGeometryInstanceQuaternion& aInstance)
    {

#ifdef __CUDA_ARCH__
        UniformGrid* myPtr = dcUniformGridsBasePtr + aInstance.getIndex();
#else
        UniformGrid* myPtr = hcUniformGridsBasePtr + aInstance.getIndex();
#endif
        BBox originalBBox;
        originalBBox.vtx[0] = myPtr->vtx[0];
        originalBBox.vtx[1] = myPtr->vtx[1];
        return aInstance.transformedBBox(originalBBox);
    }
};

typedef UnboundedGeometryInstanceQuaternion GeometryInstance;
#else
typedef GeometryInstanceMatrix GeometryInstance;
#endif

class TwoLevelGridHierarchy : public UniformGrid
{
public:
    typedef uint2                      t_Cell;
    typedef uint2                      t_Leaf;

    //float3 vtx[2];            //inherited -> bounding box
    //int res[3];               //inherited -> uniform grid
    //float3 cellSize;          //inherited -> uniform grid
    //float3 cellSizeRCP;       //inherited -> uniform grid
    //cudaPitchedPtr  cells;    //inherited -> uniform grid
    //uint* primitives;         //inherited -> uniform grid
//#define  HACKED_PARAMETERS
#ifdef HACKED_PARAMETERS
    uint* mMemoryPtrHost;
    uint* mMemoryPtrDevice;

    TwoLevelGridHierarchy():mMemoryPtrHost(NULL), mMemoryPtrDevice(NULL)
    {

    }

    HOST DEVICE uint* getInstanceIndices() const 
    { 
#ifdef __CUDA_ARCH__
        return *(uint**)(mMemoryPtrDevice);
#else
        return *(uint**)(mMemoryPtrHost);
#endif
    }

    HOST void setInstanceIndices(uint* val)
    {
        memcpy(mMemoryPtrHost, (void*)&val, sizeof(uint*));
        MY_CUDA_SAFE_CALL(cudaMemcpy(mMemoryPtrDevice, mMemoryPtrHost, getParametersSize(), cudaMemcpyHostToDevice));
    }

    HOST DEVICE GeometryInstance* getInstances() const
    {
#ifdef __CUDA_ARCH__
        return *(GeometryInstance**)(mMemoryPtrDevice + sizeof(uint*));
#else
        return *(GeometryInstance**)(mMemoryPtrHost + sizeof(uint*));
#endif
    }

    HOST void setInstances(GeometryInstance* val)
    { 
        memcpy((void*)(mMemoryPtrHost + sizeof(uint*)), (void*)&val, sizeof(GeometryInstance*));
        MY_CUDA_SAFE_CALL(cudaMemcpy(mMemoryPtrDevice, mMemoryPtrHost, getParametersSize(), cudaMemcpyHostToDevice));
    }

    HOST DEVICE UniformGrid* getGrids() const 
    { 
#ifdef __CUDA_ARCH__
        return *(UniformGrid**)(mMemoryPtrDevice + sizeof(uint*) + sizeof(GeometryInstance*));
#else
        return *(UniformGrid**)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*));
#endif
    }

    HOST void setGrids(UniformGrid* val)
    {
        memcpy((void*)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*)), (void*)&val, sizeof(UniformGrid*));
        MY_CUDA_SAFE_CALL(cudaMemcpy(mMemoryPtrDevice, mMemoryPtrHost, getParametersSize(), cudaMemcpyHostToDevice));
    }

    HOST DEVICE t_Leaf* getLeaves() const
    {
#ifdef __CUDA_ARCH__
        return *(t_Leaf**)(mMemoryPtrDevice + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*));
#else
        return *(t_Leaf**)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*));
#endif 
    }

    HOST void setLeaves(t_Leaf* val)
    {
        memcpy((void*)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*)), (void*)&val, sizeof(t_Leaf*));
        MY_CUDA_SAFE_CALL(cudaMemcpy(mMemoryPtrDevice, mMemoryPtrHost, getParametersSize(), cudaMemcpyHostToDevice));
    }

    HOST DEVICE uint getNumInstances() const
    { 
#ifdef __CUDA_ARCH__
        return *(uint*)(mMemoryPtrDevice + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*) + sizeof(t_Leaf*));
#else
        return *(uint*)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*) + sizeof(t_Leaf*));
#endif 
    }

	HOST void setNumInstances(uint val)
    {
        memcpy((void*)(mMemoryPtrHost + sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*) + sizeof(t_Leaf*)), (void*)&val, sizeof(uint));
        MY_CUDA_SAFE_CALL(cudaMemcpy(mMemoryPtrDevice, mMemoryPtrHost, getParametersSize(), cudaMemcpyHostToDevice));
    }

    static int getParametersSize()
    {
        return sizeof(uint*) + sizeof(GeometryInstance*) + sizeof(UniformGrid*) + sizeof(t_Leaf*) + sizeof(uint);
    }

    HOST void setMemoryPtr(uint* aHostPtr, uint* aDevicePtr)
    {
        mMemoryPtrHost = aHostPtr;
        mMemoryPtrDevice = aDevicePtr;
    }

    HOST void allocatePtrs()
    {
        MY_CUDA_SAFE_CALL(cudaHostAlloc((void**)&mMemoryPtrHost, TwoLevelGridHierarchy::getParametersSize(), cudaHostAllocDefault));
        MY_CUDA_SAFE_CALL(cudaMalloc((void**)&mMemoryPtrDevice, TwoLevelGridHierarchy::getParametersSize()));
    }

    HOST void freePtrs()
    {
        if (mMemoryPtrDevice != NULL || mMemoryPtrHost != NULL)
        {
            MY_CUDA_SAFE_CALL(cudaFreeHost(mMemoryPtrHost));
            mMemoryPtrHost = NULL;
            MY_CUDA_SAFE_CALL(cudaFree(mMemoryPtrDevice));
            mMemoryPtrDevice = NULL;
        }
    }
#else    
    uint*           instanceIndices;
    GeometryInstance* instances;
    UniformGrid*    grids;
    t_Leaf*         leaves;
    uint            numInstances;
    uint  numPrimitiveReferences;

    HOST DEVICE uint* getInstanceIndices() const 
    { 

        return instanceIndices;
    }

    HOST void setInstanceIndices(uint* val)
    {
        instanceIndices = val;
    }

    HOST DEVICE GeometryInstance* getInstances() const
    {
        return instances;
    }

    HOST void setInstances(GeometryInstance* val)
    { 
        instances = val;
    }

    HOST DEVICE UniformGrid* getGrids() const 
    { 
        return grids;
    }

    HOST void setGrids(UniformGrid* val)
    {
        grids = val;
    }

    HOST DEVICE t_Leaf* getLeaves() const
    {
        return leaves; 
    }

    HOST void setLeaves(t_Leaf* val)
    {
        leaves = val;
    }

    HOST DEVICE uint getNumInstances() const
    { 
        return numInstances;
    }

    HOST void setNumInstances(uint val)
    {
        numInstances = val;
    }

    static int getParametersSize()
    {
        return sizeof(uint);
    }

    HOST void setMemoryPtr(uint* aHostPtr, uint* aDevicePtr)
    {
        //dummy
    }

    HOST void allocatePtrs()
    {
        //dummy
    }

    HOST void freePtrs()
    {
        //dummy
    }
#endif
    //////////////////////////////////////////////////////////////////////////
    //inherited -> uniform grid

    //HOST DEVICE const float3 getResolution() const; 

    //HOST DEVICE float3 getCellSize() const;

    //HOST DEVICE float3 getCellSizeRCP() const;

    //HOST DEVICE int getCellIdLinear(int aIdX, int aIdY, int aIdZ);

    //HOST DEVICE int3 getCellId3D(int aLinearId);

    //HOST DEVICE t_Cell getCell(int aIdX, int aIdY, int aIdZ);

    //HOST DEVICE void setCell(int aIdX, int aIdY, int aIdZ, uint2 aVal);

    //HOST DEVICE int3 getCellIdAt(float3 aPosition);

    //HOST DEVICE uint2 getCellAt(float3 aPosition);

    //HOST DEVICE float3 getCellCenter (int aIdX, int aIdY, int aIdZ) const;

    //HOST DEVICE uint getPrimitiveId(uint aId);

    //////////////////////////////////////////////////////////////////////////
};

template<>
class BBoxExtractor< TwoLevelGridHierarchy >
{
public:
    DEVICE HOST static BBox get(const TwoLevelGridHierarchy& aUGrid)
    {
        BBox result;
        result.vtx[0] = aUGrid.vtx[0];
        result.vtx[1] = aUGrid.vtx[1];
        return result;
    }
};

#endif // TWOLEVELGRIDHIERARCHY_H_INCLUDED_AA48BA45_DEA8_4023_B4C9_D8D125325683
