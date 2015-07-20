#include "StdAfx.hpp"
#include "Test/TestAlgebra.h"

#include "Core/Algebra.hpp"
#include "RT/Structure/TwoLevelGridHierarchy.h"
#include "RT/Primitive/BBox.hpp"
#include "Utils/RandomNumberGenerators.hpp"

#include "Utils/CUDAUtil.h"

SimpleRandomNumberGenerator genRand(1231415u);

const float ANGLE_THRESHOLD = 1.f - cosf(1.396263401595464); // 1.f - cos(80 deg) = 1.f - 0.0003f

void TestQuaternions::initRay(float3& aOrg, float3& aDirRCP) const
{
    const float3 origin = make_float3(-100.f, -100.f, -100.f);
    const float3 extent = make_float3(200.f, 200.f, 200.f);

    aOrg = make_float3(genRand(), genRand(), genRand());
    aOrg = aOrg * extent - origin;

    aDirRCP = make_float3(1.f, 1.f, 1.f) / ~make_float3(genRand(), genRand(), genRand());
}

bool TestQuaternions::checkTransformations(
    float m00, float m10, float m20, float m30,
    float m01, float m11, float m21, float m31,
    float m02, float m12, float m22, float m32,

    float n00, float n10, float n20, float n30,
    float n01, float n11, float n21, float n31,
    float n02, float n12, float n22, float n32) const
{
    if (fabsf(m00 - n00) > EPS) cudastd::logger::out << "m00 and n00 differ: m = " << m00 << " n = " << n00 << "\n";
    if (fabsf(m01 - n01) > EPS) cudastd::logger::out << "m01 and n01 differ: m = " << m01 << " n = " << n01 << "\n";
    if (fabsf(m02 - n02) > EPS) cudastd::logger::out << "m02 and n02 differ: m = " << m02 << " n = " << n02 << "\n";
    if (fabsf(m10 - n10) > EPS) cudastd::logger::out << "m10 and n10 differ: m = " << m10 << " n = " << n10 << "\n";
    if (fabsf(m11 - n11) > EPS) cudastd::logger::out << "m11 and n11 differ: m = " << m11 << " n = " << n11 << "\n";
    if (fabsf(m12 - n12) > EPS) cudastd::logger::out << "m12 and n12 differ: m = " << m12 << " n = " << n12 << "\n";
    if (fabsf(m20 - n20) > EPS) cudastd::logger::out << "m20 and n20 differ: m = " << m20 << " n = " << n20 << "\n";
    if (fabsf(m21 - n21) > EPS) cudastd::logger::out << "m21 and n21 differ: m = " << m21 << " n = " << n21 << "\n";
    if (fabsf(m22 - n22) > EPS) cudastd::logger::out << "m22 and n22 differ: m = " << m22 << " n = " << n22 << "\n";
    if (fabsf(m30 - n30) > EPS) cudastd::logger::out << "m30 and n30 differ: m = " << m30 << " n = " << n30 << "\n";
    if (fabsf(m31 - n31) > EPS) cudastd::logger::out << "m31 and n31 differ: m = " << m31 << " n = " << n31 << "\n";
    if (fabsf(m32 - n32) > EPS) cudastd::logger::out << "m32 and n32 differ: m = " << m32 << " n = " << n32 << "\n";


    return
        fabsf(m00 - n00) + fabsf(m10 - n10) + fabsf(m20 - n20) + fabsf(m30 - n30) +
        fabsf(m01 - n01) + fabsf(m11 - n11) + fabsf(m21 - n21) + fabsf(m31 - n31) +
        fabsf(m02 - n02) + fabsf(m12 - n12) + fabsf(m22 - n22) + fabsf(m32 - n32) < 4.f * EPS;

}

bool TestQuaternions::checkRay(
    const float3& aRayOrg1, const float3& aRayDirRCP1,
    const float3& aRayOrg2, const float3& aRayDirRCP2) const
{
    if (fabsf(aRayOrg1.x - aRayOrg2.x) > EPS) cudastd::logger::out << "aRayOrg1 x and aRayOrg2 x differ: (" << aRayOrg1.x << " vs " << aRayOrg2.x << ")\n";
    if (fabsf(aRayOrg1.y - aRayOrg2.y) > EPS) cudastd::logger::out << "aRayOrg1 y and aRayOrg2 y differ: (" << aRayOrg1.y << " vs " << aRayOrg2.y << ")\n";
    if (fabsf(aRayOrg1.z - aRayOrg2.z) > EPS) cudastd::logger::out << "aRayOrg1 z and aRayOrg2 z differ: (" << aRayOrg1.z << " vs " << aRayOrg2.z << ")\n";
    if (fabsf(aRayDirRCP1.x - aRayDirRCP2.x) > 100.f * EPS) cudastd::logger::out << "aRayDirRCP1 x and aRayDirRCP2 x differ: (" << aRayDirRCP1.x << " vs " << aRayDirRCP2.x << ")\n";
    if (fabsf(aRayDirRCP1.y - aRayDirRCP2.y) > 100.f * EPS) cudastd::logger::out << "aRayDirRCP1 y and aRayDirRCP2 y differ: (" << aRayDirRCP1.y << " vs " << aRayDirRCP2.y << ")\n";
    if (fabsf(aRayDirRCP1.z - aRayDirRCP2.z) > 100.f * EPS) cudastd::logger::out << "aRayDirRCP1 z and aRayDirRCP2 z differ: (" << aRayDirRCP1.z << " vs " << aRayDirRCP2.z << ")\n";

    float3 aRayDir1 = make_float3(1.f, 1.f, 1.f) / aRayDirRCP1;
    float3 aRayDir2 = make_float3(1.f, 1.f, 1.f) / aRayDirRCP2;
    if (dot(aRayDir1, aRayDir2) < ANGLE_THRESHOLD)
    {
        cudastd::logger::out << "aRayDir1 x and aRayDir2 x differ: (" << aRayDir1.x << " vs " << aRayDir2.x << ")\n";
        cudastd::logger::out << "aRayDir1 y and aRayDir2 y differ: (" << aRayDir1.y << " vs " << aRayDir2.y << ")\n";
        cudastd::logger::out << "aRayDir1 z and aRayDir2 z differ: (" << aRayDir1.z << " vs " << aRayDir2.z << ")\n";
    }

    return
        fabsf(aRayOrg1.x - aRayOrg2.x) +
        fabsf(aRayOrg1.y - aRayOrg2.y) +
        fabsf(aRayOrg1.z - aRayOrg2.z) < EPS &&
        dot(aRayDir1, aRayDir2) >= ANGLE_THRESHOLD;
}

const size_t NUM_SAMPLES = 100u;

bool TestQuaternions::run() const
{

    float m00; float m10; float m20; float m30;
    float m01; float m11; float m21; float m31;
    float m02; float m12; float m22; float m32;

    float Qm00; float Qm10; float Qm20; float Qm30;
    float Qm01; float Qm11; float Qm21; float Qm31;
    float Qm02; float Qm12; float Qm22; float Qm32;

    float Mm00; float Mm10; float Mm20; float Mm30;
    float Mm01; float Mm11; float Mm21; float Mm31;
    float Mm02; float Mm12; float Mm22; float Mm32;


    GeometryInstance instanceQ;
    GeometryInstanceMatrix instanceM;

    instanceQ.vtx[0] = make_float3(-10.f, -10.f, -10.f);
    instanceQ.vtx[1] = make_float3(10.f, 10.f, 10.f);
    instanceM.vtx[0] = make_float3(-10.f, -10.f, -10.f);
    instanceM.vtx[1] = make_float3(10.f, 10.f, 10.f);

    instanceQ.setIndex(124u);
    instanceM.index = 124u;

    if (instanceQ.getIndex() != instanceM.index)
    {
        cudastd::logger::out << "Object index mismatch! \n";
        return 1;
    }

    float3 RayOrg;
    float3 RayDirRCP;
    
    ///////////////////////////////////////////////////////
    //Test Identity
    m00 = 1.f; m10 = 0.f; m20 = 0.f; m30 = 0.f;
    m01 = 0.f; m11 = 1.f; m21 = 0.f; m31 = 0.f;
    m02 = 0.f; m12 = 0.f; m22 = 1.f; m32 = 0.f;

    instanceQ.setIdentityTransormation();
    instanceM.setIdentityTransormation();

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "Identity Transform: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "Identity Transform: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "Identity Transform: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R 90 deg y, T 0.5
    m00 = 0.f; m10 = 0.f; m20 = 1.f; m30 = 0.5f;
    m01 = 0.f; m11 = 1.f; m21 = 0.f; m31 = 0.5f;
    m02 =-1.f; m12 = 0.f; m22 = 0.f; m32 = 0.5f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R 90 deg y, T 0.5: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R 90 deg y, T 0.5: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R 90 deg y, T 0.5: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R 180 deg y, T -0.5
    m00 = -1.f; m10 =  0.f; m20 =  0.f; m30 = -0.5f;
    m01 =  0.f; m11 =  1.f; m21 =  0.f; m31 = -0.5f;
    m02 =  0.f; m12 =  0.f; m22 = -1.f; m32 = -0.5f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R 180 deg y, T -0.5: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R 180 deg y, T -0.5: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R 180 deg y, T -0.5: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R 90 deg y 90 deg x, T 0.1 0.2 0.3
    m00 = 0.f; m10 = -1.f; m20 = 0.f; m30 = 0.1f;
    m01 = 1.f; m11 =  0.f; m21 = 0.f; m31 = 0.2f;
    m02 = 0.f; m12 =  0.f; m22 = 1.f; m32 = 0.3f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R 90 deg y 90 deg x, T 0.1 0.2 0.3: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R 90 deg y 90 deg x, T 0.1 0.2 0.3: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R 90 deg y 90 deg x, T 0.1 0.2 0.3: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R -90 deg x, T 0
    m00 = 1.f; m10 = 0.f; m20 = 0.f; m30 = 0.f;
    m01 = 0.f; m11 = 0.f; m21 = 1.f; m31 = 0.f;
    m02 = 0.f; m12 = -1.f; m22 = 0.f; m32 = 0.f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R -90 deg x: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R -90 deg x: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R -90 deg x: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R -90 deg x, 90 deg y, T 0.2
    m00 = 0.f; m10 = -1.f; m20 = 0.f; m30 = 0.2f;
    m01 = 0.f; m11 = 0.f; m21 = 1.f; m31 = 0.2f;
    m02 = -1.f; m12 = 0.f; m22 = 0.f; m32 = 0.2f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R -90 deg x, 90 deg y, T 0.2, mirror col 0
    m00 = 0.f; m10 = -1.f; m20 = 0.f; m30 = 0.2f;
    m01 = 0.f; m11 = 0.f; m21 = 1.f; m31 = 0.2f;
    m02 = 1.f; m12 = 0.f; m22 = 0.f; m32 = 0.2f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 0: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 0: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 0: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    //R -90 deg x, 90 deg y, T 0.2, mirror col 1
    m00 = 0.f; m10 = 1.f; m20 = 0.f; m30 = 0.2f;
    m01 = 0.f; m11 = 0.f; m21 = 1.f; m31 = 0.2f;
    m02 = -1.f; m12 = 0.f; m22 = 0.f; m32 = 0.2f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 1: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 1: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 1: transformed rays do not match! \n";
            return 1;
        }

    }
    ///////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    //R -90 deg x, 90 deg y, T 0.2, mirror col 2
    m00 = 0.f; m10 = -1.f; m20 = 0.f; m30 = 0.2f;
    m01 = 0.f; m11 = 0.f; m21 = -1.f; m31 = 0.2f;
    m02 = -1.f; m12 = 0.f; m22 = 0.f; m32 = 0.2f;

    instanceQ.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceM.setTransformation(
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32);

    instanceQ.getTransformation(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32);

    instanceM.getTransformation(
        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32);

    if (!checkTransformations(
        Qm00, Qm10, Qm20, Qm30,
        Qm01, Qm11, Qm21, Qm31,
        Qm02, Qm12, Qm22, Qm32,

        Mm00, Mm10, Mm20, Mm30,
        Mm01, Mm11, Mm21, Mm31,
        Mm02, Mm12, Mm22, Mm32))
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 2: matrix mismatch! \n";
        return 1;
    }

    if (instanceQ.isIdentityTransformation() != instanceM.isIdentityTransformation())
    {
        cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 2: identity check failed! \n";
        return 1;
    }

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        initRay(RayOrg, RayDirRCP);

        float3 RayOrgQ = RayOrg;
        float3 RayDirRCPQ = RayDirRCP;

        float3 RayOrgM = RayOrg;
        float3 RayDirRCPM = RayDirRCP;

        instanceQ.transformRay(RayOrgQ, RayDirRCPQ);
        instanceM.transformRay(RayOrgM, RayDirRCPM);

        if (!checkRay(RayOrgQ, RayDirRCPQ, RayOrgM, RayDirRCPM))
        {
            cudastd::logger::out << "R -90 deg x, 90 deg y, T 0.2, mirror col 2: transformed rays do not match! \n";
            return 1;
        }
    }

    return 0;

}

