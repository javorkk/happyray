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

#include "CUDAStdAfx.h"
#include "Textures.h"
#include "RT/RTEngine.h"

#include "Core/Algebra.hpp"

#include "Application/WFObjectUploader.h"

#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/TexturedPrimitiveArray.h"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Primitive/Camera.h"
#include "RT/Primitive/Material.hpp"
#include "RT/Structure/3DTextureMemoryManager.h"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UGridSortBuilder.h"
#include "RT/Structure/TLGridMemoryManager.h"
#include "RT/Algorithm/TLGridSortBuilder.h"
#include "RT/Algorithm/RayGenerators.h"
#include "RT/Algorithm/RayTriangleIntersector.h"

#include "RT/Integrator/SimpleIntegrator.h"
#include "RT/Integrator/PathTracer.h"
#include "RT/Integrator/AOIntegrator.h" //USE_3D_TEXTURE defined there
#include "RT/Integrator/TLGridHierarchyAOIntegrator.h"
#include "RT/Integrator/AORayExporter.h"
#include "RT/Integrator/SimpleRayTraverser.h"

//////////////////////////////////////////////////////////////////////////
#include "RT/Structure/TwoLevelGridHierarchy.h"
#include "RT/Structure/TLGridHierarchyMemoryManager.h"
#include "RT/Algorithm/TLGridHierarchySortBuilder.h"
#include "RT/Algorithm/TLGridHierarchyTraverser.h"
//////////////////////////////////////////////////////////////////////////



typedef Triangle    t_Primitive;

//#define TLGRIDHIERARCHY
#define TLGRID

#ifdef TLGRIDHIERARCHY 
typedef TLGridHierarchyMemoryManager                t_MemoryManager;
typedef TLGridHierarchySortBuilder                  t_AccStructBuilder;
typedef TwoLevelGridHierarchy                       t_AccStruct;
#define Traverser_t                                 TLGridHierarchyTraverser
float                                               sTopLevelDensity = 1.2f;
float                                               sLeafLevelDensity = 1.2f;
#elif defined TLGRID
typedef TLGridMemoryManager             t_MemoryManager;
typedef TLGridSortBuilder<t_Primitive>  t_AccStructBuilder;
typedef TwoLevelGrid                    t_AccStruct;
#define Traverser_t                     TLGridTraverser
float                                   sTopLevelDensity = 0.12f;//1.2f;// 0.0625f;
float                                   sLeafLevelDensity =  2.4f;   //5.f;// 2.2f;
#else
const bool exact  = true; //true = exact triangle insertion, false = fast construction
typedef UGridMemoryManager              t_MemoryManager;
typedef UGridSortBuilder<t_Primitive, exact>   t_AccStructBuilder;
typedef UniformGrid                     t_AccStruct;
#define Traverser_t                     UGridTraverser
float                                   sTopLevelDensity = 0.12f;
float                                   sLeafLevelDensity = 1.2f; //dummy
#endif

int                                             sFrameId = 0;
TexturedPrimitiveArray<t_Primitive>             sTriangleArray;
VtxAttributeArray<t_Primitive, float3>          sTriangleNormalArray;
PrimitiveAttributeArray<PhongMaterial>          sMaterialArray;
PrimitiveAttributeArray<TexturedPhongMaterial>  sTexMaterialArray;
TextureMemoryManager                            sDefaultTextureManager;
AreaLightSourceCollection                       sLights;
t_MemoryManager                                 sMemoryManager;
t_AccStructBuilder                              sBuilder;

Camera                                          sCamera;
int                                             sResX;
int                                             sResY;

RegularPrimaryRayGenerator< RegularPixelSampler<2,2>, true >
    sRegularRayGen;

RandomPrimaryRayGenerator< GaussianPixelSampler, true >
    sRandomRayGen;

SimpleIntegrator<
    Triangle,
    RegularPrimaryRayGenerator< RegularPixelSampler<2,2>, true >,
    t_AccStruct,
    Traverser_t,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorReg;

SimpleIntegrator<
    Triangle,
    RandomPrimaryRayGenerator< GaussianPixelSampler, true >,
    t_AccStruct,
    Traverser_t,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorRnd;

bool sUseInputRays = false;
SimpleRayBuffer sSimpleRayBuffer(NULL);
RayLoader<SimpleRayBuffer> sSimpleRayLoader(sSimpleRayBuffer);

SimpleRayTraverser<
    Triangle,
    RayLoader< SimpleRayBuffer >,
    t_AccStruct,
    Traverser_t,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleRayTraverser;

PathTracer<
    Triangle,
    t_AccStruct,
    Traverser_t,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sPathTracer;

AOIntegrator<
    Triangle,
    t_AccStruct,
    Traverser_t,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sAOIntegrator;

//AORayExporter<
//    Triangle,
//    t_AccStruct,
//    Traverser_t,
//    MollerTrumboreIntersectionTest,
//    MollerTrumboreIntersectionTest
//>                           sAOIntegrator;

#ifdef TLGRIDHIERARCHY
TLGHAOIntegrator<
    Triangle,
    TLGridHierarchyTraverser,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sTLGHAOIntegrator;
#endif

void RTEngine::init()
{
    sResX = 0;
    sResY = 0;
}

float RTEngine::getBoundingBoxDiagonalLength()
{
    return len(sMemoryManager.bounds.vtx[1] - sMemoryManager.bounds.vtx[0]);
}

void RTEngine::upload(
    const WFObject& aFrame1,
    const WFObject& aFrame2,
    const float aCoeff)
{
    ObjUploader uploader;

    uploader.uploadObjFrameVertexData(
        aFrame1, aFrame2, aCoeff, 
        sMemoryManager.bounds.vtx[0],
        sMemoryManager.bounds.vtx[1], sTriangleArray);

    uploader.uploadObjFrameVertexIndexData(
        aFrame1, aFrame2, sTriangleArray);

    uploader.uploadObjFrameNormalData(
        aFrame1, aFrame2, 0.f, sTriangleNormalArray);

    uploader.uploadObjFrameNormalIndexData(
        aFrame1, aFrame2, sTriangleNormalArray);

    uploader.uploadObjFrameMaterialData(aFrame2, sMaterialArray);

#ifdef USE_3D_TEXTURE
    bool updateTexture = sDefaultTextureManager.bounds.vtx[0].x != sMemoryManager.bounds.vtx[0].x ||
        sDefaultTextureManager.bounds.vtx[0].y != sMemoryManager.bounds.vtx[0].y ||
        sDefaultTextureManager.bounds.vtx[0].z != sMemoryManager.bounds.vtx[0].z ||
        sDefaultTextureManager.resX != sMemoryManager.resX ||
        sDefaultTextureManager.resY != sMemoryManager.resY ||
        sDefaultTextureManager.resZ != sMemoryManager.resZ;

    if(updateTexture)
    {
        sDefaultTextureManager.bounds = sMemoryManager.bounds;
        sDefaultTextureManager.resX = cudastd::max(sMemoryManager.resX, 1);
        sDefaultTextureManager.resY = cudastd::max(sMemoryManager.resX, 1);
        sDefaultTextureManager.resZ = cudastd::max(sMemoryManager.resX, 1);
        sDefaultTextureManager.allocateDataHost();
        for(int z = 0; z < sDefaultTextureManager.resZ; ++z)
        {
            for(int y = 0; y < sDefaultTextureManager.resY; ++y)
            {
                for(int x = 0; x < sDefaultTextureManager.resX; ++x)
                {
                    bool oddCell = (x + y + z) % 2 == 1;
                    float3 texColor;
                    if(oddCell)
                    {
                        texColor = make_float3(0.15f, 0.2f, 0.3f);
                    }
                    else
                    {
                        texColor = make_float3(0.7f, 0.7f, 0.7f);
                    }
                    sDefaultTextureManager.getTexel(x,y,z) = texColor;
                }
            }
        }
        sDefaultTextureManager.allocateDataDevice();
        sDefaultTextureManager.copyDataHostToDevice();

        uploader.uploadObjFrameTextureData(aFrame2, sTexMaterialArray, sDefaultTextureManager);
    }
#endif

    sAOIntegrator.setAlpha(len(sMemoryManager.bounds.vtx[1] - sMemoryManager.bounds.vtx[0]) * 0.05f);

#ifdef TLGRIDHIERARCHY
    sTLGHAOIntegrator.setAlpha(len(sMemoryManager.bounds.vtx[1] - sMemoryManager.bounds.vtx[0]) * 0.05f);

    if(aFrame1.getNumInstances() <= 1u)
        return;

    sMemoryManager.allocateGrids(aFrame1.getNumObjects());
    for (size_t it = 0; it < aFrame1.getNumObjects(); ++it)
    {
        int2 primitiveRange = aFrame1.getObjectRange(it);
        sMemoryManager.primitiveCounts[it] = primitiveRange.y - primitiveRange.x;

        float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int primId = primitiveRange.x; primId < primitiveRange.y; ++primId)
        {
            WFObject::Face face = aFrame1.getFace(primId);
            const float3& vtx1 = aFrame1.getVertex(face.vert1);
            minBound = min(minBound, vtx1);
            maxBound = max(maxBound, vtx1);

            const float3& vtx2 = aFrame1.getVertex(face.vert2);
            minBound = min(minBound, vtx2);
            maxBound = max(maxBound, vtx2);

            const float3& vtx3 = aFrame1.getVertex(face.vert3);
            minBound = min(minBound, vtx3);
            maxBound = max(maxBound, vtx3);
        }

        minBound.x = minBound.x - 10.f * EPS;
        minBound.y = minBound.y - 10.f * EPS;
        minBound.z = minBound.z - 10.f * EPS;
        maxBound.x = maxBound.x + 10.f * EPS;
        maxBound.y = maxBound.y + 10.f * EPS;
        maxBound.z = maxBound.z + 10.f * EPS;

        if (minBound.x >= maxBound.x ||
            minBound.y >= maxBound.y ||
            minBound.z >= maxBound.z
            )
        {
            cudastd::logger::out << "Empty bounding box! Tile id: " << it << "\n";
        }
        sMemoryManager.gridsHost[it].vtx[0].x = minBound.x;
        sMemoryManager.gridsHost[it].vtx[0].y = minBound.y;
        sMemoryManager.gridsHost[it].vtx[0].z = minBound.z;
        sMemoryManager.gridsHost[it].vtx[1].x = maxBound.x;
        sMemoryManager.gridsHost[it].vtx[1].y = maxBound.y;
        sMemoryManager.gridsHost[it].vtx[1].z = maxBound.z;

        float3 diagonal = sMemoryManager.gridsHost[it].vtx[1] - sMemoryManager.gridsHost[it].vtx[0];
        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = sLeafLevelDensity;
        const float magicConstant =
            powf(lambda * static_cast<float>(sMemoryManager.primitiveCounts[it]) / volume, 0.3333333f);

        float3 resolution = diagonal * magicConstant;
        int resX = static_cast<int>(resolution.x);
        int resY = static_cast<int>(resolution.y);
        int resZ = static_cast<int>(resolution.z);
        sMemoryManager.gridsHost[it].res[0] = resX > 0 ? resX : 1;
        sMemoryManager.gridsHost[it].res[1] = resY > 0 ? resY : 1;
        sMemoryManager.gridsHost[it].res[2] = resZ > 0 ? resZ : 1;

        sMemoryManager.gridsHost[it].setCellSize((sMemoryManager.gridsHost[it].vtx[1] - sMemoryManager.gridsHost[it].vtx[0]) / make_float3(sMemoryManager.gridsHost[it].res[0], sMemoryManager.gridsHost[it].res[1], sMemoryManager.gridsHost[it].res[2]));
        sMemoryManager.gridsHost[it].setCellSizeRCP(make_float3(sMemoryManager.gridsHost[it].res[0], sMemoryManager.gridsHost[it].res[1], sMemoryManager.gridsHost[it].res[2]) / (sMemoryManager.gridsHost[it].vtx[1] - sMemoryManager.gridsHost[it].vtx[0]));
    }
    sMemoryManager.copyGridsHostToDevice();

    //Instances
    sMemoryManager.allocateDeviceInstances(aFrame1.getNumInstances());
    sMemoryManager.allocateHostInstances(aFrame1.getNumInstances());

    for (size_t it = 0; it < aFrame1.getNumInstances(); ++it)
    {   
        const WFObject::Instance& instance = aFrame1.getInstance(it);
        sMemoryManager.instancesHost[it].setIndex(instance.objectId);
        sMemoryManager.instancesHost[it].setTransformation(
            instance.m00, instance.m10, instance.m20, instance.m30,
            instance.m01, instance.m11, instance.m21, instance.m31,
            instance.m02, instance.m12, instance.m22, instance.m32);
#ifndef COMPACT_INSTANCES
        sMemoryManager.instancesHost[it].vtx[0] = instance.min;
        sMemoryManager.instancesHost[it].vtx[1] = instance.max;
#endif
    }

    sMemoryManager.copyInstancesHostToDevice();
#endif

}

void RTEngine::buildAccStruct()
{
#ifdef TLGRIDHIERARCHY
    sBuilder.init(sMemoryManager, sMemoryManager.instancesSize / sizeof(GeometryInstance), sTopLevelDensity, sLeafLevelDensity);
    sBuilder.build(sMemoryManager, sMemoryManager.gridsSize / sizeof(UniformGrid), sMemoryManager.primitiveCounts, sTriangleArray);
#else
    sBuilder.init(sMemoryManager, (uint)sTriangleArray.numPrimitives, sTopLevelDensity, sLeafLevelDensity);
    sBuilder.build(sMemoryManager, sTriangleArray);
#endif
    //TLGridHierarchySortBuilder< t_Primitive > dbgBuilder;
    //TLGridHierarchyMemoryManager dbgMemManager;
    //dbgMemManager.bounds = sMemoryManager.bounds;
    //dbgBuilder.init(dbgMemManager, (uint)sTriangleArray.numPrimitives, sTopLevelDensity, 5.f);
    //dbgBuilder.build(dbgMemManager, sTriangleArray);
}

void RTEngine::setCamera(
    const float3& aPosition,
    const float3& aOrientation,
    const float3& aUp,
    const float   aFOV,
    const int     aX,
    const int     aY )
{
    sCamera.init(aPosition, aOrientation, aUp, aFOV, aX, aY);

    sResX = aX;
    sResY = aY;

    sRegularRayGen.dcCamera = sCamera;
    sRegularRayGen.dcRegularPixelSampler.resX = (float)aX;
    sRegularRayGen.dcRegularPixelSampler.resY = (float)aY;
    sRandomRayGen.dcCamera  = sCamera;
    sRandomRayGen.dcRandomPixelSampler.resX = (float)aX;
    sRandomRayGen.dcRandomPixelSampler.resY = (float)aY;
    sPathTracer.setResolution(aX,aY); //TODO: Only reserve memory if the integrator is used

}

void RTEngine::setLights(const AreaLightSourceCollection& aLights)
{
    sLights = aLights;
}

void RTEngine::renderFrame(FrameBuffer& aFrameBuffer, const int aImageId, const int aRenderMode)
{
#ifdef TLGRIDHIERARCHY
    t_AccStruct grid = sMemoryManager.getParameters();
    sRandomRayGen.sampleId = aImageId;
    sTLGHAOIntegrator.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray, grid, sRandomRayGen, aFrameBuffer, aImageId);
#else
    t_AccStruct grid = sMemoryManager.getParameters();

    switch ( aRenderMode ) 
    {
    case 0:
        if (sUseInputRays)
        {
            sSimpleRayTraverser.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray, grid, sSimpleRayLoader, aFrameBuffer, 0);
        } 
        else if(aImageId < 4)
        {
            sRegularRayGen.sampleId = aImageId;
            sSimpleIntegratorReg.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray, grid, sRegularRayGen, aFrameBuffer, aImageId);
        }
        else
        {
            sRandomRayGen.sampleId = aImageId;
            sSimpleIntegratorRnd.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray, grid, sRandomRayGen, aFrameBuffer, aImageId);

        }
        break;
    case 1:
        sRandomRayGen.sampleId = aImageId;
        sPathTracer.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray, sLights, grid, sRandomRayGen, aFrameBuffer, aImageId);
        break;
    case 2:
        sRandomRayGen.sampleId = aImageId;
#ifdef USE_3D_TEXTURE
        sAOIntegrator.integrate(sTriangleArray, sTriangleNormalArray, sTexMaterialArray, grid, sRandomRayGen, aFrameBuffer, aImageId);
#else
        sAOIntegrator.integrate(sTriangleArray, sTriangleNormalArray, sMaterialArray,grid, sRandomRayGen, aFrameBuffer, aImageId);
#endif
    default:
        break;
    }//switch ( mRenderMode )
#endif
}

void RTEngine::setGridDensities(float aTopLvl, float aLeafLvl)
{
    sTopLevelDensity = aTopLvl;
    sLeafLevelDensity = aLeafLvl;
}

void RTEngine::cleanup()
{
    sMemoryManager.cleanup();
    sTriangleArray.cleanup();
    sTriangleNormalArray.cleanup();
    sMaterialArray.cleanup();
    sTexMaterialArray.cleanup();
    sDefaultTextureManager.cleanup();
    sSimpleIntegratorReg.cleanup();
    sSimpleIntegratorRnd.cleanup();
    sPathTracer.cleanup();
    sAOIntegrator.cleanup();
    
}

void RTEngine::setInputRayFileName(const std::string& aFileName)
{
    sSimpleRayTraverser.setInputRaysFileName(aFileName);
    sUseInputRays = true;
}

