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



typedef Triangle    t_Primitive;

#define TLGRID

#ifdef TLGRID
typedef TLGridMemoryManager             t_MemoryManager;
typedef TLGridSortBuilder<t_Primitive>  t_AccStructBuilder;
typedef TwoLevelGrid                    t_AccStruct;
#define Traverser_t                     TLGridTraverser
float                                   sTopLevelDensity = 0.0625f;
float                                   sLeafLevelDensity = 1.2f;
#else
const bool exact  = true; //true = exact triangle insertion, false = fast construction
typedef UGridMemoryManager              t_MemoryManager;
typedef UGridSortBuilder<t_Primitive, exact>   t_AccStructBuilder;
typedef UniformGrid                     t_AccStruct;
#define Traverser_t                     UGridTraverser
float                                   sTopLevelDensity = 5.f;
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


void RTEngine::init()
{
    sResX = 0;
    sResY = 0;
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

    sAOIntegrator.setAlpha(len(sMemoryManager.bounds.vtx[1]- sMemoryManager.bounds.vtx[0]) * 0.05f);

}

void RTEngine::buildAccStruct()

{
    sBuilder.init(sMemoryManager, (uint)sTriangleArray.numPrimitives, sTopLevelDensity, sLeafLevelDensity);
    sBuilder.build(sMemoryManager, sTriangleArray);
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
    t_AccStruct grid = sMemoryManager.getParameters();

    switch ( aRenderMode ) 
    {
    case 0:
        if(aImageId < 4)
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
}
