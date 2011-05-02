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

#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Primitive/Camera.h"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UGridSortBuilder.h"
#include "RT/Structure/TLGridMemoryManager.h"
#include "RT/Algorithm/TLGridSortBuilder.h"
#include "RT/Algorithm/RayGenerators.h"
#include "RT/Algorithm/RayTriangleIntersector.h"

#include "RT/Integrator/SimpleIntegrator.h"

typedef Triangle    t_Primitive;

#define TLGRID

#ifdef TLGRID
typedef TLGridMemoryManager             t_MemoryManager;
typedef TLGridSortBuilder<t_Primitive>  t_AccStructBuilder;
typedef TwoLevelGrid                    t_AccStruct;
float                                   sTopLevelDensity = 0.0625f;
float                                   sLeafLevelDensity = 1.2f;
#else
typedef UGridMemoryManager              t_MemoryManager;
typedef UGridSortBuilder<t_Primitive>   t_AccStructBuilder;
typedef UniformGrid                     t_AccStruct;
float                                   sTopLevelDensity = 5.f;
float                                   sLeafLevelDensity = 1.2f; //dummy
#endif

int                                             sFrameId = 0;
PrimitiveArray<t_Primitive>                     sTriangleArray;
PrimitiveAttributeArray<t_Primitive, float3>    sTriangleNormalArray;
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
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorReg;

SimpleIntegrator<
    Triangle,
    RandomPrimaryRayGenerator< GaussianPixelSampler, true >,
    t_AccStruct,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorRnd;


void RTEngine::init()
{}

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

}

void RTEngine::buildAccStruct()

{
    sBuilder.init(sMemoryManager, sTriangleArray.numPrimitives, sTopLevelDensity, sLeafLevelDensity);
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

}

void RTEngine::renderFrame(FrameBuffer& aFrameBuffer, const int aImageId)
{
    t_AccStruct grid = sMemoryManager.getParameters();

    if(aImageId < 4)
    {
        sRegularRayGen.dcImageId = aImageId;
        sSimpleIntegratorReg.integrate(sTriangleArray, sTriangleNormalArray, grid, sRegularRayGen, aFrameBuffer, aImageId);
    }
    else
    {
        sRandomRayGen.dcImageId = aImageId;
        sSimpleIntegratorRnd.integrate(sTriangleArray, sTriangleNormalArray, grid, sRandomRayGen, aFrameBuffer, aImageId);

    }
}

void RTEngine::cleanup()
{
    sMemoryManager.cleanup();
    sTriangleArray.cleanup();
}
