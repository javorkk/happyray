#include "CUDAStdAfx.h"
#include "RT/RTEngine.h"

#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Primitive/Camera.h"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UGridSortBuilder.h"
#include "RT/Algorithm/RayGenerators.h"
#include "RT/Algorithm/RayTriangleIntersector.h"

#include "RT/Integrator/SimpleIntegrator.h"

float                       sGridDensity = 5.f;
int                         sFrameId = 0;
PrimitiveArray<Triangle>    sTriangleArray;
PrimitiveAttributeArray<Triangle, float3> sTriangleNormalArray;
UniformGridMemoryManager    sUGridMemoryManager;
UGridSortBuilder<Triangle>  sGridBuilder;
Camera                      sCamera;
int                         sResX;
int                         sResY;
   
RegularPrimaryRayGenerator< RegularPixelSampler<2,2>, true >
                            sRegularRayGen;

RandomPrimaryRayGenerator< GaussianPixelSampler, true >
                            sRandomRayGen;
  


SimpleIntegrator<
    Triangle,
    RegularPrimaryRayGenerator< RegularPixelSampler<2,2>, true >,
    UniformGrid,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorReg;

SimpleIntegrator<
    Triangle,
    RandomPrimaryRayGenerator< GaussianPixelSampler, true >,
    UniformGrid,
    MollerTrumboreIntersectionTest,
    MollerTrumboreIntersectionTest
>                           sSimpleIntegratorRnd;

void StaticRTEngine::init()
{}

void StaticRTEngine::upload(
    const WFObject& aScene)
{
    ObjUploader uploader;

    uploader.uploadObjFrameVertexData(
        aScene, aScene, 0.f, 
        sUGridMemoryManager.bounds.vtx[0],
        sUGridMemoryManager.bounds.vtx[1], sTriangleArray);

    uploader.uploadObjFrameVertexIndexData(
        aScene, aScene, sTriangleArray);

    uploader.uploadObjFrameNormalData(
        aScene, aScene, 0.f, sTriangleNormalArray);

    uploader.uploadObjFrameNormalIndexData(
        aScene, aScene, sTriangleNormalArray);


    sGridBuilder.init(sUGridMemoryManager, sTriangleArray.numPrimitives, sGridDensity);
    sGridBuilder.build(sUGridMemoryManager, sTriangleArray);
}

void StaticRTEngine::setCamera(
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

void StaticRTEngine::renderFrame(FrameBuffer& aFrameBuffer, const int aImageId)
{
   
    UniformGrid grid = sUGridMemoryManager.getParameters();

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

void StaticRTEngine::cleanup()
{
    sUGridMemoryManager.freeCellMemoryDevice();
    sUGridMemoryManager.freeCellMemoryHost();
    sUGridMemoryManager.freePairsBufferPair();
    sUGridMemoryManager.freeRefCountsBuffer();
    sUGridMemoryManager.freePrimitiveIndicesBuffer();
    sUGridMemoryManager.cleanup();
    
    sTriangleArray.cleanup();
}