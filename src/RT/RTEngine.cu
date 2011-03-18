#include "CUDAStdAfx.h"
#include "RT/RTEngine.h"

#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Primitive/Triangle.hpp"
#include "RT/Structure/UGridMemoryManager.h"
#include "RT/Algorithm/UGridSortBuilder.h"

float                       sGridDensity = 5.f;
int                         sFrameId = 0;
PrimitiveArray<Triangle>    sTriangleArray;
UniformGridMemoryManager    sUGridMemoryManager;
UGridSortBuilder<Triangle>  sGridBuilder;

void StaticRTEngine::init(
    const WFObject& aScene)
{
    ObjUploader uploader;

    uploader.uploadObjFrameVertexData(
        aScene, aScene, 0.f, 
        sUGridMemoryManager.bounds.vtx[0],
        sUGridMemoryManager.bounds.vtx[1], sTriangleArray);

    uploader.uploadObjFrameIndexData(
        aScene, aScene, sTriangleArray);

    uploader.uploadObjFrameNormalData(
        aScene, aScene, 0.f, sTriangleArray);

    sGridBuilder.init(sUGridMemoryManager, sTriangleArray.numPrimitives, sGridDensity);
    sGridBuilder.build(sUGridMemoryManager, sTriangleArray);
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