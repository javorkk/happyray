#include "CUDAStdAfx.h"
#include "RT/RTEngine.h"

float                                       StaticRTEngine::sGridDensity = 5.f;
int                                         StaticRTEngine::sFrameId = 0;
PrimitiveArray<Triangle>                    StaticRTEngine::sTriangleArray;
UniformGridMemoryManager                    StaticRTEngine::sUGridMemoryManager;
UGridSortBuilder<Triangle>                  StaticRTEngine::sGridBuilder;

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
