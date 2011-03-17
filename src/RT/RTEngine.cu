#include "CUDAStdAfx.h"
#include "RT/RTEngine.h"

int                          StaticRTEngine::sFrameId = 0;
PrimitiveArray<Triangle>     StaticRTEngine::sTriangleArray;

void StaticRTEngine::init(
    const WFObject& aScene)
{
    ObjUploader uploader;

    float3 minBound, maxBound;
    uploader.uploadObjFrameVertexData(
        aScene, aScene, 0.f, minBound, maxBound, sTriangleArray);

    uploader.uploadObjFrameIndexData(
        aScene, aScene, sTriangleArray);

    uploader.uploadObjFrameNormalData(
        aScene, aScene, 0.f, sTriangleArray);
}
