#ifdef _MSC_VER
#pragma once
#endif

#ifndef PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF
#define PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF

#include "CUDAStdAfx.h"
#include "DeviceConstants.h"
#include "Core/Algebra.hpp"
#include "Core/Flags.hpp"

#include "RT/Primitive/Material.hpp"
#include "RT/Primitive/LightSource.hpp"
#include "RT/Structure/FrameBuffer.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Structure/RayBuffers.h"
#include "RT/Structure/MemoryManager.h"

#include "RT/Algorithm/RayTracingKernels.h"
#include "Utils/RandomNumberGenerators.hpp"

static const int RFRACTION_PATH_MAX_DEPTH = 8;
static const int OCCLUSIONSAMPLESX = 1;
static const int OCCLUSIONSAMPLESY = 1;
static const int NUMOCCLUSIONSAMPLES  = OCCLUSIONSAMPLESX * OCCLUSIONSAMPLESY;

//////////////////////////////////////////////////////////////////////////
//in DeviceConstants.h:
//DEVICE_NO_INLINE CONSTANT PrimitiveAttributeArray<PhongMaterial>                    dcMaterialStorage;
//DEVICE_NO_INLINE CONSTANT uint                                                      dcNumPixels;
//DEVICE_NO_INLINE CONSTANT float dcPrimesRCP[] = {0.5f, 0.333333f, 0.2f, 0.142857f,
//0.09090909f, 0.07692307f, 0.058823529f, 0.0526315789f, 0.04347826f,
//0.034482758f, 0.032258064f};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//Path Tracing Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template< 
    class tPrimitive,
    class tAccelerationStructure,
    template <class, class, bool> class tTraverser,
    class tIntersector >
        GLOBAL void tracePath(
        PrimitiveArray<tPrimitive>                                  aStorage,
        VtxAttributeArray<tPrimitive, float3>                       aNormalStorage,
        tAccelerationStructure                                      dcGrid,
        SimpleRayBuffer                                             aRayBuffer,
        const int                                                   aImageId,
        ImportanceBuffer                                            oImportanceBuffer,
        int*                                                        aGlobalMemoryPtr)
    {
        //typedef KISSRandomNumberGenerator                           t_RNG;
        typedef SimpleRandomNumberGenerator                           t_RNG;


        extern SHARED uint sharedMem[];

        //float3* rayOrg =
        //    (float3*)(sharedMem + t_Intersector::SHAREDMEMSIZE);
        //float3* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

#if __CUDA_ARCH__ >= 110
        volatile uint*  nextRayArray = sharedMem;
        volatile uint*  rayCountArray = nextRayArray + RENDERTHREADSY;

        if (threadId1DInWarp32() == 0u)
        {
            rayCountArray[warpId32()] = 0u;
        }

        volatile uint& localPoolNextRay = nextRayArray[warpId32()];
        volatile uint& localPoolRayCount = rayCountArray[warpId32()];

        while (true)
        {
            if (localPoolRayCount==0 && threadId1DInWarp32() == 0)
            {
                localPoolNextRay = atomicAdd(&aGlobalMemoryPtr[0], BATCHSIZE);
                localPoolRayCount = BATCHSIZE;
            }
            // get rays from local pool
            uint myRayIndex = localPoolNextRay + threadId1DInWarp32();
            if (ALL(myRayIndex >= dcNumPixels))
            {
                return;
            }

            if (myRayIndex >= dcNumPixels) //keep whole warp active
            {
                myRayIndex = dcNumPixels;
            }

            if (threadId1DInWarp32() == 0)
            {
                localPoolNextRay += WARPSIZE;
                localPoolRayCount -= WARPSIZE;
            }
#else
        for(uint myRayIndex = globalThreadId1D(); myRayIndex < dcNumPixels;
            myRayIndex += numThreads())
        {
#endif
            //////////////////////////////////////////////////////////////////////////
            //Initialization

            //1st flag: has valid intersection at begin of path
            //2nd flag: still has valid intersection
            //3rd flag: intersected primitive is transparent
            //4th flag: true = reflect, false = refract
            //data: path depth
            flag4 myFlags;

            uint* sharedMemNew = sharedMem + RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2;
            float3 rayOrg = rep(0.f);
            float3& rayDir = (((float3*)sharedMemNew)[threadId1D32()]);

            float rayT = -1.f;
            uint  bestHit = aStorage.numPrimitives;


            myFlags.setFlag1(myRayIndex < dcNumPixels);
            if (myFlags.getFlag1())
            {
                rayT = aRayBuffer.loadDistance(myRayIndex, dcNumPixels);
                bestHit = aRayBuffer.loadBestHit(myRayIndex, dcNumPixels);
                myFlags.setFlag2(rayT < FLT_MAX && bestHit < aStorage.numPrimitives);
            }

            if (!myFlags.getFlag1() || !myFlags.getFlag2())
            {
                rayT = -1.f;
                myFlags.setFlag2To0();
                myFlags.setFlag3To0();
            }

            //t_RNG genRand(3643u + myRayIndex * 4154207u * dcSeed + myRayIndex,
            //    1761919u + myRayIndex * 2746753u * dcSeed /*+ globalThreadId1D(8116093u)*/,
            //    331801u + myRayIndex + dcSeed /*+ globalThreadId1D(91438197u)*/,
            //    10499029u);
            t_RNG genRand(globalThreadId1D() * globalThreadId1D() * globalThreadId1D() + 
                1761919u + myRayIndex * 2746753u * aImageId +
                (myRayIndex + aImageId * dcNumPixels) * 
                (myRayIndex + aImageId * dcNumPixels) *
                (myRayIndex + aImageId * dcNumPixels) );

            float3 attenuation = rep(1.f);

            float indexOfRefraction;

            if (myFlags.getFlag2())
            {
                rayOrg = aRayBuffer.loadOrigin(myRayIndex, dcNumPixels);
                rayDir = aRayBuffer.loadDirection(myRayIndex, dcNumPixels);
                PhongMaterial material = dcMaterialStorage[bestHit];
                myFlags.setFlag3(material.isTransparent(indexOfRefraction));

                if (!myFlags.getFlag3())
                {
                    float3 diffReflectance = material.getDiffuseReflectance();
                    float albedoD = (diffReflectance.x * 0.222f +
                        diffReflectance.y * 0.7067f +
                        diffReflectance.z * 0.0713f) * M_PI;
                    float3 specReflectance = material.getSpecularReflectance();
                    float specExponent = material.getSpecularExponent();
                    float albedoS = (specReflectance.x * 0.222f +
                        specReflectance.y * 0.7067f +
                        specReflectance.z * 0.0713f) * 2.f * M_PI / (specExponent + 2);
                    
                    float rnum = genRand();
                    if (rnum < albedoD)
                    {
                        myFlags.setFlag4(true);
                        attenuation *= rep(1.f / albedoD);
                        //pi is to account for cosine weighted hemisphere sampling
                        attenuation.x = attenuation.x * diffReflectance.x * M_PI;
                        attenuation.y = attenuation.y * diffReflectance.y * M_PI;
                        attenuation.z = attenuation.z * diffReflectance.z * M_PI;
                    }
                    else if (rnum < albedoD + albedoS)
                    {
                        myFlags.setFlag4To0();
                        attenuation *= rep(1.f / albedoS);
                        //(specExp + 1)/(2 * Pi) is to account for power-cosine weighted hemisphere sampling
                        attenuation.x = attenuation.x * specReflectance.x / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                        attenuation.y = attenuation.y * specReflectance.y / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                        attenuation.z = attenuation.z * specReflectance.z / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                    }
                    else
                    {
                        attenuation *= rep(1.f / (1.f - albedoS - albedoD));
                        myFlags.setFlag2To0();
                        rayT = -1.f;
                    }
                }
            }


            while(ANY(myFlags.getFlag2()))
            {
                if(myFlags.getFlag2())
                {
                    tPrimitive prim = aStorage[bestHit];
                    float3& vert0 = prim.vtx[0];
                    float3& vert1 = prim.vtx[1];
                    float3& vert2 = prim.vtx[2];

                    float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

                    //Compute barycentric coordinates
                    vert0 = vert0 - rayOrg;
                    vert1 = vert1 - rayOrg;
                    vert2 = vert2 - rayOrg;

                    float3 n0 = vert1 % vert2;
                    float3 n1 = vert2 % vert0;

                    float twiceSabc_RCP = lenRCP(realNormal);
                    float u = len(n0) * twiceSabc_RCP;
                    float v = len(n1) * twiceSabc_RCP;

                    VtxAttribStruct<tPrimitive, float3> normals;
                    normals = aNormalStorage[bestHit];
                    float3& normal0 = normals.data[0];
                    float3& normal1 = normals.data[1];
                    float3& normal2 = normals.data[2];

                    float3 normal = ~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);

                    if (myFlags.getFlag3()) //transparent
                    {
                        const float dotNormalRayDir = dot(normal, rayDir);

                        float n1 = 1.000293f, n2 = indexOfRefraction;
                        if(dotNormalRayDir > 0.f)
                        {
                            //inside translucent object
                            n1 = indexOfRefraction;
                            n2 = 1.000293f;
                            //Beer's law
                            const float3 diffuseCoefficient = dcMaterialStorage[bestHit].getDiffuseReflectance();
                            attenuation.x *= expf(-rayT * diffuseCoefficient.x * M_PI);
                            attenuation.y *= expf(-rayT * diffuseCoefficient.y * M_PI);
                            attenuation.z *= expf(-rayT * diffuseCoefficient.z * M_PI);

                        }

                        float fresnel0 = (n1-n2) * (n1-n2) / ((n1+n2)*(n1+n2));

                        const float fresnelCoeff = fresnel0 + (1.f - fresnel0) *
                            powf(1.f - fabsf(dotNormalRayDir), 5.f);

                        float n_i = n1/n2;
                        const float sinNormalRefrDirSQR = n_i * n_i *
                            (1 - dotNormalRayDir * dotNormalRayDir);

                        //choose the largest contribution
                        myFlags.setFlag4(sinNormalRefrDirSQR > 1.f);

                        if (sinNormalRefrDirSQR <= 1.f && aImageId > 0)
                        {
                            HaltonNumberGenerator genQuasiRand;
                            float qrand = genQuasiRand(aImageId, myFlags.getData(),
                                dcPrimesRCP);
                            if (qrand < fresnelCoeff)
                            {
                                myFlags.setFlag4(true);
                            }
                        }

                        if (myFlags.getFlag4())
                        {
                            //reflect
                            rayOrg = rayOrg - 5.f * EPS * rayDir;
                            rayDir = rayDir - 2 * dotNormalRayDir * normal;
                        }
                        else
                        {
                            //refract
                            float cosNormalRefrDir = sqrtf(1.f - n_i * n_i * sinNormalRefrDirSQR);

                            float normalFactor = n_i * fabsf(dotNormalRayDir) - cosNormalRefrDir;

                            if (dotNormalRayDir > 0.f) normal = -normal;

                            rayOrg = rayOrg - 5.f * EPS * normal;

                            rayDir = n_i * rayDir + normalFactor * normal;

                        }

                    }
                    else if (myFlags.getFlag4())
                    {
                        //bounce diffuse
                        CosineHemisphereSampler genDir;

                        float3 randDir = genDir(genRand(), genRand());
                        if(dot(normal,rayDir) > 0.f) normal = -normal;
                        //transform coordinates
                        float3 tangent, binormal;
                        getLocalCoordinates(normal, tangent, binormal);

                        rayDir = tangent * randDir.x +
                            binormal * randDir.y + normal * randDir.z;                        

                        rayOrg = rayOrg + normal * 5.f * EPS;
                    }
                    else if(bestHit < aStorage.numPrimitives)
                    {
                        //bounce specular
                        PowerCosineHemisphereSampler genDir;
                        
                        float3 randDir = genDir(genRand(), genRand(), dcMaterialStorage[bestHit].getSpecularExponent());

                        //transform coordinates
                        float3 up, right, forward;
                        up = rayDir - 2.f * dot(normal,rayDir) * normal;

                        getLocalCoordinates(up, right, forward);

                        rayDir = right * randDir.x +
                            forward * randDir.y + up * randDir.z;                        

                        rayOrg = rayOrg + rayDir * 1E-3f;

                        attenuation *= dot(normal,rayDir);

                    }

                    rayT = FLT_MAX;

                    rayDir.x = 1.f / rayDir.x;
                    rayDir.y = 1.f / rayDir.y;
                    rayDir.z = 1.f / rayDir.z;

                }//endif thread is active

                //////////////////////////////////////////////////////////////////////////
                tTraverser<tPrimitive, tIntersector, false> traverse;
                traverse(aStorage, dcGrid, rayOrg, rayDir, rayT, bestHit, myFlags.getFlag2(), sharedMemNew);
                //////////////////////////////////////////////////////////////////////////

                if(myFlags.getFlag2())
                {

                    rayDir.x = 1.f / rayDir.x;
                    rayDir.y = 1.f / rayDir.y;
                    rayDir.z = 1.f / rayDir.z;


                    if(rayT >= FLT_MAX || myFlags.getData() >= RFRACTION_PATH_MAX_DEPTH)
                    {
                        //outside scene
                        if(rayT < FLT_MAX)
                            bestHit = dcGrid.primitives[bestHit];

                        aRayBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumPixels);
                        myFlags.setFlag2To0();

                        rayT = -1.f;
                    }
                    else
                    {
                        bestHit = dcGrid.primitives[bestHit];
                        PhongMaterial material = dcMaterialStorage[bestHit];
                        myFlags.setFlag3(material.isTransparent(indexOfRefraction));
                        if (!myFlags.getFlag3())
                        {
                            float3 diffReflectance = material.getDiffuseReflectance();
                            float albedoD = (diffReflectance.x * 0.222f +
                                diffReflectance.y * 0.7067f +
                                diffReflectance.z * 0.0713f) * M_PI;
                            float3 specReflectance = material.getSpecularReflectance();
                            float specExponent = material.getSpecularExponent();
                            float albedoS = (specReflectance.x * 0.222f +
                                specReflectance.y * 0.7067f +
                                specReflectance.z * 0.0713f) * 2.f * M_PI / (specExponent + 2.f);

                            //albedoS = (albedoS > 0.f)  ? fminf(1.f - albedoD, 0.25f): 0.f;//TODO: remove

                            float rnum = genRand();
                            if (rnum < albedoD)
                            {
                                myFlags.setFlag4(true);
                                attenuation = attenuation / albedoD;
                                //Pi is to account for cosine weighted hemisphere sampling
                                attenuation.x = attenuation.x * diffReflectance.x * M_PI;
                                attenuation.y = attenuation.y * diffReflectance.y * M_PI;
                                attenuation.z = attenuation.z * diffReflectance.z * M_PI;
                            }
                            else if (rnum < albedoD + albedoS)
                            {
                                myFlags.setFlag4To0();
                                attenuation = attenuation / albedoS;
                                //(specExp + 1)/(2 * Pi) is to account for power-cosine weighted hemisphere sampling
                                attenuation.x = attenuation.x * specReflectance.x / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                                attenuation.y = attenuation.y * specReflectance.y / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                                attenuation.z = attenuation.z * specReflectance.z / ((specExponent + 1.f) * 0.5f * M_PI_RCP);
                            }
                            else
                            {
                                attenuation = attenuation / (1.f - albedoS - albedoD);
                                aRayBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumPixels);
                                myFlags.setFlag2To0();
                                rayT = -1.f;
                            }
                        }

                        myFlags.setData(myFlags.getData() + 1);
                        rayOrg = rayOrg + rayT * rayDir;
                    }
                }//endif thread is active

            }//end while continue path

            if(myRayIndex < dcNumPixels)
                oImportanceBuffer.store(attenuation, myRayIndex);
        }//end while persistent threads
    }

//////////////////////////////////////////////////////////////////////////////////////////
//Direct Illumination Kernel
//////////////////////////////////////////////////////////////////////////////////////////

//assumes that block size is multiple of number of samples per area light
template< class tPrimitive >
GLOBAL void computeDirectIllumination(
                        PrimitiveArray<tPrimitive>              aStorage,
                        VtxAttributeArray<tPrimitive, float3>   aNormalStorage,
                        SimpleRayBuffer                         aInputBuffer,
                        DirectIlluminationBuffer                      aOcclusionBuffer,
                        AreaLightSourceCollection               aLSCollection,
                        FrameBuffer                             oFinalImage,
                        int                                     dcNumRays, //shadow rays
                        int                                     dcImageId,
                        float3*                                 aAttenuation = NULL)
{
    extern SHARED float3 sharedVec[];
    sharedVec[threadId1D()] = rep(0.f);

    float3 rayOrg = rep(0.f);
    float3 rayDir = rep(1.f);

    for(uint myRayIndex = globalThreadId1D();
        myRayIndex - threadId1D() < dcNumRays;
        myRayIndex += numThreads())
    {

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT = FLT_MAX;
        uint  bestHit = 0u;
        const uint myPixelIndex = min(dcNumPixels - 1, myRayIndex / NUMOCCLUSIONSAMPLES);

        SYNCTHREADS;

        //////////////////////////////////////////////////////////////////////////
        //load occlusion information in shared memory
        if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = aOcclusionBuffer.loadLSIntensity(myRayIndex);
            aInputBuffer.load(rayOrg, rayDir, rayT, bestHit, myPixelIndex, dcNumPixels);
        }
        //////////////////////////////////////////////////////////////////////////


        SYNCTHREADS;

        if (myRayIndex < dcNumRays && rayT < FLT_MAX )
        {
            if (sharedVec[threadId1D()].x + sharedVec[threadId1D()].y + sharedVec[threadId1D()].z > 0.f && bestHit < aStorage.numPrimitives)
            {
                tPrimitive prim = aStorage[bestHit];
                float3& vert0 = prim.vtx[0];
                float3& vert1 = prim.vtx[1];
                float3& vert2 = prim.vtx[2];

                float3 realNormal = (vert1 - vert0) % (vert2 - vert0);

                //Compute barycentric coordinates
                vert0 = vert0 - rayOrg;
                vert1 = vert1 - rayOrg;
                vert2 = vert2 - rayOrg;

                float3 n0 = vert1 % vert2;
                float3 n1 = vert2 % vert0;

                float twiceSabc_RCP = lenRCP(realNormal);
                float u = len(n0) * twiceSabc_RCP;
                float v = len(n1) * twiceSabc_RCP;

                VtxAttribStruct<tPrimitive, float3> normals;
                normals = aNormalStorage[bestHit];
                float3& normal0 = normals.data[0];
                float3& normal1 = normals.data[1];
                float3& normal2 = normals.data[2];

                float3 normal = ~(u * normal0 + v * normal1 + (1.f - u - v) * normal2);

                PhongMaterial material = dcMaterialStorage[bestHit];
                float3 diffReflectance = material.getDiffuseReflectance();
                float3 specReflectance = material.getSpecularReflectance();
                float  specExp = material.getSpecularExponent();

                float3 dirToLS = aOcclusionBuffer.loadDirToLS(myRayIndex, dcNumRays);

                float cosNormalEyeDir = dot(normal,dirToLS);

                float cosHalfVecLightDir = fmaxf(0.f,
                    dot((~(dirToLS - rayDir)),dirToLS));

                if (fabsf(dirToLS.x) +fabsf(dirToLS.y) + fabsf(dirToLS.z) > 0.f)
                {
                    float3 lsRadiance = sharedVec[threadId1D()];
                    
                    sharedVec[threadId1D()] = lsRadiance * fmaxf(0.f, cosNormalEyeDir);

                    sharedVec[threadId1D()].x *= diffReflectance.x;
                    sharedVec[threadId1D()].y *= diffReflectance.y;
                    sharedVec[threadId1D()].z *= diffReflectance.z;

                    float3 tmp = lsRadiance * fmaxf(0.f, cosNormalEyeDir) *
                            cosHalfVecLightDir *
                           fmaxf(0.f, fastPow(cosHalfVecLightDir, specExp));

                    tmp.x *= specReflectance.x;
                    tmp.y *= specReflectance.y;
                    tmp.z *= specReflectance.z;

                    sharedVec[threadId1D()] += tmp;
                }
            }
        }
        else if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = rep(0.f);
        }//endif hit point exists

        SYNCTHREADS;

        //one thread per pixel instead of per occlusion sample
        if (myRayIndex < dcNumRays && myRayIndex % NUMOCCLUSIONSAMPLES == 0u )
        {
            float3 oRadiance = rep(0.f);

            for(uint i = 0; i < NUMOCCLUSIONSAMPLES; ++i)
            {
                oRadiance =  oRadiance + sharedVec[threadId1D() + i] 
                * 1.f / (float)NUMOCCLUSIONSAMPLES;
            }

            float newSampleWeight = 1.f / (float)(dcImageId + 1);
            float oldSamplesWeight = 1.f - newSampleWeight;

            if (aAttenuation == NULL)
            {
                oFinalImage[myPixelIndex] =
                    oFinalImage[myPixelIndex] * oldSamplesWeight +
                    oRadiance * newSampleWeight;
            }
            else
            {
                oFinalImage[myPixelIndex] =
                    oFinalImage[myPixelIndex] * oldSamplesWeight +
                    oRadiance * aAttenuation[myPixelIndex] * newSampleWeight;
            }
        }

    }
}

//////////////////////////////////////////////////////////////////////////////////////////
//Final Image Assembly
//////////////////////////////////////////////////////////////////////////////////////////

static const int NUMTHREADS_ADD_II_X = 16;
static const int NUMTHREADS_ADD_II_Y = 16;

GLOBAL void addIndirectIllumination(
    float3* oFrameBuffer, //image to display
    float3* oResidue, //stores difference between image to display and the unbiased image
    float3* aAttenuation, //normalization factors for new illumination
    float3* aNewContribution, //new contribution
    int dcImageId,
    int dcResX,
    int dcResY
    )
{
    //first and last row and column are for padding
    SHARED float3 sharedMem[NUMTHREADS_ADD_II_X * NUMTHREADS_ADD_II_Y];

    uint mySharedMemIndex = threadId1D();

    //Coordinates of the input pixel of thread (0,0)
    uint2 blockOffset2D = make_uint2(
        blockIdx.x * (blockDim.x - 2),
        blockIdx.y * (blockDim.y - 2));

    //Coordinates of the input pixel
    uint2 threadOffset2D = make_uint2(
        max(1, threadIdx.x + blockOffset2D.x) - 1 ,
        max(1, threadIdx.y + blockOffset2D.y) - 1 );

    bool isInsideImage = threadOffset2D.x < dcResX + 1 && threadOffset2D.y < dcResY + 1;

    threadOffset2D = make_uint2(
        min(threadOffset2D.x, dcResX - 1),
        min(threadOffset2D.y, dcResY - 1));

    //linear index of the input pixel
    uint threadOffset1D = 
        threadOffset2D.x + threadOffset2D.y * dcResX;

    float3 pixelValue = rep(0.f);
    float3 newContribution = rep(0.f);
    float3 intensity = rep(1.f);
    float newIntensity = 0.f;

    const float newSampleWeight = 1.f / (float)(dcImageId + 1);
    const float oldSampleWeight = 1.f - newSampleWeight;

    if (threadOffset1D < dcNumPixels && isInsideImage)
    {
        pixelValue = oFrameBuffer[threadOffset1D];
        newContribution = aAttenuation[threadOffset1D] * aNewContribution[threadOffset1D];
        newContribution += oResidue[threadOffset1D];
    }

    newIntensity = dot(newContribution,intensity);
    sharedMem[mySharedMemIndex] = pixelValue * oldSampleWeight + newContribution * newSampleWeight;

    SYNCTHREADS;

    bool writeOutput = 
        threadIdx.x > 0u && threadIdx.x < blockDim.x - 1 &&
        threadIdx.y > 0u && threadIdx.y < blockDim.y - 1 &&
        threadOffset1D < dcNumPixels &&
        isInsideImage;

    if (writeOutput)
    {
        //Gaussian blurr with 3x3 kernel
        int bottom = threadId1D(threadIdx.x, threadIdx.y - 1);
        int top = threadId1D(threadIdx.x, threadIdx.y + 1);
        int bottomLeft = threadId1D(threadIdx.x - 1,threadIdx.y - 1);
        int bottomRight= threadId1D(threadIdx.x + 1, threadIdx.y - 1);
        int topLeft = threadId1D(threadIdx.x - 1, threadIdx.y + 1);
        int topRight = threadId1D(threadIdx.x + 1, threadIdx.y + 1);
        int Left =  threadId1D(threadIdx.x - 1, threadIdx.y);
        int Right = threadId1D(threadIdx.x + 1, threadIdx.y);

        float3 blurredIntensity = (
            1.f * sharedMem[topLeft]    +   2.f * sharedMem[top]    + 1.f * sharedMem[topRight] +
            2.f * sharedMem[Left]       +   4.f * sharedMem[top]    + 2.f * sharedMem[Right]    +
            1.f * sharedMem[bottomLeft] +   2.f * sharedMem[bottom] + 1.f * sharedMem[bottomRight]
        )/ 16.f;

        //float3 unbiasedPixelValue = pixelValue + indirectContribution;
        float variance = newIntensity;
        variance *= variance;


        if (variance < fmaxf(0.1f / static_cast<float>(dcImageId), 0.00005f))
        {
            oFrameBuffer[threadOffset1D] = pixelValue * oldSampleWeight + newContribution * newSampleWeight;
            oResidue[threadOffset1D] = rep(0.f);
        }
        else
        {
            float3 val = min(blurredIntensity, newContribution); 
            oFrameBuffer[threadOffset1D] = pixelValue * oldSampleWeight + val * newSampleWeight;
            oResidue[threadOffset1D] = newContribution - val;
        }

    }

}


    template<
        class tPrimitive,
        class tAccelerationStructure,
            template <class, class, bool> class tTraverser,
        class tPrimaryIntersector,
        class tAlternativeIntersector>

    class PathTracer
    {
        int* mGlobalMemoryPtr;
        size_t mGlobalMemorySize;
        cudaEvent_t mTrace, mShade;
        
        FrameBuffer mNewRadianceBuffer, mResidueBuffer;
    public:
        typedef RandomPrimaryRayGenerator< GaussianPixelSampler, true > t_PrimaryRayGenerator;
        typedef SimpleRayBuffer                                         t_RayBuffer;
        typedef tPrimaryIntersector                                     t_Intersector;
        typedef tAccelerationStructure                                  t_AccelerationStructure;
        typedef AreaLightShadowRayGenerator < OCCLUSIONSAMPLESX, OCCLUSIONSAMPLESY>  t_ShadowRayGenerator;
        typedef ImportanceBuffer                                        t_ImportanceBuffer;
        typedef DirectIlluminationBuffer                                      t_OcclusionBuffer;

        t_RayBuffer             rayBuffer;
      

        PathTracer():rayBuffer(t_RayBuffer(NULL)), 
            mGlobalMemorySize(0u)
        {}

        ~PathTracer()
        {
            cleanup();
        }

        HOST void setResolution(const int     aX, const int     aY)
        {            
            mNewRadianceBuffer.init(aX, aY);
            mNewRadianceBuffer.setZero();
           
            mResidueBuffer.init(aX, aY);
            mResidueBuffer.setZero();
        }

        HOST void integrate(
            PrimitiveArray<tPrimitive>&                     aStorage,
            VtxAttributeArray<tPrimitive, float3>&          aNormalStorage,
            PrimitiveAttributeArray<PhongMaterial>&         aMaterialStorage,
            AreaLightSourceCollection&                      aLightSources,
            t_AccelerationStructure&                        aAccStruct,
            t_PrimaryRayGenerator&                          aRayGenerator,
            FrameBuffer&                                    aFrameBuffer,
            const int                                       aImageId
            )
        {
            if(aLightSources.size() == 0u)
            {
                //Nothing to do
                return;
            }
            const uint sharedMemoryTrace = SHARED_MEMORY_TRACE;

            const uint numPixels = aFrameBuffer.resX * aFrameBuffer.resY;
            const uint globalMemorySize = sizeof(uint) +    //Persistent threads
                numPixels * sizeof(float3) +                //rayBuffer : rayOrg
                numPixels * sizeof(float3) +                //rayBuffer : rayDir
                numPixels * sizeof(float) +                 //rayBuffer : rayT
                numPixels * sizeof(uint) +                  //rayBuffer : primitive Id
                numPixels * sizeof(float3) +                //importanceBuffer : importance
                numPixels * NUMOCCLUSIONSAMPLES * (sizeof(float3) + sizeof(float3)) + //occlusion buffer: intensity and direction                
                0u;

            MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //primary rays
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );
            rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);

            dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
            dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );

            cudaEventCreate(&mTrace);

            trace<tPrimitive, tAccelerationStructure, t_PrimaryRayGenerator, t_RayBuffer, tTraverser, tPrimaryIntersector, false>
                <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>(
                aStorage,
                aRayGenerator,
                aAccStruct,
                rayBuffer,
                numPixels,
                mGlobalMemoryPtr);

            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Tracing primary rays failed!\n");

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //secondary rays (paths from hitpoints)
            //////////////////////////////////////////////////////////////////////////////////////////////////////

            MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );

            //MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGrid", &aAccStruct, sizeof(UniformGrid)) );
            //MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcFrameBuffer", &aFrameBuffer, sizeof(FrameBuffer)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcMaterialStorage", &aMaterialStorage, sizeof(PrimitiveAttributeArray<PhongMaterial>)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumPixels", &numPixels, sizeof(uint)) );

            t_ImportanceBuffer     importanceBuffer(mGlobalMemoryPtr + 
                1 +                                                         //Persistent threads
                numPixels * 3 + numPixels * 3 + numPixels + numPixels +     //rayBuffer
                0u);

            tracePath< tPrimitive, tAccelerationStructure, tTraverser, tPrimaryIntersector >
                <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>
                (aStorage, aNormalStorage, aAccStruct, rayBuffer, aImageId, importanceBuffer, mGlobalMemoryPtr);

            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Tracing (paths of)secondary rays  failed!\n");

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //shadow rays
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            t_OcclusionBuffer occlusionBuffer(mGlobalMemoryPtr + 
                1 +                             //Persistent threads
                numPixels * 3 +                 //rayBuffer : rayOrg
                numPixels * 3 +                 //rayBuffer : rayDir
                numPixels +                     //rayBuffer : rayT
                numPixels +                     //rayBuffer : primitive Id
                numPixels * 3 +                 //importanceBuffer : importance
                0u);

            t_ShadowRayGenerator   shadowRayGenerator(rayBuffer, occlusionBuffer, aLightSources, aImageId);


            MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );
            const uint numShadowRays = numPixels * NUMOCCLUSIONSAMPLES;
            
            trace<tPrimitive, tAccelerationStructure, t_ShadowRayGenerator, t_OcclusionBuffer, tTraverser, tPrimaryIntersector, true>
                <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>(
                aStorage,
                shadowRayGenerator,
                aAccStruct,
                occlusionBuffer,
                numShadowRays,
                mGlobalMemoryPtr);

            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Tracing shadow rays failed!\n");

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //direct illumination at the end of each path
            //////////////////////////////////////////////////////////////////////////////////////////////////////

            dim3 threadBlockDI( 24*NUMOCCLUSIONSAMPLES );
            dim3 blockGridDI  ( 120 );

            const uint sharedMemoryShade =
                threadBlockDI.x * sizeof(float3) +   //light vector   
                0u;

            computeDirectIllumination < tPrimitive >
                <<< blockGridDI, threadBlockDI, sharedMemoryShade>>>(
                aStorage,
                aNormalStorage,
                rayBuffer,
                occlusionBuffer,
                aLightSources,
                mNewRadianceBuffer,
                numShadowRays,
                aImageId,
                (float3*)importanceBuffer.getData());

            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Computing direct illumination failed!\n");

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //merge sample images
            //////////////////////////////////////////////////////////////////////////////////////////////////////


            dim3 threadBlockAdd( NUMTHREADS_ADD_II_X, NUMTHREADS_ADD_II_Y );
            dim3 blockGridAdd ( 
                aFrameBuffer.resX  / (NUMTHREADS_ADD_II_X - 2) + 1,
                aFrameBuffer.resY  / (NUMTHREADS_ADD_II_Y - 2) + 1);

            addIndirectIllumination
                <<< blockGridAdd, threadBlockAdd>>>(
                (float3*)aFrameBuffer.deviceData,
                (float3*)mResidueBuffer.deviceData,
                (float3*)importanceBuffer.getData(),
                (float3*)mNewRadianceBuffer.deviceData,
                aImageId,
                aFrameBuffer.resX,
                aFrameBuffer.resY);
            
            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Merging images failed!\n");
            cudaEventDestroy(mTrace);
        }

        HOST void cleanup()
        {
            if(mGlobalMemorySize != 0u)
            {
                MY_CUDA_SAFE_CALL( cudaFree(mGlobalMemoryPtr));
                mGlobalMemoryPtr = NULL;
                mGlobalMemorySize = 0u;
            }
            mNewRadianceBuffer.cleanup();
            mResidueBuffer.cleanup();
        }

    };

#endif // PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF
