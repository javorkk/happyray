#ifdef _MSC_VER
#pragma once
#endif

#ifndef PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF
#define PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF

#include "CUDAStdAfx.h"
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

static const int RFRACTION_PATH_MAX_DEPTH = 16;

DEVICE_NO_INLINE CONSTANT RandomPrimaryRayGenerator< GaussianPixelSampler, true >   dcRayGenerator;
DEVICE_NO_INLINE CONSTANT FrameBuffer                                               dcFrameBuffer;
DEVICE_NO_INLINE CONSTANT PrimitiveAttributeArray<PhongMaterial>                    dcMaterialStorage;
DEVICE_NO_INLINE CONSTANT AreaLightSourceCollection                                 dcLightSources;
DEVICE_NO_INLINE CONSTANT uint                                                      dcNumRays;

DEVICE_NO_INLINE CONSTANT float dcPrimesRCP[] = {0.5f, 0.333333f, 0.2f, 0.142857f,
0.09090909f, 0.07692307f, 0.058823529f, 0.0526315789f, 0.04347826f,
0.034482758f, 0.032258064f};

template< 
    class tPrimitive,
    class tAccelerationStructure,
    template <class, class> class tTraverser,
    class tIntersector>
        GLOBAL void tracePath(
        PrimitiveArray<tPrimitive>                                  aStorage,
        VtxAttributeArray<tPrimitive, float3>                       aNormalStorage,
        tAccelerationStructure                                      dcGrid,
        SimpleRayBuffer                                             aRayBuffer,
        const int                                                   aImageId,
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
            if (ALL(myRayIndex >= dcNumRays))
            {
                return;
            }

            if (myRayIndex >= dcNumRays) //keep whole warp active
            {
                myRayIndex = dcNumRays;
            }

            if (threadId1DInWarp32() == 0)
            {
                localPoolNextRay += WARPSIZE;
                localPoolRayCount -= WARPSIZE;
            }
#else
        for(uint myRayIndex = globalThreadId1D(); myRayIndex < dcNumRays;
            myRayIndex += numThreads())
        {
#endif
            //////////////////////////////////////////////////////////////////////////
            //Initialization
            uint* sharedMemNew = sharedMem + RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2;
            float3 rayOrg;
            float3& rayDirRCP = (((float3*)sharedMemNew)[threadId1D32()]);

            float rayT  = dcRayGenerator(rayOrg, rayDirRCP, myRayIndex, dcNumRays);
            rayDirRCP.x = 1.f / rayDirRCP.x;
            rayDirRCP.y = 1.f / rayDirRCP.y;
            rayDirRCP.z = 1.f / rayDirRCP.z;

            uint  bestHit = aStorage.numPrimitives;
            bool traversalFlag = (rayT >= 0.f) && myRayIndex < dcNumRays;
            //////////////////////////////////////////////////////////////////////////

            tTraverser<tPrimitive, tIntersector> traverse;
            traverse(aStorage, dcGrid, rayOrg, rayDirRCP, rayT, bestHit, traversalFlag, sharedMemNew);

            //////////////////////////////////////////////////////////////////////////
            //Output
            float3 rayDir;
            rayDir.x = 1.f / rayDirRCP.x;
            rayDir.y = 1.f / rayDirRCP.y;
            rayDir.z = 1.f / rayDirRCP.z;

            if(rayT < FLT_MAX)
                bestHit = dcGrid.primitives[bestHit];
            //TODO: remove this and connect ot the rest of the code
            aRayBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumRays);
            //////////////////////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////////////////////////
            //Initialization

            //1st flag: has valid intersection at begin of path
            //2nd flag: still has valid intersection
            //3rd flag: intersected primitive is transparent
            //4th flag: true = reflect, false = refract
            //data: path depth
            flag4 myFlags;
            myFlags.setFlag1(myRayIndex < dcNumRays);
            if (myFlags.getFlag1())
            {
                rayT = aRayBuffer.loadDistance(myRayIndex, dcNumRays);
                myFlags.setFlag2(true);
                myFlags.setFlag2(rayT < FLT_MAX);
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
                (myRayIndex + aImageId * dcNumRays) * 
                (myRayIndex + aImageId * dcNumRays) *
                (myRayIndex + aImageId * dcNumRays) );

            float3 attenuation = rep(1.f);

            float indexOfRefraction;

            if (myFlags.getFlag2())
            {
                rayOrg = aRayBuffer.loadOrigin(myRayIndex, dcNumRays);
                bestHit = aRayBuffer.loadBestHit(myRayIndex, dcNumRays);
                rayDir = aRayBuffer.loadDirection(myRayIndex, dcNumRays);

                myFlags.setFlag3(dcMaterialStorage[bestHit].isTransparent(indexOfRefraction));

                if (!myFlags.getFlag3())
                {
                    float3 diffReflectance = dcMaterialStorage[bestHit].getDiffuseReflectance();
                    float albedoD = (diffReflectance.x * 0.222f +
                        diffReflectance.y * 0.7067f +
                        diffReflectance.z * 0.0713f) * M_PI;
                    float3 specReflectance = dcMaterialStorage[bestHit].getSpecularReflectance();
                    float specExponent = dcMaterialStorage[bestHit].getSpecularExponent();
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

                        if (sinNormalRefrDirSQR <= 1.f && genRand() < fresnelCoeff)
                        {
                            //TODO: use Halton numbers here
                            myFlags.setFlag4(true);
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
                    else
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
                traverse(aStorage, dcGrid, rayOrg, rayDirRCP, rayT, bestHit, traversalFlag, sharedMemNew);
                //////////////////////////////////////////////////////////////////////////

                if(myFlags.getFlag2())
                {

                    rayDir.x = 1.f / rayDir.x;
                    rayDir.y = 1.f / rayDir.y;
                    rayDir.z = 1.f / rayDir.z;


                    if(rayT >= FLT_MAX || myFlags.getData() >= RFRACTION_PATH_MAX_DEPTH)
                    {
                        //outside scene
                        aRayBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumRays);
                        myFlags.setFlag2To0();

                        rayT = -1.f;
                    }
                    else
                    {
                        myFlags.setFlag3(dcMaterialStorage[bestHit].isTransparent(indexOfRefraction));
                        if (!myFlags.getFlag3())
                        {
                            float3 diffReflectance = dcMaterialStorage[bestHit].getDiffuseReflectance();
                            float albedoD = (diffReflectance.x * 0.222f +
                                diffReflectance.y * 0.7067f +
                                diffReflectance.z * 0.0713f) * M_PI;
                            float3 specReflectance = dcMaterialStorage[bestHit].getSpecularReflectance();
                            float specExponent = dcMaterialStorage[bestHit].getSpecularExponent();
                            float albedoS = (specReflectance.x * 0.222f +
                                specReflectance.y * 0.7067f +
                                specReflectance.z * 0.0713f) * 2.f * M_PI / (specExponent + 2.f);

                            albedoS = (albedoS > 0.f)  ? fminf(1.f - albedoD, 0.25f): 0.f;

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
                                aRayBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumRays);
                                myFlags.setFlag2To0();
                                rayT = -1.f;
                            }
                        }

                        myFlags.setData(myFlags.getData() + 1);
                        rayOrg = rayOrg + rayT * rayDir;
                    }
                }//endif thread is active

                //break; //TODO: remove
            }//end while continue path
            
            float newSampleWeight = 1.f / (float)(aImageId + 1);
            float oldSamplesWeight = 1.f - newSampleWeight;

            dcFrameBuffer.deviceData[myRayIndex] = 
                dcFrameBuffer.deviceData[myRayIndex] * oldSamplesWeight +
                attenuation * newSampleWeight;
        }//end while persistent threads
    }

    template<
        class tPrimitive,
        class tAccelerationStructure,
            template <class, class> class tTraverser,
        class tPrimaryIntersector,
        class tAlternativeIntersector>

    class PathTracer
    {
        int* mGlobalMemoryPtr;
        size_t mGlobalMemorySize;
        cudaEvent_t mTrace, mShade;
    public:
        typedef RandomPrimaryRayGenerator< GaussianPixelSampler, true > t_PrimaryRayGenerator;
        typedef SimpleRayBuffer                                         t_RayBuffer;
        typedef tPrimaryIntersector                                     t_Intersector;
        typedef tAccelerationStructure                                  t_AccelerationStructure;

        t_RayBuffer rayBuffer;

        PathTracer():rayBuffer(t_RayBuffer(NULL)), mGlobalMemorySize(0u)
        {}

        ~PathTracer()
        {
            if(mGlobalMemorySize != 0u)
            {
                MY_CUDA_SAFE_CALL( cudaFree(mGlobalMemoryPtr));
            }

        }

        HOST void traceNextPath(
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

            const uint sharedMemoryTrace = SHARED_MEMORY_TRACE;

            const uint numRays = aFrameBuffer.resX * aFrameBuffer.resY;
            const uint globalMemorySize = sizeof(uint) +    //Persistent threads
                numRays * sizeof(float3) +                   //rayOrg
                numRays * sizeof(float3) +                   //rayDir
                numRays * sizeof(float) +                   //rayT
                numRays * sizeof(uint) +                    //primitive Id
                0u;

            MemoryManager::allocateDeviceArray((void**)&mGlobalMemoryPtr, globalMemorySize, (void**)&mGlobalMemoryPtr, mGlobalMemorySize);

            MY_CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );

            dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
            dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );

            rayBuffer.setMemPtr(mGlobalMemoryPtr + 1);
            //MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGrid", &aAccStruct, sizeof(UniformGrid)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRayGenerator", &aRayGenerator, sizeof(t_PrimaryRayGenerator)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcFrameBuffer", &aFrameBuffer, sizeof(FrameBuffer)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcMaterialStorage", &aMaterialStorage, sizeof(PrimitiveAttributeArray<PhongMaterial>)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcLightSources", &aLightSources, sizeof(AreaLightSourceCollection)) );
            MY_CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );
            
            cudaEventCreate(&mTrace);

            tracePath< tPrimitive, tAccelerationStructure, tTraverser, tPrimaryIntersector >
                <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace>>>
                (aStorage, aNormalStorage, aAccStruct, rayBuffer, aImageId, mGlobalMemoryPtr);

            cudaEventRecord(mTrace, 0);
            cudaEventSynchronize(mTrace);
            MY_CUT_CHECK_ERROR("Tracing path failed!\n");
            cudaEventDestroy(mTrace);

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
            traceNextPath(aStorage, aNormalStorage, aMaterialStorage, aLightSources, aAccStruct, aRayGenerator, aFrameBuffer, aImageId);
        }

    };

#endif // PATHTRACER_H_DD9E04C2_BF7C_4C6E_8560_05836A9EF2FF
