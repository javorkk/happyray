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

#include "StdAfx.hpp"
#include "Application/SceneConfiguration.hpp"
#include "Application/LightSourceLoader.hpp"
#include "Application/SceneLoader.hpp"

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

/////////////////////////////////////////
//already included in SceneLoader.hpp
#include "Application/CameraManager.hpp"
#include "Application/WFObject.hpp"
#include "Application/AnimationManager.hpp"
#include "RT/Primitive/LightSource.hpp"
/////////////////////////////////////////


void SceneLoader::insertLightSourceGeometry(const AreaLightSource& aLightSource, WFObject& oScene)
{
    bool sceneHasEmitters = false;
    for(WFObject::t_MaterialIteratorNonConst it = oScene.materialsBegin(); it != oScene.materialsEnd(); ++it)
    {
        if (it->emission.x  + it->emission.y + it->emission.z > 0.f)
        {
            //sceneHasEmitters = true;
            it->emission.x = 0.f;
            it->emission.y = 0.f;
            it->emission.z = 0.f;
        }
    }

    if (sceneHasEmitters)
    {
        return;
    }

    WFObject::Face face1(&oScene);
    WFObject::Face face2(&oScene);


    WFObject::Material material("lightsource");
    material.emission = aLightSource.intensity;

    float3 pt1, pt2, pt3, pt4;
    pt1 = aLightSource.position;
    pt2 = pt1 + aLightSource.edge1;
    pt3 = pt1 + aLightSource.edge2;
    pt4 = pt2 + aLightSource.edge2;

    face1.material = oScene.insertMaterial(material);
    face2.material = face1.material;

    face1.norm1 = face1.norm2 = face1.norm3 = face2.norm1 = face2.norm2 =
        face2.norm3 = oScene.insertNormal(aLightSource.normal);

    face1.tex1 = face1.tex2 = face1.tex3 = face2.tex1 = face2.tex2 =
        face2.tex3 = 0;

    face1.vert1 = oScene.insertVertex(pt1);
    face1.vert2 = oScene.insertVertex(pt2);
    face1.vert3 = oScene.insertVertex(pt4);

    face2.vert1 = face1.vert1;
    face2.vert2 = face1.vert3;
    face2.vert3 = oScene.insertVertex(pt3);

    oScene.insertFace(face1);
    oScene.insertFace(face2);
}

void SceneLoader::createLightSources( AreaLightSourceCollection&     oLightSources, const WFObject& aScene)
{

    float3 emission = rep(0.f);
    size_t material = aScene.getNumMaterials();
    size_t vtxIds[4];
    vtxIds[0] = vtxIds[1] = vtxIds[2] = vtxIds[3] = aScene.getNumVertices();

    for (WFObject::t_FaceIterator faceIt = aScene.facesBegin(); faceIt != aScene.facesEnd(); ++faceIt)
    {
        //is this a light source
        WFObject::t_MaterialIterator materialIt = aScene.materialsBegin() + faceIt->material;
        if(materialIt->emission.x + materialIt->emission.y + materialIt->emission.z > 0 
            && vtxIds[0] == aScene.getNumVertices())
        {
            //this is an emitting triangle, put it in the array
            vtxIds[0] = faceIt->vert1;
            vtxIds[1] = faceIt->vert2;
            vtxIds[2] = faceIt->vert3;
            material = faceIt->material;
            emission = materialIt->emission;
        }
        else if (faceIt->material == material && vtxIds[3] == aScene.getNumVertices())
        {
            uint numMatches = 0u;
            uint commonVtxIdFirstFace[3];
            uint commonVtxIdThisFace[3];
            commonVtxIdFirstFace[0] = commonVtxIdFirstFace[1] = commonVtxIdThisFace[0] = commonVtxIdThisFace[1] = 4u;
            for(uint i = 0; i < 3; ++i)
            {
                if (faceIt->vert1 == vtxIds[i])
                {
                    commonVtxIdThisFace[numMatches] = 0u;
                    commonVtxIdFirstFace[numMatches] = i;
                    ++numMatches;

                }
                else if( faceIt->vert2 == vtxIds[i])
                {
                    commonVtxIdThisFace[numMatches] = 1u;
                    commonVtxIdFirstFace[numMatches] = i;
                    ++numMatches;

                }
                else if(faceIt->vert3 == vtxIds[i])
                {
                    commonVtxIdThisFace[numMatches] = 2u;
                    commonVtxIdFirstFace[numMatches] = i;
                    ++numMatches;
                }
            }

            if (numMatches == 2)
            {
                uint nonMatchedId = 3u - commonVtxIdFirstFace[0] - commonVtxIdFirstFace[1];
                //add the fourth vertex
                const uint lastVtxId = 3u - commonVtxIdThisFace[0] - commonVtxIdThisFace[1];
                switch(lastVtxId)
                {
                case 0:
                    vtxIds[3] = faceIt->vert1;
                    break;
                case 1:
                    vtxIds[3] = faceIt->vert2;
                    break;
                case 2:
                    vtxIds[3] = faceIt->vert3;
                    break;
                default:
                    cudastd::logger::out << "Could not find valid id of the 4th light-source vertex. Skipping this emissive pair\n";
                    vtxIds[0] = vtxIds[1] = vtxIds[2] = vtxIds[3] = aScene.getNumVertices();
                    material = aScene.getNumMaterials();
                    emission = rep(0.f);
                    continue;
                };
                
                const float3 v1 = aScene.getVertex(vtxIds[nonMatchedId]);
                const float3 v2 = aScene.getVertex(vtxIds[(nonMatchedId + 2) % 3]);
                const float3 v3 = aScene.getVertex(vtxIds[3]);
                const float3 v4 = aScene.getVertex(vtxIds[(nonMatchedId + 1) % 3]);
                const float3 normal = aScene.getNormal(faceIt->norm1);
                AreaLightSource ls;
                ls.create(v1, v2, v3, v4, emission, normal);
                oLightSources.upload(ls.getArea()*len(ls.intensity), ls);                
                //re-initialize temporary variables
                vtxIds[0] = vtxIds[1] = vtxIds[2] = vtxIds[3] = aScene.getNumVertices();
                material = aScene.getNumMaterials();
                emission = rep(0.f);
            }
        }
    }


}

bool SceneLoader::loadScene(
                const char*                 CONFIGURATION,
                AnimationManager&           oAnimation,
                CameraManager&              oView,
                AreaLightSourceCollection&  oLightSources)
{
    cudastd::logger::out << "Reading configuration file...\n";

    SceneConfiguration sceneConfig = loadSceneConfiguration(CONFIGURATION);

    bool retval = true;

    bool loadAnimation = sceneConfig.hasFrameFileNamePrefix &&
        sceneConfig.hasFrameFileNameSuffix &&
        sceneConfig.numFrames != 0;

    if (loadAnimation)
    {
        cudastd::logger::out << "Loading animation...\n";
        oAnimation.read(sceneConfig.frameFileNamePrefix,
            sceneConfig.frameFileNameSuffix, sceneConfig.numFrames);
        oAnimation.setStepSize(sceneConfig.frameStepSize);

        cudastd::logger::out << "Number of primitives: " <<
            oAnimation.getFrame(0).getNumFaces() << "\n";

    }
    else if (sceneConfig.hasObjFileName)
    {
        cudastd::logger::out << "Loading scene...\n";
        oAnimation.allocateFrames(1);
        oAnimation.getFrame(0).read(sceneConfig.objFileName);

        cudastd::logger::out << "Number of primitives: " << oAnimation.getFrame(0).getNumFaces() << "\n";

    }
    else
    {
        cudastd::logger::out << "No scene specified in file "
            << CONFIGURATION << "\n";

        retval = false;
    }

    if (sceneConfig.hasCameraFileName)
    {
        cudastd::logger::out << "Loading camera configuration...\n";
        oView.read(sceneConfig.cameraFileName);
    }
    else
    {
        cudastd::logger::out << "No camera configuration specified in file "
            << CONFIGURATION << "\n";
    }

    if(loadAnimation)
    {
        if(sceneConfig.hasLightsFileName)
        {
            cudastd::logger::out << "Loading area lights...\n";
            LightSourceLoader loader;
            std::vector<AreaLightSource> lights = loader.loadFromFile(sceneConfig.lightsFileName);
            for(size_t ls = 0; ls < lights.size(); ++ls)
            {
                oLightSources.upload(lights[ls].getArea()*len(lights[ls].intensity), lights[ls]);
                for (size_t it = 0; it < oAnimation.getNumKeyFrames(); ++it)
                {
                    insertLightSourceGeometry(lights[ls], oAnimation.getFrame(it));
                }
            }
        }
        else
        {   
            createLightSources(oLightSources, oAnimation.getFrame(0));
            cudastd::logger::out << "Found " << oLightSources.size() <<" area lights.\n";
        }

        oLightSources.normalizeALSIntensities();
    }
    else if(retval)
    {
        if(sceneConfig.hasLightsFileName)
        {
            cudastd::logger::out << "Loading area lights...\n";
            LightSourceLoader loader;
            std::vector<AreaLightSource> lights = loader.loadFromFile(sceneConfig.lightsFileName);
            for(size_t ls = 0; ls < lights.size(); ++ls)
            {
                oLightSources.upload(lights[ls].getArea()*len(lights[ls].intensity), lights[ls]);
                insertLightSourceGeometry(lights[ls], oAnimation.getFrame(0));
            }
        }
        else
        {
            createLightSources(oLightSources, oAnimation.getFrame(0));
            cudastd::logger::out << "Found " << oLightSources.size() <<" area lights.\n";
        }
        
        oLightSources.normalizeALSIntensities();
    }


    if(retval == false)
    {
        oAnimation.allocateFrames(1);
        loadDefaultScene(oAnimation.getFrame(0), oView);
    }

    return retval;
}


void SceneLoader::loadDefaultScene(WFObject& oScene, CameraManager& oView)
{
    //////////////////////////////////////////////////////////////////////////
    //Make a AABB at the origin with sides 1
    float3 v000 = rep(0.f);
    float3 v100 = v000; v100.x = 1.f;
    float3 v010 = v000; v010.y = 1.f;
    float3 v110 = v010; v110.x = 1.f;
    float3 v011 = v010; v011.z = 1.f;
    float3 v111 = v011; v111.x = 1.f;
    float3 v001 = v000; v001.z = 1.f;
    float3 v101 = v001; v101.x = 1.f;

    for (int i = 0; i < 1000; ++i)
    {
        //base
        oScene.insertVertex(v000);
        oScene.insertVertex(v100);
        oScene.insertVertex(v010);
        oScene.insertVertex(v110);
        //back side
        oScene.insertVertex(v011);
        oScene.insertVertex(v111);
        //top
        oScene.insertVertex(v001);
        oScene.insertVertex(v101);
        //front side
        oScene.insertVertex(v000);
        oScene.insertVertex(v100);
        //repeat base (1 triangle)
        oScene.insertVertex(v110);
        //right side
        oScene.insertVertex(v101);
        oScene.insertVertex(v111);
        //repeat top
        oScene.insertVertex(v001);
        oScene.insertVertex(v011);
        //left side
        oScene.insertVertex(v000);
        oScene.insertVertex(v010);
    }

    //base
    oScene.insertNormal(v001);
    oScene.insertNormal(v001);
    //back side
    oScene.insertNormal(-v010);
    oScene.insertNormal(-v010);
    //top
    oScene.insertNormal(-v001);
    oScene.insertNormal(-v001);
    //front side
    oScene.insertNormal(v010);
    oScene.insertNormal(v010);
    //repeat base
    oScene.insertNormal(v001);
    //right side
    oScene.insertNormal(-v100);
    oScene.insertNormal(-v100);
    //repeat top
    oScene.insertNormal(-v001);
    oScene.insertNormal(-v001);
    //left side
    oScene.insertNormal(v100);
    oScene.insertNormal(v100);

    WFObject::Material mat;
    mat.name = "Default";
    mat.diffuseCoeff = rep(.9774f);
    mat.specularCoeff = rep(0.f);
    mat.specularExp = 1.f;
    mat.ambientCoeff = rep(0.2f);
    mat.emission = rep(0.f);

    const size_t matId = oScene.insertMaterial(mat);


    for(int faceIt = 0; faceIt < 15; ++faceIt)
    {
        WFObject::Face face(&oScene);
        int vtxId =faceIt;

        face.material = matId;
        face.vert1 = vtxId;
        face.vert2 = vtxId + 1;
        face.vert3 = vtxId + 2;
        face.norm1 = face.norm2 = face.norm3 = faceIt;

        oScene.insertFace(face);
    }


    //////////////////////////////////////////////////////////////////////////
    float3 orientation, up;
    orientation.x = 0.62797f;
    orientation.y = 0.62048f;
    orientation.z = 0.46974f;
    up.x = -0.334145f;
    up.y = -0.330159f;
    up.z =  0.882803f;
    oView.setPosition(rep(0.1f));
    oView.setOrientation(orientation);
    oView.setUp(up);
    oView.setFOV(60);
    oView.setResX(720);
    oView.setResY(480);
    oView.setRight(~(oView.getOrientation() % oView.getUp()));

}
