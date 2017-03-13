/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
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
#include "AnimationManager.hpp"


std::pair<float3, float3> AnimationManager::getBounds(size_t aFrameId)
{
    float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    if (getNumKeyFrames() == 0u)
    {
        return std::pair<float3, float3>(minBound, maxBound);
    }

    size_t frameId = aFrameId % getNumKeyFrames();

    const WFObject& frame = mKeyFrames[frameId];

    for (size_t it = 0u; it < frame.getNumVertices(); ++it)
    {
        minBound = min(frame.vertices[it], minBound);
        maxBound = max(frame.vertices[it], maxBound);
    }

    return std::pair<float3, float3>(minBound, maxBound);
}

void AnimationManager::read(const char* aFileNamePrefix,
    const char* aFileNameSuffix,
    size_t aNumFrames)
{
    const std::string fileNamePrefix = aFileNamePrefix;
    const std::string fileNameSuffix = aFileNameSuffix;

    int numDigits = 0;
    int numFrames = static_cast<int>(aNumFrames)-1;

    for (; numFrames > 0; numFrames /= 10, ++numDigits)
    {
    }

    if (aNumFrames > 0u)
    {
        for (size_t frameId = 0; frameId < aNumFrames; ++frameId)
        {
            std::string frameIdStr;
            int currNumDigits = 1;
            int hlpFrameId = static_cast<int>(frameId);
            for (; hlpFrameId > 9; hlpFrameId /= 10, ++currNumDigits)
            {
            }

            for (; currNumDigits < numDigits; ++currNumDigits)
            {
                frameIdStr.append(std::string("0"));
            }

            frameIdStr.append(itoa(static_cast<int>(frameId)));

            const std::string fileName =
                fileNamePrefix +
                frameIdStr +
                fileNameSuffix;

            WFObject currentFrame;
            currentFrame.read(fileName.c_str());
            mKeyFrames.push_back(currentFrame);
            mInterpolatable.push_back(true);
        }
    }
}

void AnimationManager::read(const char * aFileName)
{
    WFObject currentFrame;
    currentFrame.read(aFileName);
    mKeyFrames.push_back(currentFrame);
    mInterpolatable.push_back(false);
}


#include "Application/WFObjWriter.h"

void AnimationManager::dumpFrame()
{
    const size_t frameId1 = getFrameId();
    const size_t frameId2 = getNextFrameId();

    const WFObject& aKeyFrame1 = getFrame(frameId1);
    const WFObject& aKeyFrame2 = getFrame(frameId2);

    const float aCoeff = getInterpolationCoefficient();

    ObjWriter objOut;
    std::string filename("frame_dump_");
    filename.append(itoa((uint)frameId1));
    filename.append("_");
    filename.append(ftoa(aCoeff));
    //filename.append(".obj");
    
	objOut.init(filename.c_str());

    if (!objOut.objFileStream)
    {
        std::cerr << "Could not open file " << filename <<" for writing!\n";
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    //write vertex data
    //////////////////////////////////////////////////////////////////////////
    const size_t numVertices1 = aKeyFrame1.getNumVertices();
    const size_t numVertices2 = aKeyFrame2.getNumVertices();
    const size_t numVertices = numVertices2;

    size_t it = 0;
    for (; it < std::min(numVertices1, numVertices2); ++it)
    {
        float x,y,z;
        x = aKeyFrame1.getVertex(it).x * (1.f - aCoeff) + aKeyFrame2.getVertex(it).x * aCoeff;
        y = aKeyFrame1.getVertex(it).y * (1.f - aCoeff) + aKeyFrame2.getVertex(it).y * aCoeff;
        z = aKeyFrame1.getVertex(it).z * (1.f - aCoeff) + aKeyFrame2.getVertex(it).z * aCoeff;
        
        objOut.writeVertex(x,y,z);
    }

    for (; it < numVertices2; ++it)
    {
        float x, y, z;
        x = aKeyFrame2.getVertex(it).x;
        y = aKeyFrame2.getVertex(it).y;
        z = aKeyFrame2.getVertex(it).z;

        objOut.writeVertex(x, y, z);
    }


    //////////////////////////////////////////////////////////////////////////
    //write normal data
    //////////////////////////////////////////////////////////////////////////
    const size_t numNormals1 = aKeyFrame1.getNumNormals();
    const size_t numNormals2 = aKeyFrame2.getNumNormals();
    const size_t numNormals = numNormals2;

    it = 0;
    for (; it < std::min(numNormals1, numNormals2); ++it)
    {
        float x, y, z;
        x = aKeyFrame1.getNormal(it).x * (1.f - aCoeff) + aKeyFrame2.getNormal(it).x * aCoeff;
        y = aKeyFrame1.getNormal(it).y * (1.f - aCoeff) + aKeyFrame2.getNormal(it).y * aCoeff;
        z = aKeyFrame1.getNormal(it).z * (1.f - aCoeff) + aKeyFrame2.getNormal(it).z * aCoeff;
        objOut.writeVertexNormal(x, y, z);
    }

    for (; it < numNormals2; ++it)
    {
        float x, y, z;
        x = aKeyFrame2.getNormal(it).x;
        y = aKeyFrame2.getNormal(it).y;
        z = aKeyFrame2.getNormal(it).z;
        objOut.writeVertexNormal(x, y, z);
    }

    //////////////////////////////////////////////////////////////////////////
    //write indices
    //////////////////////////////////////////////////////////////////////////
    const size_t numIndices1 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices2 = aKeyFrame1.getNumFaces() * 3;
    const size_t numIndices = numIndices2;

    it = 0;
    for (; it < std::min(numIndices1, numIndices2); it += 3)
    {
        uint id0 = aKeyFrame1.getVertexIndex(it + 0);
        uint id1 = aKeyFrame1.getVertexIndex(it + 1);
        uint id2 = aKeyFrame1.getVertexIndex(it + 2);

        objOut.writeTriangleIndices(
            id0,
            id1,
            id2);
    }

    for (; it < numIndices2; it += 3)
    {
        uint id0 = aKeyFrame2.getVertexIndex(it + 0);
        uint id1 = aKeyFrame2.getVertexIndex(it + 1);
        uint id2 = aKeyFrame2.getVertexIndex(it + 2);

        objOut.writeTriangleIndices(
            id0,
            id1,
            id2);
    }

	objOut.cleanup();
}