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

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879
#define ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879

#include "Application/WFObject.hpp"

#define _SET_MEMBER(aName, aType)					                           \
    void set##aName (aType aValue)	                                           \
    {												                           \
    m##aName = aValue;								                           \
    }

#define _GET_MEMBER(aName, aType)					                           \
    aType get##aName () const			                                       \
    {												                           \
    return m##aName ;								                           \
    }


class AnimationManager
{
    std::vector<WFObject> mKeyFrames;
    std::vector<bool> mInterpolatable;
    float mCurrentFrameId; //interpolation coefficient
    float mStepSize;
public:

    AnimationManager() :
        mCurrentFrameId(0.f), mStepSize(0.8f)
    {}

    size_t getNumKeyFrames() const { return mKeyFrames.size(); }

    _GET_MEMBER(CurrentFrameId, float);
    _SET_MEMBER(CurrentFrameId, float);

    _GET_MEMBER(StepSize, float);
    _SET_MEMBER(StepSize, float);

    WFObject* getKeyFrames() { return (WFObject*)mKeyFrames.data(); }

    size_t getFrameId() const
    {
        return static_cast<size_t>(mCurrentFrameId);
    }

    size_t getNextFrameId() const
    {
        size_t nextFrameId = (getFrameId()
            + 1u
            + static_cast<size_t>(mStepSize)) % getNumKeyFrames();

        return mInterpolatable[nextFrameId] && mInterpolatable[getFrameId()] ? nextFrameId : getFrameId();
    }

    WFObject& getFrame(size_t aFrameId)
    {
        return mKeyFrames[aFrameId];
    }

    bool isInterpolatable(size_t aFrameId)
    {
        return mInterpolatable[aFrameId];
    }


    float getInterpolationCoefficient() const
    {
        return mCurrentFrameId -
            static_cast<float>(static_cast<size_t>(mCurrentFrameId));
    }

    void nextFrame()
    {
        mCurrentFrameId += mStepSize;

        if (static_cast<size_t>(mCurrentFrameId) >= getNumKeyFrames())
        {
            mCurrentFrameId -=
                static_cast<float>(getNumKeyFrames());
        }
    }

    void previousFrame()
    {
        mCurrentFrameId -= mStepSize;

        if (mCurrentFrameId < 0.f)
        {
            mCurrentFrameId +=
                static_cast<float>(getNumKeyFrames());
        }
    }

    std::pair<float3, float3> getBounds(size_t aFrameId = 0u);

    //frameFileName is aFileNamePrefix::frameIndex::aFileNameSuffix
    void read(const char* aFileNamePrefix,
        const char* aFileNameSuffix,
        size_t aNumFrames);

    void read(const char* aFileName);

    void loadEmptyFrame()
    {
        WFObject frame;
        mKeyFrames.push_back(frame);
        mInterpolatable.push_back(false);
    }

    void dumpFrame();
};

#undef _GET_MEMBER
#undef _SET_MEMBER

#endif // ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879
