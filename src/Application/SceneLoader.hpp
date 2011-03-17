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

#ifndef SCENELOADER_HPP_4B282D1A_F513_4D82_9A41_3F7E4E6A6BFC
#define SCENELOADER_HPP_4B282D1A_F513_4D82_9A41_3F7E4E6A6BFC

#include "Application/CameraManager.hpp"
#include "Application/WFObject.hpp"
#include "Application/AnimationManager.hpp"
#include "RT/Primitive/LightSource.hpp"

class SceneLoader
{

public:
void insertLightSourceGeometry(const AreaLightSource& aLightSource, WFObject& oScene);

void createLightSource( AreaLightSource& oLightSource, const WFObject& aScene);

bool loadScene( const char*          CONFIGURATION,
                AnimationManager&    oAnimation,
                WFObject&            oScene,
                CameraManager&       oView,
                AreaLightSource&     oLightSource
                );

void loadDefaultScene(WFObject& oScene, CameraManager& oView);

};


#endif // SCENELOADER_H_INCLUDED_4B282D1A_F513_4D82_9A41_3F7E4E6A6BFC
