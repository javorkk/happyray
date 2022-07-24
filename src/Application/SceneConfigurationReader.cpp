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
#include "SceneConfiguration.hpp"

#define COMMENTTOKEN "#"

SceneConfiguration loadSceneConfiguration(const char* aFileName)
{

    std::ifstream input;
    input.open(aFileName);

    if (!input)
        std::cerr << "Could not open file " << aFileName << " for reading.\n"
        << __FILE__ << __LINE__ << std::endl;

    std::string line, buff;
    int lineNumber = 0;

    SceneConfiguration configurationData;

    configurationData.objFileNames.clear();
    configurationData.cameraFileName = "";
    configurationData.cameraPathFileName = "";
    configurationData.lightsFileName = "";
    configurationData.frameFileNamePrefix = "";
    configurationData.frameFileNameSuffix = "";
    configurationData.numFrames = 0;

    configurationData.instancesFileName = "";


    while (!input.eof())
    {
        ++lineNumber;
        buff = "";
        std::getline(input, line);
        if(line.find_first_not_of(" \r\n\t") == std::string::npos)
        {
            continue;
        }

        std::replace(line.begin(), line.end(), '\r', '\0');
        line = cutComments(line, COMMENTTOKEN);

        if (line.size() < 1)
        {
            continue;
        }

        //cut out token
        buff = line.substr(
            line.find_first_not_of(" \t", 0), line.find_first_of("=", 0));
        //cut out whitespace
        buff = buff.substr(0, buff.find_last_not_of(" \t") + 1);


        if (buff == "obj" || buff == "scene")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.objFileNames.push_back(line);
        }
        else if (buff == "camera" || buff == "view" || buff == "cam")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.cameraFileName = line;
        }
        else if (buff == "camera path" || buff == "camerapath"
            || buff == "camera_path")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.cameraPathFileName = line;
        }
        else if (buff == "frame prefix" || buff == "frameprefix"
            || buff == "framename")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.frameFileNamePrefix = line;
        }
        else if (buff == "frame suffix" || buff == "framesuffix")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.frameFileNameSuffix = line;
        }
        else if (buff == "frame number" || buff == "number"
            || buff == "framenumber" || buff == "number of frames")
        {
            std::stringstream ss(line);

            while (buff != "=")
            {
                ss >> buff;
            }

            ss >> configurationData.numFrames;
        }
        else if (buff == "frame step" || buff == "framestep" ||
            buff == "step size")
        {
            std::stringstream ss(line);

            while (buff != "=")
            {
                ss >> buff;
            }
            ss >> configurationData.frameStepSize;

        }
        else if (buff == "light sources" || buff == "light source"
            || buff == "light" || buff == "lights")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.lightsFileName = line;
        }
        else if (buff == "instances")
        {
            const size_t valueStart = line.find_first_of("=", 0) + 1;
            line = line.substr(line.find_first_not_of(" \t", valueStart),
                line.size());
            configurationData.instancesFileName = line;
        }
        else
        {
            //unrecognized token
            std::cerr << "Skipping unrecognized token: " << buff
                << " at line " << lineNumber << " in file "
                << aFileName << "\n";
        }

    }//end while

    input.close();

    return configurationData;
    //   SceneConfiguration retval;
    //   retval.objFileName          = gConfigurationData.objFileNames.back().c_str();
    //   retval.cameraFileName       = gConfigurationData.cameraFileName.c_str();
    //   retval.cameraPathFileName   = gConfigurationData.cameraPathFileName.c_str();
    //   retval.lightsFileName       = gConfigurationData.lightsFileName.c_str();
    //   retval.frameFileNamePrefix  = gConfigurationData.frameFileNamePrefix.c_str();
    //   retval.frameFileNameSuffix  = gConfigurationData.frameFileNameSuffix.c_str();
    //   retval.numFrames            = gConfigurationData.numFrames;
    //   retval.frameStepSize        = gConfigurationData.frameStepSize;

    //retval.hasObjFileName = gConfigurationData.objFileNames.size == 0u;
    //   retval.hasCameraFileName        = gConfigurationData.cameraFileName != "";
    //   retval.hasCameraPathFileName    = gConfigurationData.cameraPathFileName != "";
    //   retval.hasLightsFileName        = gConfigurationData.lightsFileName != "";
    //   retval.hasFrameFileNamePrefix   = gConfigurationData.frameFileNamePrefix != "";
    //   retval.hasFrameFileNameSuffix   = gConfigurationData.frameFileNameSuffix != "";

    //   retval.instancesFileName = gConfigurationData.instancesFileName.c_str();
    //   retval.hasInstancesFileName = gConfigurationData.instancesFileName != "";

    //   return retval;
}


#undef COMMENTTOKEN
