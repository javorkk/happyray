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
#include "LightSourceLoader.hpp"

using namespace cudastd;

std::vector<AreaLightSource> LightSourceLoader::loadFromFile(const char* aFileName)
{
    std::vector<AreaLightSource> lights;

    std::ifstream input;
    input.open(aFileName);

    if(!input)
        std::cerr << "Could not open file " << aFileName << " for reading.\n"
        << __FILE__ << __LINE__ << std::endl;

    std::string line, buff;
 
    AreaLightSource current;
    current.position.x = current.position.y = current.position.z = 0.f;
    current.normal.x = current.normal.y = current.normal.z = 0.f;
    current.intensity.x = current.intensity.y = current.intensity.z = 0.f;
    current.edge1.x = current.edge1.y = current.edge1.z = 0.f;
    current.edge2.x = current.edge2.y = current.edge2.z = 0.f;

    bool hasPosition = false;
    bool hasNormal = false;
    bool hasIntensity = false;
    bool hasEdge1 = false;
    bool hasEdge2 = false;

    while ( !input.eof() )
    {
        std::getline(input, line);
        line = cutComments(line, "#");

        std::stringstream ss(line);
        ss >> buff;

        if (buff == "position")
        {
            ss >> current.position.x >> current.position.y >> current.position.z;
            hasPosition = true;
        } 
        else if (buff == "normal")
        {
            ss >> current.normal.x >> current.normal.y >> current.normal.z;
            hasNormal = true;
        } 
        else if (buff == "intensity")
        {
            ss >> current.intensity.x >> current.intensity.y >> current.intensity.z;
            hasIntensity = true;
        }
        else if (buff == "edge1")
        {
            ss >> current.edge1.x >> current.edge1.y >> current.edge1.z;
            hasEdge1 = true;
        }
        else if (buff == "edge2")
        {
            ss >> current.edge2.x >> current.edge2.y >> current.edge2.z;
            hasEdge2 = true;
        }

        if(hasPosition && hasIntensity && hasEdge1 && hasEdge2)
        {
            if (!hasNormal)
            {
                current.normal = ~(current.edge1 % current.edge2);
            }

            lights.push_back(current);

            hasPosition = false;
            hasNormal = false;
            hasIntensity = false;
            hasEdge1 = false;
            hasEdge2 = false;
            current.position.x = current.position.y = current.position.z = 0.f;
            current.normal.x = current.normal.y = current.normal.z = 0.f;
            current.intensity.x = current.intensity.y = current.intensity.z = 0.f;
            current.edge1.x = current.edge1.y = current.edge1.z = 0.f;
            current.edge2.x = current.edge2.y = current.edge2.z = 0.f;
        }
    }

    input.close();

    return lights;
}
