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

// Main.cpp : Defines the entry point for the console application. 
//

#include "StdAfx.hpp"
#include "Application/SDLGLApplication.hpp"

#include "Test/TestAlgebra.h"
#include "RT/Structure/TwoLevelGridHierarchy.h"

struct CompactUGrid : public Primitive<2>
{
    int res[3];
    uint cells;
};

#ifdef _WIN32
int wmain (int argc, char* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    //TestQuaternions test;
    //if (test.run() != 0)
    //    return 1;
    std::cerr << "Instance size: " << sizeof(GeometryInstance) << "\n";
    std::cerr << "Instance matrix size: " << sizeof(GeometryInstanceQuaternion) << "\n";
    std::cerr << "Instance matrix size: " << sizeof(GeometryInstanceMatrix) << "\n";
    std::cerr << "quaternion3f size: " << sizeof(quaternion3f) << "\n";
    std::cerr << "foat3 size: " << sizeof(float3) << "\n";
    std::cerr << "uint size: " << sizeof(uint) << "\n";
    std::cerr << "UGridPtr size: " << sizeof(UniformGrid*) << "\n";
    std::cerr << "UGrid size: " << sizeof(UniformGrid) << "\n";
    std::cerr << "CompactUGrid size: " << sizeof(CompactUGrid) << "\n";


    SDLGLApplication app;
    app.init(argc, argv);
    app.initVideo();

    while(!app.dead())
    {
        app.displayFrame();
        app.fetchEvents();
        //SDL_Delay(100);
    }

	return 0;
}

