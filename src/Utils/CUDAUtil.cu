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

#include "CUDAUtil.h"
#include <cuda_runtime_api.h>

//static member definition
cudastd::logger cudastd::logger::out;
cudastd::FloatPrecision cudastd::FloatPrecision::manip_precision;

void cudastd::getBestCUDADevice(int argc, char* argv[])
{

#ifndef __DEVICE_EMULATION__

    MY_CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );

    int deviceCount;  
    MY_CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cudastd::logger::out << "Error: no device with CUDA support found.\n";
        exit(1);
    }

    int bestDevice = 0;
    cudaDeviceProp deviceProp;
    cudaDeviceProp bestDeviceProp;
    MY_CUDA_SAFE_CALL( cudaGetDeviceProperties( &bestDeviceProp, 0) );

    for(int dev = 1; dev < deviceCount; ++dev)
    {
        MY_CUDA_SAFE_CALL( cudaGetDeviceProperties( &deviceProp, dev) );
        
        bool majorCapability = deviceProp.major >= bestDeviceProp.major;
        bool minorCapability = deviceProp.minor >= bestDeviceProp.minor;
        bool processorCount  =
            deviceProp.multiProcessorCount >=
            bestDeviceProp.multiProcessorCount;

        bool betterCapability =
            deviceProp.major > bestDeviceProp.major ||
            deviceProp.major == bestDeviceProp.major &&
            deviceProp.minor > bestDeviceProp.minor;

        bool betterAtSomething =
            deviceProp.multiProcessorCount > bestDeviceProp.multiProcessorCount 
            || deviceProp.clockRate > bestDeviceProp.clockRate;

        if (betterCapability ||
            majorCapability && minorCapability && processorCount &&
            betterAtSomething)
        {
            bestDevice = dev;
            MY_CUDA_SAFE_CALL( cudaGetDeviceProperties( &bestDeviceProp, dev) );
        }
    }

    cudastd::logger::out << "Will use device " << bestDevice 
        << " with capability " << bestDeviceProp.major << "." 
        << bestDeviceProp.minor << " (" << bestDeviceProp.name 
        << ")\n";

#if SHINOBI__CUDA_ARCH__ >= 200

    if (bestDeviceProp.major < 2)
    {
        cudastd::logger::out << 
            "Warning: code is compiled for devices with capability 2.0 or higher!\n";
    }

    if (bestDeviceProp.canMapHostMemory)
    {
        MY_CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
    }
    else
    {
        cudastd::logger::out << "Warning: this device does not support mapped memory!\n";
    }

#elif SHINOBI__CUDA_ARCH__ >= 120

    if (bestDeviceProp.major <= 1 && bestDeviceProp.minor < 2)
    {
        cudastd::logger::out << 
            "Warning: code is compiled for devices with capability 1.2 or higher!\n";
    }

    if (bestDeviceProp.canMapHostMemory)
    {
        MY_CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
    }
    else
    {
        cudastd::logger::out << "Warning: this device does not support mapped memory!\n";
    }

#elif SHINOBI__CUDA_ARCH__ >= 110

    if (bestDeviceProp.major <= 1 && bestDeviceProp.minor < 1)
    {
        cudastd::logger::out << 
            "Warning: code is compiled for devices with capability 1.1 or higher!\n";
    }

#endif // __CUDA_ARCH__ >= 120 

    MY_CUDA_SAFE_CALL(cudaSetDevice(bestDevice));

#else

    cudastd::logger::out << "Device emulation, no device is selected.\n";

#endif //not defined __DEVICE_EMULATION__
}// getBestCUDADevice()
