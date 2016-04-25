/****************************************************************************/
/* Copyright (c) 2009, Lukas Marsalek, Javor Kalojanov
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

#ifndef CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA
#define CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA

/*****************************************************************************
* Logger class to roughly simulate I/O Streams with printf. 
* For use in SPU code
* 
* Lukas Marsalek, 01.07.2008
****************************************************************************/

namespace cudastd 
{	
    class NewLine
    {
    public:
        static NewLine manip_endl;
    };	
    class Hex
    {
    public:
        static Hex manip_hex;
    };	

    class InfoSwitch
    {
    public:
        static InfoSwitch manip_infoSwitch;
    };

    class ErrorSwitch
    {
    public:
        static ErrorSwitch manip_errorSwitch;
    };

    class FloatPrecision
    {
    private:

        unsigned int precision;

    public:

        static FloatPrecision manip_precision;

        FloatPrecision& setPrecision(unsigned int prec)
        {
            precision = prec;
            return manip_precision;
        }

        unsigned int getPrecision() const
        {
            return precision;
        }
    };


    class logger
    {

    public:	

        static logger out;		

		logger();

        static NewLine endl()
        {
            return cudastd::NewLine::manip_endl;
        }

        static FloatPrecision floatPrecision(unsigned int precision)
        {
            return cudastd::FloatPrecision::manip_precision.setPrecision(precision);
        }

        static Hex hexFormat()
        {
            return cudastd::Hex::manip_hex;
        }

        void loggingSilence(bool state)
        {
            loggingOff = state;
        }

        logger& operator << (const InfoSwitch&)
        {			
            return out;
        }

        logger& operator << (const ErrorSwitch&)
        {			
            return out;
        }

		logger& operator << (const NewLine&);

		logger& operator << (const FloatPrecision& fp);

		logger& operator << (const Hex&);

		logger& operator << (const char* str);

		logger& operator << (const unsigned long long value);

		logger& operator << (const unsigned int value);

		logger& operator << (const int value);

#ifndef _WIN32

		logger& operator << (const size_t value);

#endif		
		logger& operator << (const long int value);


		logger& operator << (const float value);

    private:

        bool hexOutput;
        bool infoOn;
        bool loggingOff;		
        char fpString[256];

    };
};//namespace cudastd

/*******************************************************************************
*Classes and function to mimic implementations in the <xutility> header
*
*
*
*******************************************************************************/

namespace cudastd 
{	
    template<class tType>  __device__ __host__ inline
        const tType& max(const tType& aLeft, const tType& aRight)
    {
#ifdef __CUDA_ARCH__
        return max(aLeft, aRight);
#else
        return ((aLeft < aRight) ? aRight : aLeft);
#endif
    }

    template<class tType> __device__ __host__ inline
        const tType& min(const tType& aLeft, const tType& aRight)
    {
#ifdef __CUDA_ARCH__
        return min(aLeft, aRight);
#else
        return ((aRight < aLeft) ? aRight : aLeft);
#endif
    }

};//namespace cudastd

/*******************************************************************************
*Classes and function to mimic implementations in cutil.h
*
*
*
*******************************************************************************/

#define MY_CUDA_SAFE_CALL(call) {                                              \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
    cudastd::logger::out << "CUDA error in " << __FILE__ << " line "           \
    << __LINE__ <<" : " << cudaGetErrorString( err) << "\n";                   \
            /*exit(1);*/                                                       \
    } }                                                                        \
    /* End Macro */


#ifdef _DEBUG
#  define MY_CUT_CHECK_ERROR(errorMessage) {                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if( cudaSuccess != err) {                                                  \
        cudastd::logger::out << "Cuda error: "<< errorMessage  <<              \
        " in file " << __FILE__ << " line "<<  __LINE__ << ": "<<              \
                cudaGetErrorString( err) << "\n";                              \
        /*exit(EXIT_FAILURE);*/                                                \
    }                                                                          \
    err = cudaThreadSynchronize();                                             \
    if( cudaSuccess != err) {                                                  \
        cudastd::logger::out << "Cuda error: "<< errorMessage  <<              \
        " in file " << __FILE__ << " line "<<  __LINE__ << ": "<<              \
                cudaGetErrorString( err) << "\n";                              \
                /*exit(EXIT_FAILURE);*/                                        \
    }                                                                          \
    }
#else
#  define MY_CUT_CHECK_ERROR(errorMessage) {                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if( cudaSuccess != err) {                                                  \
        cudastd::logger::out << "Cuda error: "<< errorMessage  <<              \
        " in file " << __FILE__ << " line "<<  __LINE__ << ": "<<              \
                cudaGetErrorString( err) << "\n";                              \
                /*exit(EXIT_FAILURE); */                                       \
    }                                                                          \
    }
#endif


namespace cudastd 
{	
    void getBestCUDADevice(int argc, char* argv[]);
};//namespace cudastd



#endif // CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA
