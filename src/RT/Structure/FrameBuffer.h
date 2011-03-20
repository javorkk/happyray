#ifdef _MSC_VER
#pragma once
#endif

#ifndef FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC
#define FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC

#include "CUDAStdAfx.h"
#include "Core/Algebra.hpp"

struct FrameBuffer
{
    uint resX, resY;
    float3* deviceData;

    DEVICE float3& operator[](const uint aId)
    {
        return deviceData[aId];
    }

    DEVICE const float3& operator[](const uint aId) const
    {
        return deviceData[aId];
    }

    HOST void init(uint aResX, uint aResY)
    {
        resX = aResX;
        resY = aResY;
        MY_CUDA_SAFE_CALL( 
            cudaMalloc((void**) &deviceData, aResX * aResY * sizeof(float3)));
    }

    HOST void download(float3* aTarget) const
    {
        MY_CUDA_SAFE_CALL(
            cudaMemcpy(
                aTarget,
                deviceData,
                resX * resY * sizeof(float3),
                cudaMemcpyDeviceToHost) );
    }

    HOST void download(float3* aTarget, const int aResX, const int aResY) const
    {
        MY_CUDA_SAFE_CALL(
            cudaMemcpy(
            aTarget,
            deviceData,
            aResX * aResY * sizeof(float3),
            cudaMemcpyDeviceToHost) );
    }

    HOST void setZero()
    {
        MY_CUDA_SAFE_CALL(
            cudaMemset(
                (void*) deviceData,
                0,
                resX * resY * sizeof(float3)));
    }

    HOST void cleanup()
    {
        MY_CUDA_SAFE_CALL( cudaFree(deviceData) );
    }
};

#endif // FRAMEBUFFER_H_FF28DF39_846E_4B51_A32C_A7FE9C1D20EC
