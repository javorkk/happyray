#ifdef _MSC_VER
#pragma once
#endif

#ifndef MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8
#define MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8

#include "CUDAStdAfx.h"

#define INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY 10.f

class PhongMaterial
{
public:
    float4 diffuseReflectance; //.xyz -> diffuse reflectance .w -> index of refraction
    float4 specularReflectance; //.xyz -> specular reflectance .w -> specular exponent

    DEVICE HOST float3 getDiffuseReflectance() const
    {
        float3 retval;
        retval.x = diffuseReflectance.x;
        retval.y = diffuseReflectance.y;
        retval.z = diffuseReflectance.z;
        return retval;
    }

    DEVICE HOST float3 getSpecularReflectance() const
    {
        float3 retval;
        retval.x = specularReflectance.x;
        retval.y = specularReflectance.y;
        retval.z = specularReflectance.z;
        return retval;
    }

    DEVICE HOST float getSpecularExponent() const
    {
        return specularReflectance.w;
    }

    DEVICE HOST bool isTransparent(float& oIndexOfRefraction) const
    {
        oIndexOfRefraction = diffuseReflectance.w -
            INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY;

        return oIndexOfRefraction > 0.f;
    }

};

#undef INDEX_OF_REFRACTION_OFFSET_FOR_TRANSPARENCY

#endif // MATERIAL_HPP_E14AC36A_D813_4C80_B0E5_E872193D0EC8
