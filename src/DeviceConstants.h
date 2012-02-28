#ifdef _MSC_VER
#pragma once
#endif

#ifndef DEVICECONSTANTS_H_INCLUDED_7E64F0AA_290C_43CC_B00F_296B9DF4FB5F
#define DEVICECONSTANTS_H_INCLUDED_7E64F0AA_290C_43CC_B00F_296B9DF4FB5F


#include "CUDAStdAfx.h"
#include "RT/Structure/PrimitiveArray.h"
#include "RT/Primitive/Material.hpp"

DEVICE_NO_INLINE CONSTANT PrimitiveAttributeArray<PhongMaterial>                    dcMaterialStorage;
//DEVICE_NO_INLINE CONSTANT AreaLightSourceCollection                                 dcLightSources;
DEVICE_NO_INLINE CONSTANT uint                                                      dcNumPixels;

DEVICE_NO_INLINE CONSTANT float dcPrimesRCP[] = {0.5f, 0.333333f, 0.2f, 0.142857f,
    0.09090909f, 0.07692307f, 0.058823529f, 0.0526315789f, 0.04347826f,
    0.034482758f, 0.032258064f};

#endif // DEVICECONSTANTS_H_INCLUDED_7E64F0AA_290C_43CC_B00F_296B9DF4FB5F
