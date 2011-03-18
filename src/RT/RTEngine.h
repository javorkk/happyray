#ifdef _MSC_VER
#pragma once
#endif

#ifndef RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
#define RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5

#include "CUDAStdAfx.h"
#include "Application/WFObject.hpp"

class StaticRTEngine
{
public:

    static void init(const WFObject& aScene);
    static void cleanup();
};

#endif // RTENGINE_H_6F33553F_C8E8_4EB9_B0D1_8AA967B3CEC5
