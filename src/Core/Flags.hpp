#ifdef _MSC_VER
#pragma once
#endif

#ifndef FLAGS_HPP_CAEB5BE7_C395_4F1F_94B5_861AA011F7AF
#define FLAGS_HPP_CAEB5BE7_C395_4F1F_94B5_861AA011F7AF

#include "CUDAStdAfx.h"

#pragma region Flag Container Structures

struct flag4
{
    //1st, 2nd, 3rd byte: data (uint)
    //4th byte: 4 boolean flags
    int data; //use int not uint or bool!

    DEVICE HOST flag4():data(0x0)
    {}

    enum{
        FLAG1_SHIFT     =   0,
        FLAG2_SHIFT     =   1,
        FLAG1_MASK      =   0x1,
        FLAG3_SHIFT     =   2,
        FLAG2_MASK      =   0x2,
        FLAG4_SHIFT     =   3,
        FLAG3_MASK      =   0x4,
        DATA_SHIFT      =   4,
        FLAG4_MASK      =   0x8,
        DATA_MASK       =   0xF,
        DATA_MASKNEG    =   0xFFFFFFF0,
        FLAG4_MASKNEG   =   0xFFFFFFF7,
        FLAG3_MASKNEG   =   0xFFFFFFFB,
        FLAG2_MASKNEG   =   0xFFFFFFFD,
        FLAG1_MASKNEG   =   0xFFFFFFFE,
    };

#define SET_FLAG(aFlagId)                                                      \
    DEVICE HOST void setFlag##aFlagId(bool aVal)                               \
    { data = aVal ?                                                            \
    (data | FLAG##aFlagId##_MASK) : (data & FLAG##aFlagId##_MASKNEG); }

#define GET_FLAG(aFlagId)                                                      \
    DEVICE HOST bool getFlag##aFlagId () const                                 \
    { return (data & FLAG##aFlagId##_MASK) != 0x0; }

#define SET_FLAG_0(aFlagId)                                                    \
    DEVICE HOST void setFlag##aFlagId##To0 () {data &= FLAG##aFlagId##_MASKNEG;}

#define XOR_FLAGS(aFlag1, aFlag2)                                              \
    DEVICE HOST bool xorFlags##aFlag1##aFlag2 () const                         \
    {                                                                          \
        return (((data >> (FLAG##aFlag2##_SHIFT - FLAG##aFlag1##_SHIFT) & FLAG##aFlag1##_MASK) ^ (data))\
        & FLAG##aFlag1##_MASK) != 0x0;                                         \
    }


    SET_FLAG(1)
    SET_FLAG(2)
    SET_FLAG(3)
    SET_FLAG(4)

    GET_FLAG(1)
    GET_FLAG(2)
    GET_FLAG(3)
    GET_FLAG(4)

    SET_FLAG_0(1)
    SET_FLAG_0(2)
    SET_FLAG_0(3)
    SET_FLAG_0(4)

    XOR_FLAGS(1,2)
    XOR_FLAGS(1,3)
    XOR_FLAGS(1,4)
    XOR_FLAGS(2,3)
    XOR_FLAGS(2,4)
    XOR_FLAGS(3,4)

#undef SET_FLAG
#undef GET_FLAG
#undef SET_FLAG_0
#undef XOR_FLAGS


    DEVICE HOST bool anyFlag() const { return (data & DATA_MASK) != 0x0; }
    DEVICE HOST bool noFlag() const { return (data & DATA_MASK) == 0x0; }

    DEVICE HOST void setFlags12(bool aVal)
    {
        data = aVal ? data | FLAG1_MASK | FLAG2_MASK : 
            (data & FLAG1_MASKNEG) & FLAG2_MASKNEG;
    }

    DEVICE HOST void setData(const int aData){ data |= aData << DATA_SHIFT; }
    DEVICE HOST int getData() const { return data >> DATA_SHIFT; }
};

#pragma endregion // Flag Container Structures

#endif // FLAGS_HPP_CAEB5BE7_C395_4F1F_94B5_861AA011F7AF
