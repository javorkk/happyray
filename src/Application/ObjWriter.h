#ifdef _MSC_VER
#pragma once
#endif

#ifndef OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D
#define OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D

#include <fstream>

class ObjWriter
{
public:
    void writeVertex(std::ofstream& aStream, float aX, float aY, float aZ) const;
    void writeVertexNormal(std::ofstream& aStream, float aX, float aY, float aZ) const;
    void writeObjectHeader(std::ofstream& aStream, int aObjId) const;
    void writeTriangleIndices(std::ofstream& aStream, int aA, int aB, int aC) const;
    void writeInstance(std::ofstream& aStream,
        float aT00, float aT01, float aT02, //transformation matrix column 0 (rotation)
        float aT10, float aT11, float aT12, //transformation matrix column 1 (rotation)
        float aT20, float aT21, float aT22, //transformation matrix column 2 (rotation)
        float aT30, float aT31, float aT32, //transformation matrix column 3 (translation)
        int aObjectId,
        float aMinX, float aMinY, float aMinZ, //min bound
        float aMaxX, float aMaxY, float aMaxZ //max bound
        ) const;

};

#endif // OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D
