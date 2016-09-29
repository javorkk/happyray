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
        float aT00, float aT10, float aT20, float aT30, //transformation matrix row 0 (rotation x 3, translation)
        float aT01, float aT11, float aT21, float aT31, //transformation matrix row 1 (rotation x 3, translation)
        float aT02, float aT12, float aT22, float aT32, //transformation matrix row 2 (rotation x 3, translation)
        /*transformation matrix row 3 is 0 0 0 1 */
        int aObjectId,
        float aMinX, float aMinY, float aMinZ, //min bound
        float aMaxX, float aMaxY, float aMaxZ //max bound
        ) const;

};

#endif // OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D
