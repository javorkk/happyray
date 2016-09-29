#include "StdAfx.hpp"
#include "ObjWriter.h"

void ObjWriter::writeVertex( std::ofstream& aStream, float aX, float aY, float aZ ) const
{
    aStream << "v " << aX << " " << aY << " " << aZ << "\n";
}

void ObjWriter::writeVertexNormal( std::ofstream& aStream, float aX, float aY, float aZ ) const
{
    aStream << "vn " << aX << " " << aY << " " << aZ << "\n";

}

void ObjWriter::writeObjectHeader( std::ofstream& aStream, int aObjId ) const
{
    aStream << "o " << aObjId << "\n";
}

void ObjWriter::writeTriangleIndices( std::ofstream& aStream, int aA, int aB, int aC ) const
{
    aStream << "f ";
    aStream << aA + 1 << "//" << aA + 1 << " ";
    aStream << aB + 1 << "//" << aB + 1 << " ";
    aStream << aC + 1 << "//" << aC + 1 << "\n";

}

void ObjWriter::writeInstance(std::ofstream& aStream,
    float aT00, float aT10, float aT20, float aT30, /*transformation matrix row 0 (rotation x 3, translation) */
    float aT01, float aT11, float aT21, float aT31, /*transformation matrix row 1 (rotation x 3, translation */
    float aT02, float aT12, float aT22, float aT32, /*transformation matrix row 2 (rotation x 3, translation */
    /*transformation matrix row 3 is 0 0 0 1 */
    int aObjectId,
    float aMinX, float aMinY, float aMinZ, /*min bound */
    float aMaxX, float aMaxY, float aMaxZ /*max bound */) const
{
    aStream << "#new instance#\n";
    aStream << "obj_id " << aObjectId << "\n";
    aStream << "m_row_0 " << aT00 << " " << aT10 << " " << aT20 << " " << aT30 << "\n";
    aStream << "m_row_1 " << aT01 << " " << aT11 << " " << aT21 << " " << aT31 << "\n";
    aStream << "m_row_2 " << aT02 << " " << aT12 << " " << aT22 << " " << aT32 << "\n";
    aStream << "AABB_min " << aMinX << " " << aMinY << " " << aMinZ << "\n";
    aStream << "AABB_max " << aMaxX << " " << aMaxY << " " << aMaxZ << "\n";
}



