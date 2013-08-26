/*! \file   XYZ.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the XVZ class for 3 D coordinates 
*/
#pragma once
namespace baldr
{
class XYZ
{
    public:
        XYZ(float X=0.0f, float Y=0.0f, float Z=0.0f) : m_x(X), m_y(Y), m_z(Z) {};
        float dotProduct(XYZ v2);
        XYZ crossProduct(XYZ v2);
        XYZ scalarProduct(float k);
        XYZ scalarDivide(float k);
        XYZ add(XYZ v2);
        XYZ subtract(XYZ v2);
        float distance(XYZ v2);
        float getX() { return m_x; };
        float getY() { return m_y; };
        float getZ() { return m_z; };
        
        void setX(float x) { m_x = x; };
        void setY(float y) { m_y = y; };
        void setZ(float z) { m_z = z; };
    private:
        float m_x, m_y, m_z;
};
}
