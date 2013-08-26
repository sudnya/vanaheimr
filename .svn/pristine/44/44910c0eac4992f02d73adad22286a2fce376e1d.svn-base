/*! \file   Ray.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Ray class 
*/
#pragma once
#include <baldr/include/XYZ.h>

namespace baldr
{
class Ray
{
    public:
        Ray(XYZ coord, XYZ S0) : m_equation(coord), m_startingPoint(S0) {};
        XYZ getRayEquation() const { return this->m_equation; };
        XYZ getRayStart() const { return this->m_startingPoint; };
    private:
        XYZ m_equation;
        XYZ m_startingPoint;
};
}
