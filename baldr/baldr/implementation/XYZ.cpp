/*! \file   XYZ.cpp
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The implementation file for the XVZ class for 3 D coordinates 
*/
#include <cmath>

#include <baldr/include/XYZ.h>

namespace baldr
{
    float XYZ::dotProduct(XYZ v2)
    {
        return (this->getX()*v2.getX()) + (this->getY()*v2.getY()) + (this->getZ()*v2.getZ());
    }

    XYZ XYZ::crossProduct(XYZ v2)
    {
        return *this; //FIX ME WHEN YOU NEED ME
    }

    XYZ XYZ::scalarProduct(float k)
    {
        return XYZ((this->getX()*k), (this->getY()*k), (this->getZ()*k));
    }

    XYZ XYZ::scalarDivide(float k)
    {
        return XYZ((this->getX()/k), (this->getY()/k), (this->getZ()/k));
    }

    XYZ XYZ::add(XYZ v2)
    {
        XYZ temp;
        temp.setX (this->getX() + v2.getX());
        temp.setY (this->getY() + v2.getY());
        temp.setZ (this->getZ() + v2.getZ());
        return temp;
    }

    XYZ XYZ::subtract(XYZ v2)
    {
        XYZ temp;
        temp.setX (this->getX() - v2.getX());
        temp.setY (this->getY() - v2.getY());
        temp.setZ (this->getZ() - v2.getZ());
        return temp;
    }

    float XYZ::distance(XYZ v2)
    {
        XYZ distance = this->subtract(v2);
        return sqrt((distance.getX()*distance.getX()) + (distance.getY()*distance.getY()) + (distance.getZ()*distance.getZ()));
/*        float x, y, z;
        x = this->getX() - v2->getX();
        y = this->getY() - v2->getY();
        z = this->getZ() - v2->getZ();
        return sqrt((x*x) + (y*y) + (z*z));
*/    }

}
