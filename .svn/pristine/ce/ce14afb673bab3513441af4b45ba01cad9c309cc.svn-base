/*! \file   Sphere.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Sphere class 
        
*/
#pragma once
// Standard Library Includes

//other includes
#include <baldr/include/Shape.h>
#include <baldr/include/XYZ.h>
#include <baldr/include/Ray.h>
// Forward Declarations

namespace baldr
{
namespace SceneObjects
{
class Sphere : public Shape
{
    public:
        Sphere(float radius, XYZ centre) : m_radius(radius), m_centre(centre) {};
        virtual bool doesIntersect(const Ray& R);
    //something specific to only spheres
    private:
        float m_radius;
        XYZ   m_centre;        
};
}
}
