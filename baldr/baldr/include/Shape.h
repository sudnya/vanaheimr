/*! \file   Shape.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Shape class which is base of all 3D objects.
        
*/
#pragma once
// Standard Library Includes


// Forward Declarations
namespace baldr { class Ray; }
namespace baldr
{
namespace SceneObjects
{
class Shape
{
    public:
        virtual bool doesIntersect (const Ray& R) = 0;
};//class Shape ends
}//SceneObjects ends
}
