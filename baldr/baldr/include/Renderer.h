/*! \file   Renderer.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Renderer class 
        
*/
#pragma once
// Standard Library Includes
#include <vector>

//other includes
//#include <baldr/include/Shape.h>
#include <baldr/include/Sphere.h>
#include <baldr/include/XYZ.h>
#include <baldr/include/Ray.h>

//Forward declarations

namespace baldr
{
class Viewport
{
    public:
        Viewport(XYZ dx, XYZ dy, XYZ corner) : m_dx(dx, corner), m_dy(dy, corner), m_corner(corner) {};
        Ray m_dx, m_dy;
        XYZ m_corner;
};

class Renderer
{
    public:
        //typedef std::vector<SceneObjects::Shape*> ObjectsInScene;
        typedef std::vector<SceneObjects::Sphere> ObjectsInScene;

        Renderer(XYZ camera, Viewport viewport, unsigned width, unsigned height) :
            m_camera(camera), m_viewport(viewport), m_width(width), m_height(height) {};
        //void AddObjectToScene(SceneObjects::Shape s) { m_objects.push_back(s); };
        void AddObjectToScene(SceneObjects::Sphere s) { m_objects.push_back(s); };

        void renderScene();

    private:
        ObjectsInScene m_objects;
        XYZ m_camera;
        Viewport m_viewport;
        unsigned m_width, m_height;
};
}
