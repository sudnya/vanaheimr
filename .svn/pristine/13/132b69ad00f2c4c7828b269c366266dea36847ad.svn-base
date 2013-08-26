#include <iostream>
#include <baldr/include/Renderer.h>
#include <baldr/include/Sphere.h>

int main()
{
    std::cout << "Starting TestSphere\n";
    std::cout << "Setup Camera at (0,0,0)\n";
    baldr::XYZ cameraPos(0, 0, 0);
    baldr::XYZ dx(0,10,0);
    baldr::XYZ dy(0,0,10);
    baldr::XYZ corner(1,0,0);
    baldr::Viewport viewport(dx,dy,corner);
    
    baldr::XYZ centerOfSphere(5,5,5);
    float radius = 4;
    baldr::SceneObjects::Sphere s(radius, centerOfSphere);
    unsigned width = 80;
    unsigned height = 40;
    baldr::Renderer renderer(cameraPos, viewport, width, height);
    renderer.AddObjectToScene(s);

    renderer.renderScene();
}
