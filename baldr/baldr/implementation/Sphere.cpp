/*! \file   Sphere.cpp
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The implementation file for the Sphere class 
        
*/

// Standard Library Includes
#include <cmath>
#include <iostream>
//other includes
#include <baldr/include/Sphere.h>
// Forward Declarations

namespace baldr
{
namespace SceneObjects
{
    bool Sphere::doesIntersect(const Ray& R)
    {
        //start pt of ray is same as camera?
/*        float cameraToCentre = m_centre.distance(R.getRayStart());
        float distanceVector = std::sqrt((cameraToCentre*cameraToCentre) - (m_radius*m_radius));

        XYZ segmentAlongRay = (R.getRayEquation()).scalarProduct(distanceVector);
        XYZ pointInQuestion = R.getRayStart().add(segmentAlongRay);

        float distanceToPointInQuestion = pointInQuestion.distance(m_centre);
        std::cout << "dist to pt in q = " << distanceToPointInQuestion << " radius "
            << m_radius << "\n";
        */
        float rayMag = std::sqrt((R.getRayEquation().getX()*R.getRayEquation().getX())
                +(R.getRayEquation().getY()*R.getRayEquation().getY())
                +(R.getRayEquation().getZ()*R.getRayEquation().getZ()));
        float centreMag = std::sqrt((m_centre.getX()*m_centre.getX())+(m_centre.getY()*m_centre.getY())+(m_centre.getZ()*m_centre.getZ()));
        XYZ unitRay(R.getRayEquation().getX()/rayMag, R.getRayEquation().getY()/rayMag, R.getRayEquation().getZ()/rayMag);
        XYZ pointInQuestion(unitRay.getX()*centreMag, unitRay.getY()*centreMag, unitRay.getZ()*centreMag);
        float distanceToPointInQuestion = pointInQuestion.distance(m_centre);
//        std::cout << "dist to pt in q = " << distanceToPointInQuestion << " radius "
//            << m_radius << "\n";
        return distanceToPointInQuestion > m_radius ? 0 : 1;
    }
}
}
