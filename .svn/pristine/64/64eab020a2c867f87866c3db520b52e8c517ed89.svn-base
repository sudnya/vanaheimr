/*! \file   ConvertThreadsToSIMDPass.h
	\author Gregory Diamos <solusstutlus@gmail.com>
	\date   Tuesday September 11, 2012
	\brief  The header file for the ConvertThreadsToSIMDPass class.
*/

#pragma once

// Vanahimer Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace transforms
{

class ConvertThreadsToSIMDPass : public FunctionPass
{
public:
	ConvertThreadsToSIMDPass();
	
public:
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;

};

}

}

 
