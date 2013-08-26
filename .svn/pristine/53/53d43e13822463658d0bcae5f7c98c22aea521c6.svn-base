/*! \file   GenericSpillCodePass.h
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the GenericSpillCodePass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace codegen
{

class GenericSpillCodePass : public transforms::FunctionPass
{
public:
	GenericSpillCodePass();

public:
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;
};

}

}


