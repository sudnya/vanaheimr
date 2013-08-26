/*! \file   GenericSpillCodePass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the GenericSpillCodePass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/GenericSpillCodePass.h>

#include <vanaheimr/codegen/interface/RegisterAllocator.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace codegen
{

GenericSpillCodePass::GenericSpillCodePass()
: FunctionPass({}, "GenericSpillCodePass")
{

}

void GenericSpillCodePass::runOnFunction(Function& f)
{
	auto pass = static_cast<RegisterAllocator*>(getPass("register-allocator"));
	assert(pass != nullptr);
	
	auto spilled = pass->getSpilledRegisters();
	
	assertM(spilled.empty(), "Spilling not implemented");	
}

transforms::Pass* GenericSpillCodePass::clone() const
{
	return new GenericSpillCodePass;
}

}

}


