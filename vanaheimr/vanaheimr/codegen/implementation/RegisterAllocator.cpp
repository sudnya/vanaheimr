/*! \file   RegisterAllocator.cpp
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the RegisterAllocator class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/RegisterAllocator.h>

namespace vanaheimr
{

namespace codegen
{

RegisterAllocator::RegisterAllocator(const StringVector& analyses,
	const std::string& n)
: transforms::FunctionPass(analyses, n, {"register-allocator"})
{
	
}

RegisterAllocator::~RegisterAllocator()
{

}

void RegisterAllocator::initialize(const Module& m)
{

}

void RegisterAllocator::finalize()
{

}

}

}


