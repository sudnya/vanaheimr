/*	\file   VirtualRegister.cpp
	\date   Thursday March 1, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Operand class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/VirtualRegister.h>
#include <vanaheimr/ir/interface/Type.h>

// Standard Library Includes
#include <sstream>

namespace vanaheimr
{

namespace ir
{

VirtualRegister::VirtualRegister(const std::string& n, Id i,
	Function* f, const Type* t)
: name(n), id(i), function(f), type(t)
{

}

std::string VirtualRegister::toString() const
{
	std::stringstream stream;

	stream << type->name() << " %r" << id;

	return stream.str();
}

}

}


