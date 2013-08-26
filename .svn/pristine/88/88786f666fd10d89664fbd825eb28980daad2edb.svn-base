/*! \file   Argument.cpp
	\date   Saturday February 11, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Argument class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Argument.h>
#include <vanaheimr/ir/interface/Function.h>

namespace vanaheimr
{

namespace ir
{

Argument::Argument(const Type* t, Function* f, const std::string& n)
: _type(t), _function(f), _name(n)
{

}

std::string Argument::mangledName() const
{
	// TODO make this compatible with GCC
	return "_Z" + _function->name() + "_" + _name;
}

const std::string& Argument::name() const
{
	return _name;
}

const Type& Argument::type() const
{
	return *_type;
}

void Argument::setFunction(Function* f)
{
	_function = f;
}

}

}


