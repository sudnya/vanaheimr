/*! \file   Variable.cpp
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Variable class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Variable.h>

namespace vanaheimr
{

namespace ir
{

Variable::Variable(const std::string& name, Module* module,
	const Type* t, Linkage linkage, Visibility v)
: _name(name), _module(module), _linkage(linkage), _visibility(v), _type(t)
{

}

void Variable::setModule(Module* m)
{
	_module = m;
}

const std::string& Variable::name() const
{
	return _name;
}

Module* Variable::module()
{
	return _module;
}

Variable::Linkage Variable::linkage() const
{
	return _linkage;
}

Variable::Visibility Variable::visibility() const
{
	return _visibility;
}

const Type& Variable::type() const
{
	return *_type;
}

}

}

