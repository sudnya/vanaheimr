/*! \file   Global.cpp
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Global class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Global.h>
#include <vanaheimr/ir/interface/Constant.h>
#include <vanaheimr/ir/interface/Type.h>

namespace vanaheimr
{

namespace ir
{

Global::Global(const std::string& n, Module* m,
	const Type* t, Linkage l, Visibility v,
	Constant* c, unsigned int le)
: Variable(n, m, t, l, v), _initializer(c), _level(le)
{

}

Global::~Global()
{
	delete _initializer;
}

Global::Global(const Global& g)
: Variable(g), _initializer(0), _level(g.level())
{
	if(g.hasInitializer())
	{
		setInitializer(g.initializer()->clone());
	}
}

Global& Global::operator=(const Global& g)
{
	Variable::operator=(g);

	delete _initializer;
	
	if(g.hasInitializer())
	{
		setInitializer(g.initializer()->clone());
		setLevel(g.level());
	}
	
	return *this;
}

bool Global::hasInitializer() const
{
	return _initializer != 0;
}

Constant* Global::intializer()
{
	return _initializer;
}

const Constant* Global::initializer() const
{
	return _initializer;
}

unsigned int Global::level() const
{
	return _level;
}

size_t Global::bytes() const
{
	if(_initializer == 0)
	{
		return type().bytes();
	}
	
	return _initializer->bytes();
}

void Global::setInitializer(Constant* c)
{
	delete _initializer;
	
	_initializer = c;
}

void Global::setLevel(unsigned int l)
{
	_level = l;
}

}

}


