/*! \file   Module.cpp
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Module class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Module.h>

#include <vanaheimr/asm/interface/AssemblyWriter.h>
#include <vanaheimr/asm/interface/BinaryWriter.h>

#include <vanaheimr/compiler/interface/Compiler.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace ir
{

ModuleBase::~ModuleBase()
{

}

Module::Module(const std::string& n, compiler::Compiler* c)
: name(n), _compiler(c)
{

}

Module::Module(const Module& m)
: name(m.name), _compiler(m._compiler)
{
	operator=(m);
}

Module& Module::operator=(const Module& m)
{
	if(&m == this) return *this;
	
	clear();
	
	name      = m.name;
	_compiler = m._compiler;
	
	for(auto function : _functions)
	{
		_functions.push_back(function);
		_functions.back().setModule(this);
	}
	
	for(auto global : _globals)
	{
		_globals.push_back(global);
		_globals.back().setModule(this);
	}
	
	for(auto constant : _constants)
	{
		_constants.push_back(constant->clone());
	}
	
	return *this;
}

Module::~Module()
{
	for(constant_iterator constant = constant_begin();
		constant != constant_end(); ++constant)
	{
		delete *constant;
	}
}

Module::iterator Module::getFunction(const std::string& name)
{
	for(iterator function = begin(); function != end(); ++function)
	{
		if(function->name() == name) return function;
	}
	
	return end();
}

Module::const_iterator Module::getFunction(const std::string& name) const
{
	for(const_iterator function = begin(); function != end(); ++function)
	{
		if(function->name() == name) return function;
	}
	
	return end();
}

Module::iterator Module::insertFunction(iterator position, const Function& f)
{
	return _functions.insert(position, f);
}

Module::iterator Module::newFunction(const std::string& name,
	Variable::Linkage l, Variable::Visibility v, const ir::Type* t)
{
	assert(getFunction(name) == end());
	
	return _functions.insert(end(), Function(name, this, l, v, t));
}

Module::iterator Module::removeFunction(iterator f)
{
	return _functions.erase(f);
}

Module::global_iterator Module::getGlobal(const std::string& name)
{
	for(global_iterator global = global_begin();
		global != global_end(); ++global)
	{
		if(global->name() == name) return global;
	}
	
	return global_end();
}

Module::const_global_iterator Module::getGlobal(const std::string& name) const
{
	for(const_global_iterator global = global_begin();
		global != global_end(); ++global)
	{
		if(global->name() == name) return global;
	}
	
	return global_end();
}

Module::global_iterator Module::insertGlobal(global_iterator position,
	const Global& g)
{
	return _globals.insert(position, g);	
}

Module::global_iterator Module::newGlobal(const std::string& name,
	const Type* t, Variable::Linkage l, ir::Global::Level le)
{
	return _globals.insert(_globals.end(), Global(name, this, t, l,
		Variable::HiddenVisibility, 0, le));
}

Module::global_iterator Module::removeGlobal(global_iterator g)
{
	return _globals.erase(g);
}

void Module::writeBinary(std::ostream& stream) const
{
	as::BinaryWriter writer;
	
	writer.write(stream, *this);
}

void Module::writeAssembly(std::ostream& stream) const
{
	as::AssemblyWriter writer;
	
	writer.write(stream, *this);
}

Module::iterator Module::begin()
{
	return _functions.begin();
}

Module::const_iterator Module::begin() const
{
	return _functions.begin();
}

Module::iterator Module::end()
{
	return _functions.end();
}

Module::const_iterator Module::end() const
{
	return _functions.end();
}

size_t Module::size() const
{
	return _functions.size();
}

bool Module::empty() const
{
	return _functions.empty();
}

Module::global_iterator Module::global_begin()
{
	return _globals.begin();
}

Module::const_global_iterator Module::global_begin() const
{
	return _globals.begin();
}

Module::global_iterator Module::global_end()
{
	return _globals.end();
}

Module::const_global_iterator Module::global_end() const
{
	return _globals.end();
}

size_t Module::global_size() const
{
	return _globals.size();
}

bool Module::global_empty() const
{
	return _globals.empty();
}

Module::constant_iterator Module::constant_begin()
{
	return _constants.begin();
}

Module::const_constant_iterator Module::constant_begin() const
{
	return _constants.begin();
}

Module::constant_iterator Module::constant_end()
{
	return _constants.end();
}

Module::const_constant_iterator Module::constant_end() const
{
	return _constants.end();
}

size_t Module::constant_size() const
{
	return _constants.size();
}

bool Module::constant_empty() const
{
	return _constants.empty();
}

void Module::clear()
{
	for(auto constant : _constants) delete constant;

	_functions.clear();
	_globals.clear();
	_constants.clear();
}

}

}

