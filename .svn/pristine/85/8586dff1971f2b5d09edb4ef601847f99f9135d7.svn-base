/*! \file   Compiler.cpp
	\date   Sunday February 12, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Compiler class.
	
*/

// Standard Library Includes
#include <vanaheimr/compiler/interface/Compiler.h>
#include <vanaheimr/ir/interface/Type.h>

namespace vanaheimr
{

namespace compiler
{

static Compiler singleton;

Compiler::Compiler()
{
	// TODO Add in common types
	_types.push_back(new ir::IntegerType(this, 1) );
	_types.push_back(new ir::IntegerType(this, 8) );
	_types.push_back(new ir::IntegerType(this, 16));
	_types.push_back(new ir::IntegerType(this, 32));
	_types.push_back(new ir::IntegerType(this, 64));

	_types.push_back(new ir::FloatType(this));
	_types.push_back(new ir::DoubleType(this));

	_types.push_back(new ir::BasicBlockType(this));
}

Compiler::~Compiler()
{
	for(auto type : *this) delete type;
}

Compiler::iterator Compiler::begin()
{
	return _types.begin();
}

Compiler::const_iterator Compiler::begin() const
{
	return _types.begin();
}

Compiler::iterator Compiler::end()
{
	return _types.end();
}

Compiler::const_iterator Compiler::end() const
{
	return _types.end();
}

bool Compiler::empty() const
{
	return _types.empty();
}

size_t Compiler::size() const
{
	return _types.size();
}

Compiler::module_iterator Compiler::module_begin()
{
	return _modules.begin();
}

Compiler::const_module_iterator Compiler::module_begin() const
{
	return _modules.begin();
}

Compiler::module_iterator Compiler::module_end()
{
	return _modules.end();
}

Compiler::const_module_iterator Compiler::module_end() const
{
	return _modules.end();
}

Compiler::module_iterator Compiler::newModule(const std::string& name)
{
	return _modules.insert(_modules.end(), ir::Module(name, this));
}

Compiler::module_iterator Compiler::getModule(const std::string& name)
{
	for(module_iterator module = module_begin(); module != module_end(); ++module)
	{
		if(module->name == name) return module;
	}
	
	return module_end();
}

Compiler::const_module_iterator Compiler::getModule(
	const std::string& name) const
{
	const_module_iterator module = module_end();
	
	for( ; module != module_end(); ++module)
	{
		if(module->name == name) break;
	}
	
	return module;
}

ir::Type* Compiler::getType(const std::string& name)
{
	iterator type = _types.begin();
	
	for( ; type != _types.end(); ++type)
	{
		if((*type)->name() == name) return *type;
	}
	
	return 0;
}

const ir::Type* Compiler::getType(const std::string& typeName) const
{
	const_iterator type = _types.begin();
	
	for( ; type != _types.end(); ++type)
	{
		if((*type)->name() == typeName) return *type;
	}
	
	return 0;
}

const ir::Type* Compiler::getBasicBlockType() const
{
	return getType("_ZTBasicBlock");
}

Compiler* Compiler::getSingleton()
{
	return &singleton;
}

}

}


