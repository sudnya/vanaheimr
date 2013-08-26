/*! \file   Compiler.cpp
	\date   Sunday February 12, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Compiler class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/machine/interface/MachineModelFactory.h>
#include <vanaheimr/machine/interface/MachineModel.h>

#include <vanaheimr/parser/interface/TypeParser.h>

#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <sstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

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
	_types.push_back(new ir::VoidType(this));

	// Create the machine model
	_machineModel = machine::MachineModelFactory::createDefaultMachine();

	assert(_machineModel != nullptr);
}

Compiler::~Compiler()
{
	for(auto type : *this) delete type;

	delete getMachineModel();
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
	
Compiler::iterator Compiler::newType(const ir::Type& type)
{
	assert(getType(type.name) == nullptr);

	report("Added type: '" << type.name << "'");
	
	return _types.insert(_types.end(), type.clone());
}

Compiler::iterator Compiler::getOrInsertType(const ir::Type& type)
{
	for(iterator t = begin(); t != end(); ++t)
	{
		if(type.name == (*t)->name) return t;
	}

	return newType(type);
}

Compiler::iterator Compiler::getOrInsertType(const std::string& signature)
{
	report("Parsing type with signature: '" << signature << "'");
	
	parser::TypeParser parser(this);
	
	std::stringstream stream(signature);
	
	parser.parse(&stream);
	
	return getOrInsertType(*parser.parsedType()->clone());
}

Compiler::module_iterator Compiler::getModule(const std::string& name)
{
	for(module_iterator module = module_begin();
		module != module_end(); ++module)
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
		if((*type)->name == name) return *type;
	}
	
	return 0;
}

const ir::Type* Compiler::getType(const std::string& typeName) const
{
	const_iterator type = _types.begin();
	
	for( ; type != _types.end(); ++type)
	{
		if((*type)->name == typeName) return *type;
	}
	
	return 0;
}

const ir::Type* Compiler::getBasicBlockType() const
{
	return getType("_ZTBasicBlock");
}

const machine::MachineModel* Compiler::getMachineModel() const
{
	return _machineModel;
}

machine::MachineModel* Compiler::getMachineModel()
{
	return _machineModel;
}

void Compiler::switchToNewMachineModel(const std::string& name)
{
	delete _machineModel;
	
	_machineModel = machine::MachineModelFactory::createMachineModel(name);
	
	assert(_machineModel != nullptr);
}

Compiler* Compiler::getSingleton()
{
	return &singleton;
}

}

}


