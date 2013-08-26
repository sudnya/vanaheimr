/*! \file   Module.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Module class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/Global.h>
#include <vanaheimr/ir/interface/Constant.h>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{

/*! \brief A namespace for the internal representation */
namespace ir
{

/*! \brief A base class for modules of different types */
class ModuleBase
{
public:
	virtual ~ModuleBase();

public:
	/*! \brief Write the module to a binary */
	virtual void writeBinary(std::ostream&) const = 0;

	/*! \brief Write the module as IR to an assembly file */
	virtual void writeAssembly(std::ostream&) const = 0;
};

/*! \brief Represents a single compilation unit. */
class Module : public ModuleBase
{
public:
	typedef std::list<Function>  FunctionList;
	typedef std::list<Global>    GlobalList;
	typedef std::list<Constant*>  ConstantList;

	typedef FunctionList::iterator       iterator;
	typedef FunctionList::const_iterator const_iterator;

	typedef GlobalList::iterator         global_iterator;
	typedef GlobalList::const_iterator   const_global_iterator;

	typedef ConstantList::iterator         constant_iterator;
	typedef ConstantList::const_iterator   const_constant_iterator;

public:
	/*! \brief Create a new module with the specified name */
	Module(const std::string& name, compiler::Compiler* compiler);
	~Module();
	
public:
	/*! \brief Deep copy */
	Module(const Module& r);
	Module& operator=(const Module& r);
	
public:
	/*! \brief Get a named function in the module, return 0 if not found */
	iterator getFunction(const std::string& name);

	/*! \brief Get a named function in the module, return 0 if not found */
	const_iterator getFunction(const std::string& name) const;
	
	/*! \brief Insert a function into the module, it takes ownership */
	iterator insertFunction(iterator position, const Function& f);

	/*! \brief Add a new function */
	iterator newFunction(const std::string& name, Variable::Linkage l,
		Variable::Visibility v, const Type* t = 0);

	/*! \brief Remove a function from the module, it is deleted */
	iterator removeFunction(iterator f);

public:
	/*! \brief Get a named global in the module, return 0 if not found */
	global_iterator getGlobal(const std::string& name);

	/*! \brief Get a named global in the module, return 0 if not found */
	const_global_iterator getGlobal(const std::string& name) const;
	
	/*! \brief Insert a global into the module, it takes ownership */
	global_iterator insertGlobal(global_iterator, const Global& g);
	
	/*! \brief Create a new global, the module owns it */
	global_iterator newGlobal(const std::string& name,
		const Type* t, Variable::Linkage l, ir::Global::Level le);

	/*! \brief Remove a global from the module, it is deleted */
	global_iterator removeGlobal(global_iterator g);

public:
	/*! \brief Write the module to a binary */
	void writeBinary(std::ostream&) const;

	/*! \brief Write the module as IR to an assembly file */
	void writeAssembly(std::ostream&) const;

public:
	//! Function Iteration
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	//! Function info
	size_t size()  const;
	bool   empty() const;

public:
	//! Global Iteration
	global_iterator       global_begin();
	const_global_iterator global_begin() const;

	global_iterator       global_end();
	const_global_iterator global_end() const;

public:
	//! Global info
	size_t global_size()  const;
	bool   global_empty() const;

public:
	//! Constant Iteration
	constant_iterator       constant_begin();
	const_constant_iterator constant_begin() const;

	constant_iterator       constant_end();
	const_constant_iterator constant_end() const;

public:
	//! Constant info
	size_t constant_size()  const;
	bool   constant_empty() const;
	
public:
	void clear();
	
public:
	std::string name;
	
private:
	FunctionList _functions;
	GlobalList   _globals;
	ConstantList _constants;
	
private:
	compiler::Compiler* _compiler;
};

}

}

