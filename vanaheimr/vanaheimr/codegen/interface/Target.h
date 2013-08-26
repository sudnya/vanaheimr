/*! \file   Target.h
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Target class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class ModuleBase; } }
namespace vanaheimr { namespace ir { class Module;     } }

namespace vanaheimr
{

namespace codegen
{

/*! \brief Abstract interface for Target-specific code generation */
class Target
{
public:
	explicit Target(const std::string& name);
	virtual ~Target();

public:
	/*! \brief Allocate and return a new target with the specified name */
	static Target* createTarget(const std::string& name);

	/*! \brief Register a new target, it will be copied */
	static void registerTarget(const Target*);

public:
	/*! \brief Assign a module to the target, it is allowed to be
		modified by the target */
	void assignModule(ir::Module* module);

public:
	/*! \brief Lower the assigned module to the target ISA */
	virtual void lower() = 0;
	/*! \brief Get lowered module in the target ISA */
	virtual ir::ModuleBase* getLoweredModule() = 0;
	
public:
	/*! \brief Create a copy of the target */
	virtual Target* clone() const = 0;
	
public:
	/*! \brief Get the target name */
	const std::string& name() const;
	
protected:
	ir::Module* _module;

private:
	std::string _name;

};

}

}


