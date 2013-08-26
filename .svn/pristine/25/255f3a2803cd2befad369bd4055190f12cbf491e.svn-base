/*! \file   RegisterAllocator.h
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the RegisterAllocator class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

#include <vanaheimr/util/interface/LargeSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir      { class VirtualRegister;  } }
namespace vanaheimr { namespace machine { class PhysicalRegister; } }

namespace vanaheimr
{

namespace codegen
{

/*! \brief Presents a generic interface for register allocation */
class RegisterAllocator : public transforms::FunctionPass
{

public:
	typedef util::LargeSet<ir::VirtualRegister*> VirtualRegisterSet;

public:
	/*! \brief The default constructor sets the type */
	RegisterAllocator(const StringVector& analyses = StringVector(),
		const std::string& n = "");
	virtual ~RegisterAllocator();

public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const Module& m);
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnFunction(Function& f) = 0;		
	/*! \brief Finalize the pass */
	virtual void finalize();

public:
	/*! \brief Get the set of values that were spilled during allocation */
	virtual VirtualRegisterSet getSpilledRegisters() = 0;
	
	/*! \brief Get the mapping of a value to a named physical register */
	virtual const machine::PhysicalRegister* getPhysicalRegister(
		const ir::VirtualRegister&) const = 0;

};

}

}


