/*! \file   ChaitinBriggsRegisterAllocatorPass.h
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ChaitinBriggsRegisterAllocatorPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/RegisterAllocator.h>

#include <vanaheimr/util/interface/LargeMap.h>

// Forward Declarations
namespace vanaheimr { namespace machine { class MachineModel; } }

namespace vanaheimr
{

namespace codegen
{

class ChaitinBriggsRegisterAllocatorPass : public RegisterAllocator
{
public:
	ChaitinBriggsRegisterAllocatorPass();

public:
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;

public:
	/*! \brief Get the set of values that were spilled during allocation */
	VirtualRegisterSet getSpilledRegisters();
	
	/*! \brief Get the mapping of a value to a named physical register */
	const machine::PhysicalRegister* getPhysicalRegister(
		const ir::VirtualRegister&) const;

private:
	typedef util::LargeMap<unsigned int, unsigned int> RegisterMap;

private:
	VirtualRegisterSet _spilled;
	RegisterMap        _allocated;

private:
	const machine::MachineModel* _machine;
};

}

}


