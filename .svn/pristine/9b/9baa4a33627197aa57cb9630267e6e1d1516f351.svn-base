/*! \file   Operation.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the Operation class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class FunctionalUnitOperation; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief An opcode in the machine ISA.  Intentionally generic. */
class Operation
{
public:
	typedef std::vector<FunctionalUnitOperation*> FunctionalUnitOperationVector;

public:
	Operation(const std::string& _name,  const std::string& _special = "",
		unsigned int _latency = 0);

public:
	std::string  name;    // fully qualified name including modifiers
	std::string  special; // special property (if any)
	unsigned int latency; // latency in cycles
	
	/*! \brief all possible bindings to HW */
	FunctionalUnitOperationVector functionalUnitOperations;
};

}

}


