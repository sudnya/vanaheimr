/*! \file   TranslationTableEntry.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class Instruction; } }
namespace vanaheimr { namespace ir      { class Instruction; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A rule for translating a VIR operation into a Machine equivalent  */
class TranslationTableEntry
{
public:
	typedef std::vector<machine::Instruction*> MachineInstructionVector;

public:
	TranslationTableEntry(const std::string& _name = "");
	virtual ~TranslationTableEntry();

public:
	/*! \brief Translate IR instruction into equivalent machine instructions */
	virtual MachineInstructionVector translateInstruction(
		const ir::Instruction*) const = 0;

public:
	/*! \brief Clone the entry */
	virtual TranslationTableEntry* clone() const = 0;

public:
	std::string name; // name of the VIR operation to translate
};

}

}



