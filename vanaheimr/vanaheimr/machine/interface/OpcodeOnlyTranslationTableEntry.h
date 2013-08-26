/*! \file   OpcodeOnlyTranslationTableEntry.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the OpcodeOnlyTranslationTableEntry class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

// Forward Declarations
namespace vanaheimr { namespace machine { class Operation; } }

namespace vanaheimr { namespace ir { class Type;     } }
namespace vanaheimr { namespace ir { class Constant; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A rule for translating only the opcode in an instruction  */
class OpcodeOnlyTranslationTableEntry : public TranslationTableEntry
{
public:
	/*! \brief Create the rule and specify the opcode mapping */
	OpcodeOnlyTranslationTableEntry(const std::string& sourceOpcode,
		const std::string& destinationOpcode, const std::string& special);

public:
	/*! \brief Translate IR instruction into equivalent machine instructions */
	virtual MachineInstructionVector translateInstruction(
		const ir::Instruction*) const;

public:
	virtual TranslationTableEntry* clone() const;

public:
	std::string machineInstructionOpcode;
	std::string machineInstructionSpecialProperty;
};

}

}



