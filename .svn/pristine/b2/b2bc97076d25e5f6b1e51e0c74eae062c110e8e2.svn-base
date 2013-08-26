/*! \file   MachineToVIRInstructionTranslationRule.h
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineToVIRInstructionTranslationRule class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace ir { class Instruction; } }

namespace vanaheimr { namespace machine { class Instruction; } }

namespace vanaheimr
{

namespace transforms
{

class MachineToVIRInstructionTranslationRule
{
public:
	typedef std::vector<ir::Instruction*> InstructionVector;

public:
	MachineToVIRInstructionTranslationRule(const std::string& opcodeName);
	virtual ~MachineToVIRInstructionTranslationRule();
	
public:
	virtual InstructionVector translateMachineInstruction(
		const machine::Instruction* instruction) = 0;

	virtual MachineToVIRInstructionTranslationRule* clone() const = 0;

public:
	std::string opcode; // The opcode that the rule applies to
};

}

}


