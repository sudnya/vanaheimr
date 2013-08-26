/*! \file   MachineToVIROpcodeTranslationRule.h
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineToVIROpcodeTranslationRule class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/MachineToVIRInstructionTranslationRule.h>

namespace vanaheimr
{

namespace transforms
{

class MachineToVIROpcodeTranslationRule
: public MachineToVIRInstructionTranslationRule
{
public:
	MachineToVIROpcodeTranslationRule(const std::string& sourceOpcode,
		const std::string& destinationOpcode);
	
public:
	InstructionVector translateMachineInstruction(
		const machine::Instruction* instruction);

	MachineToVIRInstructionTranslationRule* clone() const;

public:
	std::string destinationOpcode; // The opcode that the rule translates to
};

}

}

