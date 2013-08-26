/*! \file   MachineToVIROpcodeTranslationRule.cpp
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineToVIROpcodeTranslationRule class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/MachineToVIROpcodeTranslationRule.h>

#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/ir/interface/Instruction.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

MachineToVIROpcodeTranslationRule::MachineToVIROpcodeTranslationRule(
	const std::string& s, const std::string& d)
: MachineToVIRInstructionTranslationRule(s), destinationOpcode(d)
{

}

MachineToVIROpcodeTranslationRule::InstructionVector
	MachineToVIROpcodeTranslationRule::translateMachineInstruction(
	const machine::Instruction* instruction)
{
	auto virInstruction = ir::Instruction::create(destinationOpcode,
		instruction->block);
	
	virInstruction->clear();
	
	assertM(virInstruction != nullptr, "Invalid destination opcode '" +
		destinationOpcode + "'");
	
	for(auto read : instruction->reads)
	{
		virInstruction->appendRead(read);
	}
	
	for(auto write : instruction->writes)
	{
		virInstruction->appendWrite(write);
	}

	hydrazine::log("MachineToVIROpcodeTranslationRule")
				<< "  translated to '" << virInstruction->toString() << "'.\n";
			
	InstructionVector instructions;
	
	instructions.push_back(virInstruction);
	
	return instructions;
}

MachineToVIRInstructionTranslationRule*
	MachineToVIROpcodeTranslationRule::clone() const
{
	return new MachineToVIROpcodeTranslationRule(*this);
}

}

}

