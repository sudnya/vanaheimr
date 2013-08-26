/*! \file   OpcodeOnlyTranslationTableEntry.cpp
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the OpcodeOnlyTranslationTableEntry class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/OpcodeOnlyTranslationTableEntry.h>
#include <vanaheimr/machine/interface/MachineModel.h>
#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/compiler/interface/Compiler.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

OpcodeOnlyTranslationTableEntry::OpcodeOnlyTranslationTableEntry(
	const std::string& s, const std::string& d, const std::string& sp)
: TranslationTableEntry(s), machineInstructionOpcode(d),
	machineInstructionSpecialProperty(sp)
{

}

static const Operation* getOrAddOperation(const std::string& opcode,
	const std::string& special)
{
	auto compiler = vanaheimr::compiler::Compiler::getSingleton();

	auto operation = compiler->getMachineModel()->getOperation(opcode);

	if(operation == nullptr)
	{
		compiler->getMachineModel()->addOperation(Operation(opcode, special));
		
		operation = compiler->getMachineModel()->getOperation(opcode);
	}

	return operation;
}

OpcodeOnlyTranslationTableEntry::MachineInstructionVector
	OpcodeOnlyTranslationTableEntry::translateInstruction(
	const ir::Instruction* instruction) const
{
	auto machineInstruction = new Instruction(
		getOrAddOperation(machineInstructionOpcode,
			machineInstructionSpecialProperty),
		instruction->block);
	
	machineInstruction->clear();
	
	for(auto read : instruction->reads)
	{
		machineInstruction->appendRead(read->clone());
	}
	
	for(auto write : instruction->writes)
	{
		machineInstruction->appendWrite(write->clone());
	}
	
	MachineInstructionVector instructions;
	
	hydrazine::log("OpcodeOnlyTranslationTableEntry")
		<< "  to " << machineInstruction->toString() << "\n";
	
	instructions.push_back(machineInstruction);
	
	return instructions;
}

TranslationTableEntry* OpcodeOnlyTranslationTableEntry::clone() const
{
	return new OpcodeOnlyTranslationTableEntry(*this);
}

}

}



