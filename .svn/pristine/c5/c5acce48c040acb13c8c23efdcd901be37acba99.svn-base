/*! \file   TranslationTableInstructionSelectionPass.cpp
	\date   Tuesday February 26, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the
		    TranslationTableInstructionSelectionPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/TranslationTableInstructionSelectionPass.h>

#include <vanaheimr/machine/interface/MachineModel.h>
#include <vanaheimr/machine/interface/TranslationTable.h>
#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Function.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <cassert>

namespace vanaheimr
{

namespace codegen
{

TranslationTableInstructionSelectionPass::TranslationTableInstructionSelectionPass()
: FunctionPass({}, "TranslationTableInstructionSelectionPass")
{
	
}

void TranslationTableInstructionSelectionPass::runOnFunction(Function& f)
{
	// Parallel for all
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		_lowerBlock(*block);
	}
}

transforms::Pass* TranslationTableInstructionSelectionPass::clone() const
{
	return new TranslationTableInstructionSelectionPass;
}

static void lowerInstruction(ir::BasicBlock::InstructionList& instructions,
	ir::Instruction* instruction,
	const machine::TranslationTable* translationTable);

void TranslationTableInstructionSelectionPass::_lowerBlock(BasicBlock& block)
{
	hydrazine::log("TranslationTableInstructionSelectionPass")
		<< "Running on basic block " << block.name() << "\n";
	
	auto machineModel = compiler::Compiler::getSingleton()->getMachineModel();
	
	BasicBlock::InstructionList loweredInstructions;

	auto translationTable = machineModel->translationTable();
	
	// Parallel for all and final gather
	for(auto instruction : block)
	{
		lowerInstruction(loweredInstructions, instruction, translationTable);
	}

	// Swap out the block contents, deallocate it
	block.clear();

	block.assign(loweredInstructions.begin(), loweredInstructions.end());
}

static void lowerInstruction(ir::BasicBlock::InstructionList& instructions,
	ir::Instruction* instruction,
	const machine::TranslationTable* translationTable)
{
	hydrazine::log("TranslationTableInstructionSelectionPass")
		<< " For instruction: " << instruction->toString() << "\n";
	
	// if the translation table is missing
	if(translationTable == nullptr )
	{
		// if the instruction was already a machine instruction,
		//  assume no translation is needed
		if(instruction->isMachineInstruction())
		{
			hydrazine::log("TranslationTableInstructionSelectionPass")
				<< "  skipped, already a machine instruction.\n";
			
			instructions.push_back(instruction->clone());
			return;
		}
		
		// otherwise, it really is an error
		throw std::runtime_error("No translation table, cannot translate " +
			instruction->toString());
	}

	auto machineInstructions =
		translationTable->translateInstruction(instruction);

	if(machineInstructions.empty())
	{
		// if the translation fails, but the instruction was
		//  already a machine instruction, assume no translation is needed
		if(instruction->isMachineInstruction())
		{
			hydrazine::log("TranslationTableInstructionSelectionPass")
				<< "  skipped, already a machine instruction.\n";
			
			instructions.push_back(instruction->clone());
			return;
		}
		
		// otherwise, it really is an error
		throw std::runtime_error("No translation table entry matches " +
			instruction->toString());
	}
	
	instructions.insert(instructions.end(), machineInstructions.begin(),
		machineInstructions.end());
}

}

}

