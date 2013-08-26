/*	\file   OperandAccess.cu
	\date   Tuesday February 5, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for Operand accessor functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/OperandAccess.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Operand.h>
#include <vanaheimr/asm/interface/Instruction.h>

// Forward Declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }

namespace archaeopteryx
{

namespace executive
{

typedef vanaheimr::as::Operand          Operand;
typedef vanaheimr::as::RegisterOperand  RegisterOperand;
typedef vanaheimr::as::PredicateOperand PredicateOperand;
typedef vanaheimr::as::ImmediateOperand ImmediateOperand;
typedef vanaheimr::as::IndirectOperand  IndirectOperand;
typedef vanaheimr::as::OperandContainer OperandContainer;

__device__ uint64_t getRegisterOperand(
	const Operand& operand, CoreSimBlock* block, unsigned threadId)
{
	const RegisterOperand& reg =
		static_cast<const RegisterOperand&>(operand); 

	uint64_t value = block->getRegister(threadId, reg.reg);

	return value;
}

__device__ uint64_t getImmediateOperand(
	const Operand& operand, CoreSimBlock* block, unsigned threadId)
{
	const ImmediateOperand& imm =
		static_cast<const ImmediateOperand&>(operand); 

	return imm.uint;
}

__device__ uint64_t getPredicateOperand(
	const Operand& operand, CoreSimBlock* block, unsigned threadId)
{
	const PredicateOperand& reg =
		static_cast<const PredicateOperand&>(operand); 
	//FIX ME	
	
	uint64_t value = block->getRegister(threadId, reg.reg);

	switch(reg.modifier)
	{
	case PredicateOperand::StraightPredicate:
	{
		value = value;
		break;
	}
	// TODO
	}

	return value;
}

__device__ uint64_t getIndirectOperand(
	const Operand& operand, CoreSimBlock* block, unsigned threadId)
{
	const IndirectOperand& indirect =
		static_cast<const IndirectOperand&>(operand); 
	
	uint64_t address = block->getRegister(threadId, indirect.reg) +
		indirect.offset;

	//FIXMe	
	return address;
}

__device__ uint64_t getSymbolOperand(
	const Operand& operand, CoreSimBlock* block, unsigned threadId)
{
	device_assert_m(false, "Symbol operands not supported in emulator, "
		"they should have been lowered!");

	return 0;
}

typedef uint64_t (*GetOperandValuePointer)(const Operand&,
	CoreSimBlock*, unsigned);

__device__ GetOperandValuePointer getOperandFunctionTable[] = {
	getRegisterOperand,
	getImmediateOperand,
	getPredicateOperand,
	getIndirectOperand,
	getSymbolOperand
};

__device__ uint64_t getOperand(const Operand& operand,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	GetOperandValuePointer function = getOperandFunctionTable[operand.mode];

	return function(operand, parentBlock, threadId);
}

__device__ uint64_t getOperand(
	const OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	return getOperand(operandContainer.asOperand, parentBlock, threadId);
}

__device__ unsigned int getReturnRegister(const vanaheimr::as::Call* call,
	CoreSimBlock* block)
{
	device_assert(call->returnArguments > 0);
	
	vanaheimr::as::OperandContainer operand;
	
	block->binary()->copyDataToAddress(&operand, call->returnArgumentOffset,
		sizeof(vanaheimr::as::OperandContainer));

	device_assert(operand.asOperand.mode == vanaheimr::as::Operand::Register);
	
	return operand.asRegister.reg;
}

__device__ uint64_t getOperand(const vanaheimr::as::Call* call,
	CoreSimBlock* block, unsigned threadId, unsigned int index)
{
	device_assert(call->arguments > index);
	
	vanaheimr::as::OperandContainer operand;
	
	block->binary()->copyDataToAddress(&operand,
		call->argumentOffset + index * sizeof(vanaheimr::as::OperandContainer),
		sizeof(vanaheimr::as::OperandContainer));
	
	return getOperand(operand, block, threadId);
}

__device__ void setRegister(OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId, uint64_t result)
{
	const RegisterOperand& reg = operandContainer.asRegister;

	parentBlock->setRegister(threadId, reg.reg, result);
}

__device__ void setRegister(unsigned int reg, CoreSimBlock* parentBlock,
	unsigned threadId, const uint64_t result)
{
	parentBlock->setRegister(threadId, reg, result);
}

}

}


