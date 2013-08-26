/*	\file   OperandAccess.h
	\date   Tuesday February 5, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for Operand accessor functions.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Operand.h>

// Forward Declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }

namespace vanaheimr { namespace as { class Call; } }

namespace archaeopteryx
{

namespace executive
{

// Operand Access
__device__ uint64_t getOperand(
	const vanaheimr::as::OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId);

__device__ uint64_t getOperand(const vanaheimr::as::Operand& operand,
	CoreSimBlock* parentBlock, unsigned threadId);

__device__ unsigned int getReturnRegister(const vanaheimr::as::Call* call,
	CoreSimBlock* parentBlock);
	
__device__ uint64_t getOperand(const vanaheimr::as::Call* call,
	CoreSimBlock* parentBlock, unsigned threadId, unsigned int index);

// Register Access
__device__ void setRegister(vanaheimr::as::OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId, uint64_t result);

__device__ void setRegister(unsigned int reg, CoreSimBlock* parentBlock,
	unsigned threadId, uint64_t result);

}

}


