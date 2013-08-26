/*	\file   PhysicalRegisterOperand.cpp
	\date   Monday June 25, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the PhysicalRegisterOperand class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/PhysicalRegisterOperand.h>

#include <vanaheimr/machine/interface/PhysicalRegister.h>

#include <vanaheimr/ir/interface/VirtualRegister.h>

namespace vanaheimr
{

namespace machine
{

PhysicalRegisterOperand::PhysicalRegisterOperand(const PhysicalRegister* p,
	VirtualRegister* reg, Instruction* i)
: RegisterOperand(reg, i), physicalRegister(p)
{

}

PhysicalRegisterOperand::Operand* PhysicalRegisterOperand::clone() const
{
	return new PhysicalRegisterOperand(*this);
}

std::string PhysicalRegisterOperand::toString() const
{
	if(physicalRegister == nullptr) return virtualRegister->toString();
	
	return physicalRegister->name();
}

PhysicalRegisterOperand::PhysicalRegisterOperand(const PhysicalRegister* p,
	VirtualRegister* reg, Instruction* i, OperandMode m)
: RegisterOperand(reg, i, m), physicalRegister(p)
{

}

}

}


