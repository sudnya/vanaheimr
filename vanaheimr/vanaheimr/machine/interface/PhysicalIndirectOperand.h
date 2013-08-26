/*	\file   PhysicalIndirectOperand.h
	\date   Monday June 25, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the PhysicalIndirectOperand class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/machine/interface/PhysicalRegisterOperand.h>

namespace vanaheimr
{

namespace machine
{

/*! \brief This class represents an indirect operand after assignment to
	a physical register */
class PhysicalIndirectOperand : public PhysicalRegisterOperand
{
public:
	PhysicalIndirectOperand(const PhysicalRegister* preg,
		VirtualRegister* reg, int64_t offset, Instruction* i);

public:
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The offset to add to the register */
	int64_t offset;

};

}

}


