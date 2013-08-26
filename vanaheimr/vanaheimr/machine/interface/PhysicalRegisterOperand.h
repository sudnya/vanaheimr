/*	\file   PhysicalRegisterOperand.h
	\date   Monday June 25, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the PhysicalRegisterOperand class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Operand.h>

// Forward Declarations
namespace vanaheimr { namespace machine { class PhysicalRegister; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief This class represents a register operand after assignment to
	a physical register */
class PhysicalRegisterOperand : public vanaheimr::ir::RegisterOperand
{
public:
	typedef vanaheimr::ir::VirtualRegister VirtualRegister;
	typedef vanaheimr::ir::Instruction     Instruction;
	typedef vanaheimr::ir::Operand         Operand;

public:
	PhysicalRegisterOperand(const PhysicalRegister* preg,
		VirtualRegister* reg, Instruction* i);

public:
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The physical register being accessed */
	const PhysicalRegister* physicalRegister;

protected:
	PhysicalRegisterOperand(const PhysicalRegister* preg, VirtualRegister* reg,
		Instruction* i, OperandMode m);

};

}

}


