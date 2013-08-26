/*	\file   PhysicalIndirectOperand.cpp
	\date   Monday June 25, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the PhysicalIndirectOperand class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/PhysicalIndirectOperand.h>

#include <vanaheimr/machine/interface/PhysicalRegister.h>

#include <vanaheimr/ir/interface/VirtualRegister.h>

// Standard Library Includes
#include <sstream>

namespace vanaheimr
{

namespace machine
{

PhysicalIndirectOperand::PhysicalIndirectOperand(const PhysicalRegister* p,
	VirtualRegister* r, int64_t o, Instruction* i)
: PhysicalRegisterOperand(p, r, i, Indirect), offset(o)
{

}

PhysicalIndirectOperand::Operand* PhysicalIndirectOperand::clone() const
{
	return new PhysicalIndirectOperand(*this);
}

std::string PhysicalIndirectOperand::toString() const
{
	std::stringstream stream;
		
	stream << "[ ";
	
	stream << PhysicalRegisterOperand::toString();

	stream << " + " << std::hex << offset << std::dec << " ]";

	return stream.str();
}

}

}


