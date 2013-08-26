/*	\file   Operand.cpp
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Operand class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Operand.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>
#include <vanaheimr/ir/interface/Variable.h>
#include <vanaheimr/ir/interface/Type.h>
#include <vanaheimr/ir/interface/Argument.h>
#include <vanaheimr/ir/interface/Constant.h>

// Standard Library Includes
#include <sstream>

namespace vanaheimr
{

namespace ir
{

Operand::Operand(OperandMode mode, Instruction* i)
: instruction(i), _mode(mode)
{

}

Operand::~Operand()
{
	
}

bool Operand::isRegister() const
{
	if(mode() == Register || mode() == Indirect) 
	{
		return true;
	}

	if(mode() == Predicate)
	{
		auto pred = static_cast<const PredicateOperand*>(this);

		return pred->modifier == PredicateOperand::InversePredicate ||
			pred->modifier == PredicateOperand::StraightPredicate;
	}

	return false;
}

bool Operand::isAddress() const
{
	return mode() == Address;
}

bool Operand::isArgument() const
{
	return mode() == Argument;
}

bool Operand::isImmediate() const
{
	return mode() == Immediate;
}

bool Operand::isIndirect() const
{
	return mode() == Indirect;
}

bool Operand::isBasicBlock() const
{
	if(mode() != Address) return false;

	const AddressOperand* address = static_cast<const AddressOperand*>(this);

	return address->globalValue->type().isBasicBlock();
}

Operand::OperandMode Operand::mode() const
{
	return _mode;
}

bool Operand::isConstant() const
{
	return false;
}

Constant* Operand::getConstantValue() const
{
	return nullptr;
}

RegisterOperand::RegisterOperand(VirtualRegister* reg, Instruction* i)
: Operand(Register, i), virtualRegister(reg)
{
	
}

const Type* RegisterOperand::type() const
{
	return virtualRegister->type;
}

Operand* RegisterOperand::clone() const
{
	return new RegisterOperand(*this);
}

std::string RegisterOperand::toString() const
{
	return virtualRegister->toString();
}

RegisterOperand::RegisterOperand(VirtualRegister* reg, Instruction* i,
	OperandMode m)
: Operand(m, i), virtualRegister(reg)
{

}

ImmediateOperand::ImmediateOperand(uint64_t v, Instruction* i, const Type* t)
: Operand(Immediate, i), dataType(t)
{
	uint = v;
}

ImmediateOperand::ImmediateOperand(double d, Instruction* i, const Type* t)
: Operand(Immediate, i), dataType(t)
{
	fp = d;
}

const Type* ImmediateOperand::type() const
{
	return dataType;
}

Operand* ImmediateOperand::clone() const
{
	return new ImmediateOperand(*this);
}

std::string ImmediateOperand::toString() const
{
	std::stringstream stream;

	stream << type()->name << " ";
		
	stream << "0x" << std::hex << uint << std::dec;

	return stream.str();
}

bool ImmediateOperand::isConstant() const
{
	return true;
}

Constant* ImmediateOperand::getConstantValue() const
{
	if(dataType->isInteger())
	{
		return new IntegerConstant(uint, dataType->bytes() * 8);
	}
	
	if(dataType->isSinglePrecisionFloat())
	{
		return new FloatingPointConstant((float)fp);
	}
	
	// Assume double
	return new FloatingPointConstant((double)fp);
}

PredicateOperand::PredicateOperand(VirtualRegister* reg,
	PredicateModifier mod, Instruction* i)
: RegisterOperand(reg, i, Predicate), modifier(mod)
{

}

PredicateOperand::PredicateOperand(PredicateModifier mod, Instruction* i)
: RegisterOperand(nullptr, i, Predicate), modifier(mod)
{

}

bool PredicateOperand::isAlwaysTrue() const
{
	return modifier == PredicateTrue;
}

Operand* PredicateOperand::clone() const
{
	return new PredicateOperand(*this);
}

std::string PredicateOperand::toString() const
{
	std::stringstream stream;

	switch(modifier)
	{
	case ir::PredicateOperand::InversePredicate:
	{
		stream << "!";
		
		// fall through
	}
	case ir::PredicateOperand::StraightPredicate:
	{
		stream << "@";
		
		stream << virtualRegister->toString();

		break;
	}
	case ir::PredicateOperand::PredicateTrue:
	{
		stream << "@pt";
		break;
	}
	case ir::PredicateOperand::PredicateFalse:
	{
		stream << "!@pt";
		break;
	}
	}

	return stream.str();
}

IndirectOperand::IndirectOperand(VirtualRegister* reg, int64_t o,
	Instruction* i)
: RegisterOperand(reg, i, Indirect), offset(o)
{

}

Operand* IndirectOperand::clone() const
{
	return new IndirectOperand(*this);
}

std::string IndirectOperand::toString() const
{
	std::stringstream stream;
		
	stream << "[ ";
	
	stream << virtualRegister->toString();

	stream << " + " << std::hex << offset << std::dec << " ]";

	return stream.str();
}

AddressOperand::AddressOperand(Variable* value, Instruction* i)
: Operand(Address, i), globalValue(value)
{
	
}

const Type* AddressOperand::type() const
{
	return &globalValue->type();
}

Operand* AddressOperand::clone() const
{
	return new AddressOperand(*this);
}

std::string AddressOperand::toString() const
{
	if(globalValue == nullptr) return "nullptr";

	std::stringstream stream;
		
	if(!globalValue->type().isBasicBlock())
	{	
		stream << globalValue->type().name;
		
		stream << " ";
	}
		
	stream << globalValue->name();

	return stream.str();
}

ArgumentOperand::ArgumentOperand(ir::Argument* a, Instruction* i)
: Operand(Argument, i), argument(a)
{
	
}

const Type* ArgumentOperand::type() const
{
	return &argument->type();
}

Operand* ArgumentOperand::clone() const
{
	return new ArgumentOperand(*this);
}

std::string ArgumentOperand::toString() const
{
	std::stringstream stream;

	stream << argument->type().name;
	
	stream << " ";

	stream << argument->name();
	
	return stream.str();
}

}

}

