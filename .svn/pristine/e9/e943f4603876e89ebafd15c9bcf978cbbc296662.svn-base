/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>

// Stdandard Library Includes
#include <sstream>
#include <typeinfo>

namespace vanaheimr
{

/*! \brief A namespace for the compiler internal representation */
namespace ir
{

Instruction::Instruction(Opcode o, BasicBlock* b)
: opcode(o), guard(0), block(b)
{
	reads.push_back(guard);
}

Instruction::~Instruction()
{
	clear();
}

Instruction::Instruction(const Instruction& i)
: opcode(i.opcode), block(i.block)
{
	for(auto operand : i.reads)
	{
		if(operand != 0)
		{
			reads.push_back(operand->clone());
		}
		else
		{
			reads.push_back(0);
		}
	}
	
	guard = static_cast<PredicateOperand*>(reads[0]);
	
	for(auto operand : i.writes)
	{
		if(operand != 0)
		{
			writes.push_back(operand->clone());
		}
		else
		{
			writes.push_back(0);
		}
	}
}

Instruction& Instruction::operator=(const Instruction& i)
{
	if(this == &i) return *this;
	
	clear();
	
	opcode = i.opcode;
	block  = i.block;
	
	for(auto operand : i.reads)
	{
		if(operand != 0)
		{
			reads.push_back(operand->clone());
		}
		else
		{
			reads.push_back(0);
		}
	}
	
	guard = static_cast<PredicateOperand*>(reads[0]);
	
	for(auto operand : i.writes)
	{
		if(operand != 0)
		{
			writes.push_back(operand->clone());
		}
		else
		{
			writes.push_back(0);
		}
	}
	
	return *this;
}

void Instruction::setGuard(PredicateOperand* p)
{
	delete guard;
	
	reads[0] = p;
	guard    = p;
}

bool Instruction::isLoad() const
{
	return opcode == Ld  || opcode == Atom;
}

bool Instruction::isStore() const
{
	return opcode == St  || opcode == Atom;
}

bool Instruction::isBranch() const
{
	return opcode == Bra || opcode == Call;
}

bool Instruction::isCall() const
{
	return opcode == Call;
}

bool Instruction::isUnary() const
{
	return dynamic_cast<const UnaryInstruction*>(this) != nullptr;
}

bool Instruction::isBinary() const
{
	return dynamic_cast<const BinaryInstruction*>(this) != nullptr;
}

std::string Instruction::toString() const
{
	std::stringstream stream;
	
	if(!guard->isAlwaysTrue())
	{
		stream << guard->toString() << " ";
	}
	
	stream << toString(opcode) << " ";
	
	for(auto write : writes)
	{
		if(write != *writes.begin()) stream << ", ";
		
		stream << write->toString();
	}
	
	if(!writes.empty() && reads.size() > 1)
	{
		stream << ", ";
	}
	
	bool first = true;

	for(auto read : reads)
	{
		if(read == *reads.begin()) continue;
		
		if(!first)
		{
			 stream << ", ";
		}
		else
		{
			first = false;
		}
		
		stream << read->toString();
	}

	return stream.str();

}

void Instruction::clear()
{
	for(auto operand : reads)  delete operand;
	for(auto operand : writes) delete operand;
	
	reads.clear();
	writes.clear();
}

std::string Instruction::toString(Opcode o)
{
	switch(o)
	{
	case Add:     return "Add";
	case And:     return "And";
	case Ashr:    return "Ashr";
	case Atom:    return "Atom";
	case Bar:     return "Bar";
	case Bitcast: return "Bitcast";
	case Bra:     return "Bra";
	case Call:    return "Call";
	case Fdiv:    return "Fdiv";
	case Fmul:    return "Fmul";
	case Fpext:   return "Fpext";
	case Fptosi:  return "Fptosi";
	case Fptoui:  return "Fptoui";
	case Fptrunc: return "Fptrunc";
	case Frem:    return "Frem";
	case Launch:  return "Launch";
	case Ld:      return "Ld";
	case Lshr:    return "Lshr";
	case Membar:  return "Membar";
	case Mul:     return "Mul";
	case Or:      return "Or";
	case Ret:     return "Ret";
	case Setp:    return "Setp";
	case Sext:    return "Sext";
	case Sdiv:    return "Sdiv";
	case Shl:     return "Shl";
	case Sitofp:  return "Sitofp";
	case Srem:    return "Srem";
	case St:      return "St";
	case Sub:     return "Sub";
	case Trunc:   return "Trunc";
	case Udiv:    return "Udiv";
	case Uitofp:  return "Uitofp";
	case Urem:    return "Urem";
	case Xor:     return "Xor";
	case Zext:    return "Zext";
	default:      break;
	}
	
	return "InvalidOpcode";
}

UnaryInstruction::UnaryInstruction(Opcode o, BasicBlock* b)
: Instruction(o, b), d(0), a(0)
{
	writes.push_back(d);
	 reads.push_back(a);
}

UnaryInstruction::UnaryInstruction(const UnaryInstruction& i)
: Instruction(i), d(writes[0]), a(reads[1])
{
	
}

UnaryInstruction& UnaryInstruction::operator=(const UnaryInstruction& i)
{
	if(&i == this) return *this;
	
	Instruction::operator=(i);
	
	d = writes[0];
	a =  reads[1];
	
	return *this;
}

void UnaryInstruction::setD(Operand* o)
{
	delete d;
	
	d         = o;
	writes[0] = o;
}

void UnaryInstruction::setA(Operand* o)
{
	delete a;
	
	a        = o;
	reads[1] = o;
}

BinaryInstruction::BinaryInstruction(Opcode o, BasicBlock* bb)
: Instruction(o, bb), d(0), a(0), b(0)
{
	writes.push_back(d);
	 reads.push_back(a);
	 reads.push_back(b);
}

BinaryInstruction::BinaryInstruction(const BinaryInstruction& i)
: Instruction(i), d(writes.back()), a(reads[1]), b(reads[2])
{
	
}

BinaryInstruction& BinaryInstruction::operator=(const BinaryInstruction& i)
{
	if(&i == this) return *this;
	
	Instruction::operator=(i);
	
	d = writes[0];
	a =  reads[1];
	b =  reads[2];
	
	return *this;
}

void BinaryInstruction::setD(Operand* o)
{
	delete d;
	
	d         = o;
	writes[0] = o;
}

void BinaryInstruction::setA(Operand* o)
{
	delete a;
	
	a        = o;
	reads[1] = o;
}

void BinaryInstruction::setB(Operand* o)
{
	delete b;
	
	b        = o;
	reads[2] = o;
}

ComparisonInstruction::ComparisonInstruction(Opcode o,
	Comparison c, BasicBlock* b)
: BinaryInstruction(o, b), comparison(c)
{

}

Add::Add(BasicBlock* b)
: BinaryInstruction(Instruction::Add, b)
{

}

Instruction* Add::clone() const
{
	return new Add(*this);
}

/*! \brief An and instruction */
And::And(BasicBlock* b)
: BinaryInstruction(Instruction::And, b)
{

}

Instruction* And::clone() const
{
	return new And(*this);
}

/*! \brief Perform arithmetic shift right */
Ashr::Ashr(BasicBlock* b)
: BinaryInstruction(Instruction::Ashr, b)
{

}

Instruction* Ashr::clone() const
{
	return new Ashr(*this);
}

/*! \brief An atomic operation instruction */
Atom::Atom(Operation o, BasicBlock* b)
: BinaryInstruction(Instruction::Atom, b), operation(o), c(0)
{
	reads.push_back(c);
}

Atom::Atom(const Atom& i)
: BinaryInstruction(i)
{
	c = reads[3];
}

Atom& Atom::operator=(const Atom& i)
{
	if(&i == this) return *this;
	
	BinaryInstruction::operator=(i);
	
	c = reads[3];
	
	return *this;
}

Instruction* Atom::clone() const
{
	return new Atom(*this);
}

/*! \brief Perform a thread group barrier */
Bar::Bar(BasicBlock* b)
: Instruction(Instruction::Bar, b)
{

}

Instruction* Bar::clone() const
{
	return new Bar(*this);
}

/*! \brief Perform a raw bitcast */
Bitcast::Bitcast(BasicBlock* b)
: UnaryInstruction(Instruction::Bitcast, b)
{

}

Instruction* Bitcast::clone() const
{
	return new Bitcast(*this);
}

/*! \brief Perform a branch */
Bra::Bra(BranchModifier m, BasicBlock* b)
: Instruction(Instruction::Bra, b), target(0), modifier(m)
{
	reads.push_back(0);
}

Bra::Bra(const Bra& i)
: Instruction(i), modifier(i.modifier)
{
	target = reads[1];
}

Bra& Bra::operator=(const Bra& i)
{
	if(this == &i) return *this;
	
	Instruction::operator=(i);
	
	target   = reads[1];
	modifier = i.modifier;
	
	return *this;
}

void Bra::setTarget(Operand* o)
{
	delete target;

	target   = o;
	reads[1] = o;
}

Instruction* Bra::clone() const
{
	return new Bra(*this);
}

/*! \brief Branch and save the return pc */
Call::Call(BranchModifier m, BasicBlock* b)
: Bra(m, b), link(0)
{
	reads.push_back(0);
}

Call::Call(const Call& i)
: Bra(i), link(reads[2])
{

}

Call& Call::operator=(const Call& i)
{
	if(this == &i) return *this;
	
	Bra::operator=(i);
	
	link = reads[2];
	
	return *this;
}

Instruction* Call::clone() const
{
	return new Call(*this);
}

/*! \brief Floating point division */
Fdiv::Fdiv(BasicBlock* b)
: BinaryInstruction(Instruction::Fdiv, b)
{

}

Instruction* Fdiv::clone() const
{
	return new Fdiv(*this);
}

/*! \brief Floating point multiplication */
Fmul::Fmul(BasicBlock* b)
: BinaryInstruction(Instruction::Fmul, b)
{

}

Instruction* Fmul::clone() const
{
	return new Fmul(*this);
}

/*! \brief A floating point precision extension instruction */
Fpext::Fpext(BasicBlock* b)
: UnaryInstruction(Instruction::Fpext, b)
{

}

Instruction* Fpext::clone() const
{
	return new Fpext(*this);
}

/*! \brief A floating point to signed integer instruction */
Fptosi::Fptosi(BasicBlock* b)
: UnaryInstruction(Instruction::Fptosi, b)
{

}

Instruction* Fptosi::clone() const
{
	return new Fptosi(*this);
}

/*! \brief A floating point to unsigned integer instruction */
Fptoui::Fptoui(BasicBlock* b)
: UnaryInstruction(Instruction::Fptoui, b)
{

}

Instruction* Fptoui::clone() const
{
	return new Fptoui(*this);
}

/*! \brief A floating point precision truncate instruction */
Fptrunc::Fptrunc(BasicBlock* b)
: UnaryInstruction(Instruction::Fptrunc, b)
{

}

Instruction* Fptrunc::clone() const
{
	return new Fptrunc(*this);
}

/*! \brief Floating point remainder */
Frem::Frem(BasicBlock* b)
: BinaryInstruction(Instruction::Frem, b)
{

}

Instruction* Frem::clone() const
{
	return new Frem(*this);
}

/*! \brief Launch a new HTA at the specified entry point */
Launch::Launch(BasicBlock* b)
: Instruction(Instruction::Launch, b)
{

}

Instruction* Launch::clone() const
{
	return new Launch(*this);
}

/*! \brief Load a value from memory */
Ld::Ld(BasicBlock* b)
: UnaryInstruction(Instruction::Ld, b)
{

}

Instruction* Ld::clone() const
{
	return new Ld(*this);
}

/*! \brief Logical shift right */
Lshr::Lshr(BasicBlock* b)
: BinaryInstruction(Instruction::Lshr, b)
{

}

Instruction* Lshr::clone() const
{
	return new Lshr(*this);
}

/*! \brief Wait until memory operations at the specified level have completed */
Membar::Membar(Level l, BasicBlock* b)
: Instruction(Instruction::Membar, b), level(l)
{

}

Instruction* Membar::clone() const
{
	return new Membar(*this);
}

/*! \brief Multiply two operands together */
Mul::Mul(BasicBlock* b)
: BinaryInstruction(Instruction::Mul, b)
{

}

Instruction* Mul::clone() const
{
	return new Mul(*this);
}

/*! \brief Perform a logical OR operation */
Or::Or(BasicBlock* b)
: BinaryInstruction(Instruction::Or, b)
{

}

Instruction* Or::clone() const
{
	return new Or(*this);
}

/*! \brief Return from the current function call, or exit */
Ret::Ret(BasicBlock* b)
: Instruction(Instruction::Ret, b)
{

}

Instruction* Ret::clone() const
{
	return new Ret(*this);
}

/*! \brief Compare two operands and set a third predicate */
Setp::Setp(Comparison c, BasicBlock* b)
: ComparisonInstruction(Instruction::Setp, c, b)
{

}

Instruction* Setp::clone() const
{
	return new Setp(*this);
}

/*! \brief Sign extend an integer */
Sext::Sext(BasicBlock* b)
: UnaryInstruction(Instruction::Sext, b)
{

}

Instruction* Sext::clone() const
{
	return new Sext(*this);
}

/*! \brief Perform signed division */
Sdiv::Sdiv(BasicBlock* b)
: BinaryInstruction(Instruction::Sdiv, b)
{

}

Instruction* Sdiv::clone() const
{
	return new Sdiv(*this);
}

/*! \brief Perform shift left */
Shl::Shl(BasicBlock* b)
: BinaryInstruction(Instruction::Shl, b)
{
	
}

Instruction* Shl::clone() const
{
	return new Shl(*this);
}

/*! \brief Convert a signed int to a floating point */
Sitofp::Sitofp(BasicBlock* b)
: UnaryInstruction(Instruction::Sitofp, b)
{

}

Instruction* Sitofp::clone() const
{
	return new Sitofp(*this);
}

/*! \brief Perform a signed remainder operation */
Srem::Srem(BasicBlock* b)
: BinaryInstruction(Instruction::Srem, b)
{

}

Instruction* Srem::clone() const
{
	return new Srem(*this);
}

/*! \brief Perform a store operation */
St::St(BasicBlock* b)
: Instruction(Instruction::St, b)
{
	reads.push_back(0);
	reads.push_back(0);
	
	d = reads[1];
	a = reads[2];
}

St::St(const St& s)
: Instruction(s)
{
	d = reads[1];
	a = reads[2];
}

St& St::operator=(const St& s)
{
	if(&s == this) return *this;
	
	Instruction::operator=(s);
	
	d = reads[1];
	a = reads[2];
	
	return *this;
}

void St::setD(Operand* o)
{
	delete d;
	
	d        = o;
	reads[1] = d;
}

void St::setA(Operand* o)
{
	delete a;
	
	a        = o;
	reads[2] = o;
}

Instruction* St::clone() const
{
	return new St(*this);
}

/*! \brief Perform a subtract operation */
Sub::Sub(BasicBlock* b)
: BinaryInstruction(Instruction::Sub, b)
{

}

Instruction* Sub::clone() const
{
	return new Sub(*this);
}

/*! \brief Truncate an integer */
Trunc::Trunc(BasicBlock* b)
: UnaryInstruction(Instruction::Trunc, b)
{

}

Instruction* Trunc::clone() const
{
	return new Trunc(*this);
}

/*! \brief Perform an unsigned division operation */
Udiv::Udiv(BasicBlock* b)
: BinaryInstruction(Instruction::Udiv, b)
{

}

Instruction* Udiv::clone() const
{
	return new Udiv(*this);
}

/*! \brief Convert an unsigned int to a floating point */
Uitofp::Uitofp(BasicBlock* b)
: UnaryInstruction(Instruction::Uitofp, b)
{

}

Instruction* Uitofp::clone() const
{
	return new Uitofp(*this);
}

/*! \brief Perform an unsigned remainder operation */
Urem::Urem(BasicBlock* b)
: BinaryInstruction(Instruction::Urem, b)
{

}

Instruction* Urem::clone() const
{
	return new Urem(*this);
}

/*! \brief Perform a logical XOR operation */
Xor::Xor(BasicBlock* b)
: BinaryInstruction(Instruction::Xor, b)
{

}

Instruction* Xor::clone() const
{
	return new Xor(*this);
}

/*! \brief Zero extend an integer */
Zext::Zext(BasicBlock* b)
: UnaryInstruction(Instruction::Zext, b)
{

}

Instruction* Zext::clone() const
{
	return new Zext(*this);
}

}

}

