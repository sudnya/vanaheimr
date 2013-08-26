/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Stdandard Library Includes
#include <sstream>
#include <typeinfo>
#include <cassert>

namespace vanaheimr
{

/*! \brief A namespace for the compiler internal representation */
namespace ir
{

Instruction::Instruction(Opcode o, BasicBlock* b, Id id)
: opcode(o), block(b), _id(id), _metadata(nullptr)
{
	reads.push_back(nullptr); // for the guard
}

Instruction::~Instruction()
{
	clear();
}

Instruction::Instruction(const Instruction& i)
: opcode(i.opcode), block(i.block), _id(i.id())
{
	for(auto operand : i.reads)
	{
		if(operand != nullptr)
		{
			reads.push_back(operand->clone());
			
			reads.back()->instruction = this;
		}
		else
		{
			reads.push_back(nullptr);
		}
	}
	
	for(auto operand : i.writes)
	{
		if(operand != nullptr)
		{
			writes.push_back(operand->clone());
			
			writes.back()->instruction = this;
		}
		else
		{
			writes.push_back(nullptr);
		}
	}
}

Instruction& Instruction::operator=(const Instruction& i)
{
	if(this == &i) return *this;
	
	clear();
	
	opcode = i.opcode;
	block  = i.block;
	
	_id = i.id();
	
	for(auto operand : i.reads)
	{
		if(operand != nullptr)
		{
			reads.push_back(operand->clone());
			
			reads.back()->instruction = this;
		}
		else
		{
			reads.push_back(nullptr);
		}
	}
	
	for(auto operand : i.writes)
	{
		if(operand != nullptr)
		{
			writes.push_back(operand->clone());
			
			writes.back()->instruction = this;
		}
		else
		{
			writes.push_back(nullptr);
		}
	}
	
	return *this;
}

void Instruction::setGuard(PredicateOperand* p)
{
	delete guard();
	
	reads[0] = p;
}

Instruction::PredicateOperandPointer Instruction::guard()
{
	assert(!reads.empty());
	
	return static_cast<PredicateOperandPointer>(reads[0]);
}

const Instruction::PredicateOperandPointer Instruction::guard() const
{
	assert(!reads.empty());
	
	return static_cast<const PredicateOperandPointer>(reads[0]);
}

Instruction::Id Instruction::id() const
{
	return _id;
}

unsigned int Instruction::index() const
{
	unsigned int index = 0;
	
	for(auto instruction : *block)
	{
		if(instruction == this)
		{
			return index;
		}
		++index;
	}
	
	assertM(false, "Could not find instruction in parent block.");
	
	return index;
}

void Instruction::appendWrite(Operand* newOperand)
{
	newOperand->instruction = this;

	writes.push_back(newOperand);
}

void Instruction::appendRead(Operand* newOperand)
{
	newOperand->instruction = this;

	reads.push_back(newOperand);
}

void Instruction::replaceOperand(Operand* original, Operand* newOperand)
{
	assert(original->instruction == this);

	for(auto read = reads.begin(); read != reads.end(); ++read)
	{
		if(*read == original)
		{
			delete *read;
			*read = newOperand;
			return;
		}
	}

	for(auto write = writes.begin(); write != writes.end(); ++write)
	{
		if(*write == original)
		{
			delete *write;
			*write = newOperand;
			return;
		}
	}
	
	assertM(false, "Operand " << original->toString()
		<< " not found in owning instruction " << toString());
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

bool Instruction::isReturn() const
{
	return opcode == Ret;
}

bool Instruction::isIntrinsic() const
{
	if(opcode != Call) return false;
	
	auto call = dynamic_cast<const ir::Call*>(this);
	
	assert(call != nullptr);
	
	return call->isIntrinsic();
}

bool Instruction::isPhi() const
{
	return opcode == Phi;
}

bool Instruction::isPsi() const
{
	return opcode == Psi;
}

bool Instruction::isMachineInstruction() const
{
	return opcode == Machine;
}

bool Instruction::accessesMemory() const
{
	return isLoad() || isStore();
}

bool Instruction::isMemoryBarrier() const
{
	return opcode == Membar || opcode == Bar;
}

bool Instruction::isUnary() const
{
	return dynamic_cast<const UnaryInstruction*>(this) != nullptr;
}

bool Instruction::isBinary() const
{
	return dynamic_cast<const BinaryInstruction*>(this) != nullptr;
}

bool Instruction::isComparison() const
{
	return dynamic_cast<const ComparisonInstruction*>(this) != nullptr;
}

std::string Instruction::toString() const
{
	std::stringstream stream;
	
	if(!guard()->isAlwaysTrue())
	{
		stream << guard()->toString() << " ";
	}
	
	stream << opcodeString() << " ";
	
	std::string modifier = modifierString();
	
	if(!modifier.empty())
	{
		stream << modifier << " ";
	}
	
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
		if(read == guard()) continue;
		
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

std::string Instruction::modifierString() const
{
	return "";
}

std::string Instruction::opcodeString() const
{
	return toString(opcode);
}

void Instruction::eraseFromBlock()
{
	block->erase(this);
}

void Instruction::clear()
{
	for(auto operand : reads)  delete operand;
	for(auto operand : writes) delete operand;
	
	 reads.clear();
	writes.clear();
}	

Instruction::Opcode Instruction::parseOpcode(const std::string& opcode)
{
	if(opcode == "add")           return Add;
	if(opcode == "and")           return And;
	if(opcode == "ashr")          return Ashr;
	if(opcode == "atom")          return Atom;
	if(opcode == "bar")           return Bar;
	if(opcode == "bitcast")       return Bitcast;
	if(opcode == "bra")           return Bra;
	if(opcode == "call")          return Call;
	if(opcode == "fdiv")          return Fdiv;
	if(opcode == "fmul")          return Fmul;
	if(opcode == "fpext")         return Fpext;
	if(opcode == "fptosi")        return Fptosi;
	if(opcode == "fptoui")        return Fptoui;
	if(opcode == "fptrunc")       return Fptrunc;
	if(opcode == "frem")          return Frem;
	if(opcode == "getelementptr") return Getelementptr;
	if(opcode == "launch")        return Launch;
	if(opcode == "ld")            return Ld;
	if(opcode == "lshr")          return Lshr;
	if(opcode == "membar")        return Membar;
	if(opcode == "mul")           return Mul;
	if(opcode == "or")            return Or;
	if(opcode == "ret")           return Ret;
	if(opcode == "setp")          return Setp;
	if(opcode == "sext")          return Sext;
	if(opcode == "sdiv")          return Sdiv;
	if(opcode == "shl")           return Shl;
	if(opcode == "sitofp")        return Sitofp;
	if(opcode == "srem")          return Srem;
	if(opcode == "st")            return St;
	if(opcode == "sub")           return Sub;
	if(opcode == "trunc")         return Trunc;
	if(opcode == "udiv")          return Udiv;
	if(opcode == "uitofp")        return Uitofp;
	if(opcode == "urem")          return Urem;
	if(opcode == "xor")           return Xor;
	if(opcode == "zext")          return Zext;
	if(opcode == "phi")           return Phi;
	if(opcode == "psi")           return Psi;
	if(opcode == "machine")       return Machine;
	
	return InvalidOpcode;
}

std::string Instruction::toString(Opcode o)
{
	switch(o)
	{
	case Add:           return "add";
	case And:           return "and";
	case Ashr:          return "ashr";
	case Atom:          return "atom";
	case Bar:           return "bar";
	case Bitcast:       return "bitcast";
	case Bra:           return "bra";
	case Call:          return "call";
	case Fdiv:          return "fdiv";
	case Fmul:          return "fmul";
	case Fpext:         return "fpext";
	case Fptosi:        return "fptosi";
	case Fptoui:        return "fptoui";
	case Fptrunc:       return "fptrunc";
	case Frem:          return "frem";
	case Getelementptr: return "getelementptr";
	case Launch:        return "launch";
	case Ld:            return "ld";
	case Lshr:          return "lshr";
	case Membar:        return "membar";
	case Mul:           return "mul";
	case Or:            return "or";
	case Ret:           return "ret";
	case Setp:          return "setp";
	case Sext:          return "sext";
	case Sdiv:          return "sdiv";
	case Shl:           return "shl";
	case Sitofp:        return "sitofp";
	case Srem:          return "srem";
	case St:            return "st";
	case Sub:           return "sub";
	case Trunc:         return "trunc";
	case Udiv:          return "udiv";
	case Uitofp:        return "uitofp";
	case Urem:          return "urem";
	case Xor:           return "xor";
	case Zext:          return "zext";
	case Phi:           return "phi";
	case Psi:           return "psi";
	case Machine:       return "MACHINE-SPECIFIC";
	case InvalidOpcode: break;
	}
	
	return "InvalidOpcode";
}

Instruction* Instruction::create(Opcode o, BasicBlock* b)
{
	ir::Instruction* instruction = nullptr;

	switch(o)
	{
	case Add:           instruction = new ir::Add;           break;
	case And:           instruction = new ir::And;           break;
	case Ashr:          instruction = new ir::Ashr;          break;
	case Atom:          instruction = new ir::Atom;          break;
	case Bar:           instruction = new ir::Bar;           break;
	case Bitcast:       instruction = new ir::Bitcast;       break;
	case Bra:           instruction = new ir::Bra;           break;
	case Call:          instruction = new ir::Call;          break;
	case Fdiv:          instruction = new ir::Fdiv;          break;
	case Fmul:          instruction = new ir::Fmul;          break;
	case Fpext:         instruction = new ir::Fpext;         break;
	case Fptosi:        instruction = new ir::Fptosi;        break;
	case Fptoui:        instruction = new ir::Fptoui;        break;
	case Fptrunc:       instruction = new ir::Fptrunc;       break;
	case Frem:          instruction = new ir::Frem;          break;
	case Getelementptr: instruction = new ir::Getelementptr; break;
	case Launch:        instruction = new ir::Launch;        break;
	case Ld:            instruction = new ir::Ld;            break;
	case Lshr:          instruction = new ir::Lshr;          break;
	case Membar:        instruction = new ir::Membar;        break;
	case Mul:           instruction = new ir::Mul;           break;
	case Or:            instruction = new ir::Or;            break;
	case Ret:           instruction = new ir::Ret;           break;
	case Setp:          instruction = new ir::Setp;          break;
	case Sext:          instruction = new ir::Sext;          break;
	case Sdiv:          instruction = new ir::Sdiv;          break;
	case Shl:           instruction = new ir::Shl;           break;
	case Sitofp:        instruction = new ir::Sitofp;        break;
	case Srem:          instruction = new ir::Srem;          break;
	case St:            instruction = new ir::St;            break;
	case Sub:           instruction = new ir::Sub;           break;
	case Trunc:         instruction = new ir::Trunc;         break;
	case Udiv:          instruction = new ir::Udiv;          break;
	case Uitofp:        instruction = new ir::Uitofp;        break;
	case Urem:          instruction = new ir::Urem;          break;
	case Xor:           instruction = new ir::Xor;           break;
	case Zext:          instruction = new ir::Zext;          break;
	case Phi:           instruction = new ir::Phi;           break;
	case Psi:           instruction = new ir::Psi;           break;
	case Machine:       break;
	case InvalidOpcode: break;
	}
	
	assert(instruction != nullptr);
	
	instruction->block = b;

	return instruction;
}

Instruction* Instruction::create(const std::string& opcode, BasicBlock* b)
{
	return create(parseOpcode(opcode), b);
}

UnaryInstruction::UnaryInstruction(Opcode o, BasicBlock* b)
: Instruction(o, b)
{
	writes.push_back(nullptr); // d
	 reads.push_back(nullptr); // a
}

void UnaryInstruction::setD(Operand* o)
{
	delete d();
	
	d() = o;
}

void UnaryInstruction::setA(Operand* o)
{
	delete a();
	
	a() = o;
}

Instruction::OperandPointer& UnaryInstruction::d()
{
	assert(writes.size() > 0);
	
	return writes[0];
}

const Instruction::OperandPointer& UnaryInstruction::d() const
{
	assert(writes.size() > 0);
	
	return writes[0];
}

Instruction::OperandPointer& UnaryInstruction::a()
{
	assert(reads.size() > 1);
	
	return reads[1];
}

const Instruction::OperandPointer& UnaryInstruction::a() const
{
	assert(reads.size() > 1);
	
	return reads[1];
}

BinaryInstruction::BinaryInstruction(Opcode o, BasicBlock* bb)
: Instruction(o, bb)
{
	writes.push_back(nullptr); // d
	 reads.push_back(nullptr); // a 
	 reads.push_back(nullptr); // b
}

void BinaryInstruction::setD(Operand* o)
{
	delete d();
	
	d() = o;
}

void BinaryInstruction::setA(Operand* o)
{
	delete a();
	
	a() = o;
}

void BinaryInstruction::setB(Operand* o)
{
	delete b();
	
	b() = o;
}	

Instruction::OperandPointer& BinaryInstruction::d()
{
	assert(writes.size() > 0);

	return writes[0];
}

const Instruction::OperandPointer& BinaryInstruction::d() const
{
	assert(writes.size() > 0);

	return writes[0];
}

Instruction::OperandPointer& BinaryInstruction::a()
{
	assert(reads.size() > 1);

	return reads[1];
}

const Instruction::OperandPointer& BinaryInstruction::a() const
{
	assert(reads.size() > 1);

	return reads[1];
}

Instruction::OperandPointer& BinaryInstruction::b()
{
	assert(reads.size() > 2);

	return reads[2];
}

const Instruction::OperandPointer& BinaryInstruction::b() const
{
	assert(reads.size() > 2);

	return reads[2];
}

ComparisonInstruction::ComparisonInstruction(Opcode o,
	Comparison c, BasicBlock* b)
: BinaryInstruction(o, b), comparison(c)
{

}

std::string ComparisonInstruction::toString() const
{
	return Instruction::toString();
}

std::string ComparisonInstruction::modifierString() const
{
	return toString(comparison);
}

std::string ComparisonInstruction::toString(Comparison c)
{
	switch(c)
	{
	case OrderedEqual:            return "eq";
	case OrderedNotEqual:         return "ne";
	case OrderedLessThan:         return "lt";
	case OrderedLessOrEqual:      return "le";
	case OrderedGreaterThan:      return "gt";
	case OrderedGreaterOrEqual:   return "ge";
	case UnorderedEqual:          return "equ";
	case UnorderedNotEqual:       return "neu";
	case UnorderedLessThan:       return "ltu";
	case UnorderedLessOrEqual:    return "leu";
	case UnorderedGreaterThan:    return "gtu";
	case UnorderedGreaterOrEqual: return "geu";
	case IsANumber:               return "num";
	case NotANumber:              return "nan";
	case InvalidComparison:       break;
	}

	return "InvalidComparison";
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
: BinaryInstruction(Instruction::Atom, b), operation(o)
{
	reads.push_back(nullptr); // c
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
: Instruction(Instruction::Bra, b), modifier(m)
{
	reads.push_back(nullptr); // target
}

void Bra::setTarget(Operand* o)
{
	delete target();

	target() = o;
}

Instruction::OperandPointer& Bra::target()
{
	assert(reads.size() > 1);
	
	return reads[1];
}

const Instruction::OperandPointer& Bra::target() const
{
	assert(reads.size() > 1);
	
	return reads[1];
}

BasicBlock* Bra::targetBasicBlock()
{
	assert(target() != nullptr);
	assert(target()->isBasicBlock());
	
	auto block = static_cast<AddressOperand*>(target());
	
	return static_cast<BasicBlock*>(block->globalValue);
}

const BasicBlock* Bra::targetBasicBlock() const
{
	assert(target() != nullptr);
	assert(target()->isBasicBlock());
	
	auto block = static_cast<const AddressOperand*>(target());
	
	return static_cast<const BasicBlock*>(block->globalValue);
}

bool Bra::isUnconditional() const
{
	return guard()->isAlwaysTrue();
}

Instruction* Bra::clone() const
{
	return new Bra(*this);
}

/*! \brief Branch and save the return pc */
Call::Call(BasicBlock* b)
: Instruction(Instruction::Call, b)
{
	reads.push_back(nullptr); // target
}

bool Call::isIntrinsic() const
{
	if(!target()->isAddress()) return false;
	
	auto addressOperand = static_cast<AddressOperand*>(target());
	
	if(!addressOperand->globalValue->type().isFunction()) return false;
	
	return addressOperand->globalValue->name().find("_Zintrinsic_") == 0;
}

void Call::setTarget(Operand* o)
{
	delete target();

	target() = o;	
}

void Call::addReturn(Operand* o)
{
	writes.push_back(o);
}

void Call::addArgument(Operand* o)
{
	reads.push_back(o);
}

Instruction::OperandPointer& Call::target()
{
	assert(reads.size() > 1);

	return reads[1];
}

const Instruction::OperandPointer& Call::target() const
{
	assert(reads.size() > 1);

	return reads[1];
}

Call::OperandVector Call::returned()
{
	return writes;
}

Call::ConstOperandVector Call::returned() const
{
	ConstOperandVector operands;
	
	for(auto write : writes)
	{
		operands.push_back(write);
	}
	
	return operands;
}

Call::OperandVector Call::arguments()
{
	assert(reads.size() > 1);
	
	OperandVector operands;
	
	auto read = reads.begin(); ++read; ++read;
	for(; read != reads.end(); ++read)
	{
		operands.push_back(*read);
	}
	
	return operands;
}

Call::ConstOperandVector Call::arguments() const
{
	assert(reads.size() > 1);
	
	ConstOperandVector operands;
	
	auto read = reads.begin(); ++read; ++read;
	for(; read != reads.end(); ++read)
	{
		operands.push_back(*read);
	}
	
	return operands;
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

/*! \brief Get a pointer to a subtype of an aggregate */

Getelementptr::Getelementptr(BasicBlock* b)
: UnaryInstruction(Instruction::Getelementptr, b)
{

}

Instruction* Getelementptr::clone() const
{
	return new Getelementptr(*this);
}

const Type* Getelementptr::getSelectedType() const
{
	const Type* selected = a()->type();

	for(auto read = reads.begin() + 2; read != reads.end(); ++read)
	{
		assert(selected->isAggregate());
		
		auto aggregate = static_cast<const AggregateType*>(selected);
		
		if((*read)->isImmediate())
		{
			auto immediate = static_cast<const ImmediateOperand*>(*read);
		
			selected = aggregate->getTypeAtIndex(immediate->uint);
			
			continue;
		}
		
		assert((*read)->isRegister());
		
		selected = aggregate->getTypeAtIndex(0);
	}

	return selected;
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
	reads.push_back(nullptr); // d
	reads.push_back(nullptr); // a
}

void St::setD(Operand* o)
{
	delete d();
	
	d() = o;
}

void St::setA(Operand* o)
{
	delete a();
	
	a() = o;
}

St::OperandPointer& St::d()
{
	assert(reads.size() > 1);

	return reads[1];
}

const St::OperandPointer& St::d() const
{
	assert(reads.size() > 1);

	return reads[1];
}

St::OperandPointer& St::a()
{
	assert(reads.size() > 2);

	return reads[2];
}

const St::OperandPointer& St::a() const
{
	assert(reads.size() > 2);

	return reads[2];
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

/*! \brief Phi join node */
Phi::Phi(BasicBlock* b)
: Instruction(Instruction::Phi, b)
{
	writes.push_back(nullptr);
}

Phi::Phi(const Phi& i)
: Instruction(i)
{

}

Phi& Phi::operator=(const Phi& i)
{
	if(&i == this) return *this;

	Instruction::operator=(i);
	
	return *this;
}

void Phi::setD(RegisterOperand* o)
{
	delete d();
	
	writes[0] = o;
}

void Phi::addSource(RegisterOperand* source, AddressOperand* predecessor)
{
	 reads.push_back(source);
	 reads.push_back(predecessor);
}

void Phi::removeSource(BasicBlock* predecessor)
{
	auto readPosition = reads.begin();
	for(++readPosition; readPosition != reads.end(); ++readPosition)
	{
		++readPosition;
		assert(readPosition != reads.end());
		
		auto operand = static_cast<AddressOperand*>(*readPosition);
		
		if(operand->globalValue != predecessor) continue;
		
		--readPosition;
		delete *readPosition;
		readPosition = reads.erase(readPosition);
		delete *readPosition;
		reads.erase(readPosition);
		
		break;
	}
	
	assertM(false, "Phi instruction " << toString()
		<< " does not contain basic block " << predecessor->name());
}

Phi::RegisterOperandPointer Phi::d()
{
	assert(writes.size() > 0);
	
	return static_cast<RegisterOperandPointer>(writes[0]);
}

const Phi::RegisterOperandPointer Phi::d() const
{
	assert(writes.size() > 0);
	
	return static_cast<RegisterOperandPointer>(writes[0]);
}

Phi::RegisterOperandVector Phi::sources()
{
	assert(reads.size() > 0);

	RegisterOperandVector sourceOperands;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		sourceOperands.push_back(static_cast<RegisterOperandPointer>(*read));
		++read;
		assert(read != reads.end());
	}
	
	return sourceOperands;
}

Phi::ConstRegisterOperandVector Phi::sources() const
{
	assert(reads.size() > 0);

	ConstRegisterOperandVector sourceOperands;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		sourceOperands.push_back(
			static_cast<const RegisterOperandPointer>(*read));
		++read;
		assert(read != reads.end());
	}
	
	return sourceOperands;
}

Phi::BasicBlockVector Phi::blocks()
{
	BasicBlockVector blockVector;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		++read;
		assert(read != reads.end());
		
		auto operand = static_cast<AddressOperand*>(*read);
		
		blockVector.push_back(static_cast<BasicBlock*>(operand->globalValue));
	}
	
	return blockVector;
}

Phi::ConstBasicBlockVector Phi::blocks() const
{
	ConstBasicBlockVector blockVector;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		++read;
		assert(read != reads.end());
		
		auto operand = static_cast<const AddressOperand*>(*read);
		
		blockVector.push_back(static_cast<const BasicBlock*>(
			operand->globalValue));
	}
	
	return blockVector;
}

Phi::AddressOperandVector Phi::blockOperands()
{
	AddressOperandVector operands;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		++read;
		assert(read != reads.end());
		
		operands.push_back(static_cast<AddressOperand*>(*read));
	}
	
	return operands;
}

Phi::ConstAddressOperandVector Phi::blockOperands() const
{
	ConstAddressOperandVector operands;
	
	auto read = reads.begin(); ++read;
	
	for(; read != reads.end(); ++read)
	{
		++read;
		assert(read != reads.end());
		
		operands.push_back(static_cast<const AddressOperand*>(*read));
	}
	
	return operands;
}

Instruction* Phi::clone() const
{
	return new Phi(*this);
}

Psi::Psi(BasicBlock* b)
: Instruction(Instruction::Psi, b)
{
	writes.push_back(nullptr);
}

void Psi::setD(RegisterOperand* o)
{
	delete d();
	
	writes[0] = o;
}

void Psi::addSource(PredicateOperand* predicate, RegisterOperand* source)
{
	reads.push_back(predicate);
	reads.push_back(   source);
}

void Psi::removeSource(PredicateOperand* predicate)
{
	auto readPosition = reads.begin(); ++readPosition;

	for(; readPosition != reads.end(); ++readPosition)
	{
		if(*readPosition != predicate) continue;
		
		auto next = readPosition; ++next;
		
		assert(next != reads.end());
		
		reads.erase(readPosition);
		reads.erase(next);

		return;
	}
}

Psi::RegisterOperandPointer Psi::d()
{
	assert(writes.size() > 0);
	
	return static_cast<RegisterOperandPointer>(writes[0]);
}

const Psi::RegisterOperandPointer Psi::d() const
{
	assert(writes.size() > 0);
	
	return static_cast<RegisterOperandPointer>(writes[0]);
}

Psi::RegisterOperandVector Psi::sources()
{
	assert(reads.size() > 0);

	RegisterOperandVector sourceOperands;
	
	// skip the guard
	auto read = reads.begin(); ++read;

	for(; read != reads.end(); ++read)
	{
		++read;

		assert(read != reads.end());
		
		sourceOperands.push_back(static_cast<RegisterOperandPointer>(*read));
	}
	
	return sourceOperands;
}

Psi::PredicateOperandVector Psi::predicates()
{
	assert(reads.size() > 0);

	PredicateOperandVector sourceOperands;
	
	// skip the guard
	auto read = reads.begin(); ++read;

	for(; read != reads.end(); ++read)
	{
		sourceOperands.push_back(static_cast<PredicateOperandPointer>(*read));
		
		++read;

		assert(read != reads.end());
	}
	
	return sourceOperands;
}

Instruction* Psi::clone() const
{
	return new Psi(*this);
}

}

}

