/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Operand.h>

// Standard Library Includes 
#include <vector>
#include <string>

// Forward Declaration
namespace vanaheimr { namespace ir { class BasicBlock; } }
namespace vanaheimr { namespace ir { class MetaData;   } }

namespace vanaheimr
{

/*! \brief A namespace for the compiler internal representation */
namespace ir
{

/*! \brief A programmer efficient class for representing a single instruction */
class Instruction
{
public:
	/*! \brief The set of possible instructions */
	enum Opcode
	{
		Add,
		And,
		Ashr,
		Atom,
		Bar,
		Bitcast,
		Bra,
		Call,
		Fdiv,
		Fmul,
		Fpext,
		Fptosi,
		Fptoui,
		Fptrunc,
		Frem,
		Getelementptr,
		Launch,
		Ld,
		Lshr,
		Membar,
		Mul,
		Or,
		Ret,
		Setp,
		Sext,
		Sdiv,
		Shl,
		Sitofp,
		Srem,
		St,
		Sub,
		Trunc,
		Udiv,
		Uitofp,
		Urem,
		Xor,
		Zext,
		
		// IR Analysis
		Phi,
		Psi,
		
		// Machine
		Machine,
		
		// Invalid
		InvalidOpcode
	};

	typedef Operand* OperandPointer;
	typedef std::vector<OperandPointer> OperandVector;
	typedef PredicateOperand* PredicateOperandPointer;
	
	typedef unsigned int Id;

public:
	Instruction(Opcode = InvalidOpcode, BasicBlock* b = nullptr, Id id = 0);
	virtual ~Instruction();

	Instruction(const Instruction&);
	Instruction& operator=(const Instruction&);

public:
	/*! \brief Sets the predicate guard, the instruction now owns it */
	void setGuard(PredicateOperand* g);

	/*! \brief Get the guard predicate */
	      PredicateOperandPointer guard();
	const PredicateOperandPointer guard() const;

public:
	/*! \brief The instruction Id, unique within the function */
	Id id() const;

public:
	/*! \brief The index of the instruction within the basic block */
	unsigned int index() const;

public:
	/*! \brief Append an operand to the set of writes, it is now owned
		by the instruction */
	void appendWrite(Operand* newOperand);

	/*! \brief Append an operand to the set of reads, it is now owned
		by the instruction */
	void appendRead(Operand* newOperand);

	/*! \brief Replace a matching operand with another,
	     the original operand is deleted,
	     the new operand is now owned by the instruction */
	void replaceOperand(Operand* original, Operand* newOperand);

public:
	virtual bool isLoad()      const;
	virtual bool isStore()     const;
	virtual bool isBranch()    const;
	virtual bool isCall()      const;
	virtual bool isReturn()    const;
	virtual bool isIntrinsic() const;
	
	virtual bool accessesMemory()  const;
	virtual bool isMemoryBarrier() const;

public:	
	bool isPhi()       const;
	bool isPsi()       const;

	bool isMachineInstruction() const;

public:
	bool isUnary()      const;
	bool isBinary()     const;
	bool isComparison() const;

public:
	virtual std::string toString() const;
	virtual std::string modifierString() const;
	virtual std::string opcodeString() const;
	
public:
	/*! \brief Erase the instruction from the parent block */
	void eraseFromBlock();
	/*! \brief Clear the reader/writer sets. */
	void clear();

public:
	virtual Instruction* clone() const = 0;

public:
	static std::string toString(Opcode o);
	static Opcode parseOpcode(const std::string& opcode);
	static Instruction* create(Opcode o, BasicBlock* b);
	static Instruction* create(const std::string& opcode, BasicBlock* b);
	
public:
	/*! \brief The instruction opcode */
	Opcode opcode;

public:
	/*! \brief The list of all operands read by the instruction */
	OperandVector reads;
	/*! \brief The list of all operands written by the instruction */
	OperandVector writes;

public:
	/*!  \brief The basic block that the instruction is contained in */
	BasicBlock* block;
	
private:
	Id        _id;
	MetaData* _metadata;

};

/*! \brief A unary instruction */
class UnaryInstruction : public Instruction
{
public:
	explicit UnaryInstruction(Opcode = InvalidOpcode, BasicBlock* b = nullptr);

public:
	/*! \brief Set the destination, the instruction takes ownership */
	void setD(Operand* d);
	/*! \brief Set the source, the instruction takes ownership */
	void setA(Operand* a);

public:
	virtual Instruction* clone() const = 0;

public:
	/*! \brief The destination operand. */
	      OperandPointer& d();
	const OperandPointer& d() const;
	/*! \brief The source operand. */
	      OperandPointer& a();
	const OperandPointer& a() const;

};

/*! \brief A binary instruction */
class BinaryInstruction : public Instruction
{
public:
	explicit BinaryInstruction(Opcode = InvalidOpcode, BasicBlock* b = nullptr);

public:
	/*! \brief Set the destination, the instruction takes ownership */
	void setD(Operand* d);
	/*! \brief Set the first source, the instruction takes ownership */
	void setA(Operand* a);
	/*! \brief Set the second source, the instruction takes ownership */
	void setB(Operand* b);

public:
	virtual Instruction* clone() const = 0;

public:
	/*! \brief The destination operand. */
	      OperandPointer& d();
	const OperandPointer& d() const;
	/*! \brief The first source operand. */
	      OperandPointer& a();
	const OperandPointer& a() const;
	/*! \brief The second source operand. */
	      OperandPointer& b();
	const OperandPointer& b() const;
};

/*! \brief An instruction involving a comparison */
class ComparisonInstruction : public BinaryInstruction
{
public:
	/*! \brief All possible comparisons */
	enum Comparison
	{
		OrderedEqual,
		OrderedNotEqual,
		OrderedLessThan,
		OrderedLessOrEqual,
		OrderedGreaterThan,
		OrderedGreaterOrEqual,
		UnorderedEqual,
		UnorderedNotEqual,
		UnorderedLessThan,
		UnorderedLessOrEqual,
		UnorderedGreaterThan,
		UnorderedGreaterOrEqual,
		IsANumber,
		NotANumber,
		InvalidComparison
	};

public:
	ComparisonInstruction(Opcode o = InvalidOpcode,
		Comparison c = InvalidComparison, BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const = 0;

public:
	virtual std::string modifierString() const;
	virtual std::string toString() const;
	static std::string toString(Comparison c);

public:
	/*! \brief The comparison operation */
	Comparison comparison;
};

/*! \brief An add instruction */
class Add : public BinaryInstruction
{
public:
	explicit Add(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief An and instruction */
class And : public BinaryInstruction
{
public:
	explicit And(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform arithmetic shift right */
class Ashr : public BinaryInstruction
{
public:
	explicit Ashr(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;
};

/*! \brief An atomic operation instruction */
class Atom : public BinaryInstruction
{
public:
	/*! \brief Possible atomic operations */
	enum Operation
	{
		AtomicAnd,
		AtomicOr,
		AtomicXor,
		AtomicCas,
		AtomicExch,
		AtomicAdd,
		AtomicInc,
		AtomicDec, 
		AtomicMin,
		AtomicMax,		
		InvalidOperation
	};

public:
	explicit Atom(Operation op = InvalidOperation, BasicBlock* b = nullptr);

public:
	/*! \brief Set the third source, the instruction takes ownership */
	void setC(Operand* c);
	/*! \brief Get the third source */
	      OperandPointer& c();
	const OperandPointer& c() const;

public:
	virtual Instruction* clone() const;

public:
	Operation operation;
};

/*! \brief Perform a thread group barrier */
class Bar : public Instruction
{
public:
	explicit Bar(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;
	
};

/*! \brief Perform a raw bitcast */
class Bitcast : public UnaryInstruction
{
public:
	explicit Bitcast(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform a branch */
class Bra : public Instruction
{
public:
	enum BranchModifier
	{
		UniformBranch,
		MultitargetBranch,
		InvalidModifier
	};

public:
	explicit Bra(BranchModifier m = InvalidModifier, BasicBlock* b = nullptr);

public:
	/*! \brief Set the target operand, the instruction takes onwership */
	void setTarget(Operand* o);
	/*! \brief Get the target operand. */	
	      OperandPointer& target();
	const OperandPointer& target() const;
	
public:
	/*! \brief Get the target basic block */
	BasicBlock*       targetBasicBlock();
	/*! \brief Get the target basic block */
	const BasicBlock* targetBasicBlock() const;

public:
	/*! \brief Is the branch unconditional? */
	bool isUnconditional() const;

public:
	virtual Instruction* clone() const;

public:
	BranchModifier modifier;
};

/*! \brief Branch and save the return pc */
class Call : public Instruction
{
public:
	typedef std::vector<Operand*>       OperandVector;
	typedef std::vector<const Operand*> ConstOperandVector;
	
public:
	explicit Call(BasicBlock* b = nullptr);

public:
	/*! \brief Is the call to an intrinsic */
	bool isIntrinsic() const;

public:
	/*! \brief Set the target operand, the instruction takes onwership */
	void setTarget(Operand* o);
	/*! \brief Add a return operand, the instruction takes ownership  */
	void addReturn(Operand* o);
	/*! \brief Add an argument operand, the instruction takes ownership  */
	void addArgument(Operand* o);

public:
	OperandPointer&       target();
	const OperandPointer& target() const;

	OperandVector      returned();
	ConstOperandVector returned() const;

	OperandVector      arguments();
	ConstOperandVector arguments() const;

public:
	virtual Instruction* clone() const;

};

/*! \brief Floating point division */
class Fdiv : public BinaryInstruction
{
public:
	explicit Fdiv(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Floating point multiplication */
class Fmul : public BinaryInstruction
{
public:
	explicit Fmul(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief A floating point precision extension instruction */
class Fpext : public UnaryInstruction
{
public:
	explicit Fpext(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief A floating point to signed integer instruction */
class Fptosi : public UnaryInstruction
{
public:
	explicit Fptosi(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief A floating point to unsigned integer instruction */
class Fptoui : public UnaryInstruction
{
public:
	explicit Fptoui(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief A floating point precision truncate instruction */
class Fptrunc : public UnaryInstruction
{
public:
	explicit Fptrunc(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Floating point remainder */
class Frem : public BinaryInstruction
{
public:
	explicit Frem(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Get a pointer to a subtype of an aggregate */
class Getelementptr : public UnaryInstruction
{
public:
	Getelementptr(BasicBlock* b = nullptr);

public:
	const Type* getSelectedType() const;

public:
	virtual Instruction* clone() const;

};

/*! \brief Launch a new HTA at the specified entry point */
class Launch : public Instruction
{
public:
	explicit Launch(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Load a value from memory */
class Ld : public UnaryInstruction
{
public:
	explicit Ld(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Logical shift right */
class Lshr : public BinaryInstruction
{
public:
	explicit Lshr(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Wait until memory operations at the specified level have completed */
class Membar : public Instruction
{
public:
	enum Level
	{
		Thread = 1,
		Warp   = 2,
		Cta    = 3,
		Kernel = (uint32_t)-1,
		InvalidLevel = 0
	};

public:
	explicit Membar(Level = InvalidLevel, BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

public:
	Level level;	
};


/*! \brief Multiply two operands together */
class Mul : public BinaryInstruction
{
public:
	explicit Mul(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform a logical OR operation */
class Or : public BinaryInstruction
{
public:
	explicit Or(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Return from the current function call, or exit */
class Ret : public Instruction
{
public:
	explicit Ret(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Compare two operands and set a third predicate */
class Setp : public ComparisonInstruction
{
public:
	explicit Setp(Comparison c = InvalidComparison, BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Sign extend an integer */
class Sext : public UnaryInstruction
{
public:
	explicit Sext(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform signed division */
class Sdiv : public BinaryInstruction
{
public:
	explicit Sdiv(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform shift left */
class Shl : public BinaryInstruction
{
public:
	explicit Shl(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Convert a signed int to a floating point */
class Sitofp : public UnaryInstruction
{
public:
	explicit Sitofp(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform a signed remainder operation */
class Srem : public BinaryInstruction
{
public:
	explicit Srem(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform a store operation */
class St : public Instruction
{
public:
	explicit St(BasicBlock* b = nullptr);

public:
	/*! \brief Set the destination, the instruction takes ownership */
	void setD(Operand* d);
	/*! \brief Set the source, the instruction takes ownership */
	void setA(Operand* a);

	/*! \brief Get the destination */
	      OperandPointer& d();
	const OperandPointer& d() const;
	
	/*! \brief Get the source */
	      OperandPointer& a();
	const OperandPointer& a() const;

public:
	virtual Instruction* clone() const;
	
};

/*! \brief Perform a subtract operation */
class Sub : public BinaryInstruction
{
public:
	explicit Sub(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Truncate an integer */
class Trunc : public UnaryInstruction
{
public:
	explicit Trunc(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;
	
};

/*! \brief Perform an unsigned division operation */
class Udiv : public BinaryInstruction
{
public:
	explicit Udiv(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Convert an unsigned int to a floating point */
class Uitofp : public UnaryInstruction
{
public:
	explicit Uitofp(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform an unsigned remainder operation */
class Urem : public BinaryInstruction
{
public:
	explicit Urem(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Perform a logical XOR operation */
class Xor : public BinaryInstruction
{
public:
	explicit Xor(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;

};

/*! \brief Zero extend an integer */
class Zext : public UnaryInstruction
{
public:
	explicit Zext(BasicBlock* b = nullptr);

public:
	virtual Instruction* clone() const;
	
};

/*! \brief A PHI node defines a new value from multiple source values that are
	merged together from multiple control flow paths */
class Phi : public Instruction
{
public:
	typedef RegisterOperand*                    RegisterOperandPointer;
	typedef AddressOperand*                     AddressOperandPointer;
	typedef std::vector<RegisterOperandPointer> RegisterOperandVector;
	typedef std::vector<AddressOperandPointer>  AddressOperandVector;
	typedef std::vector<BasicBlock*>            BasicBlockVector;
	typedef std::vector<const BasicBlock*>      ConstBasicBlockVector;
	typedef std::vector<const RegisterOperand*> ConstRegisterOperandVector;
	typedef std::vector<const AddressOperand*>  ConstAddressOperandVector;
	
public:
	explicit Phi(BasicBlock* b = nullptr);

	Phi(const Phi&);
	Phi& operator=(const Phi&);

public:
	/*! \brief Set the destination, the instruction takes ownership */
	void setD(RegisterOperand* d);
	/*! \brief Add a new source, the instruction takes ownership */
	void addSource(RegisterOperand* source, AddressOperand* predecessor);
	/*! \brief Remove a source from the specified basic block */
	void removeSource(BasicBlock* predecessor);

public:
	      RegisterOperandPointer d();
	const RegisterOperandPointer d() const;
	
	     RegisterOperandVector sources();
	ConstRegisterOperandVector sources() const;
	
public:
 	     BasicBlockVector blocks();
	ConstBasicBlockVector blocks() const;

public:
	     AddressOperandVector blockOperands();
	ConstAddressOperandVector blockOperands() const;

public:
	virtual Instruction* clone() const;
};

/*! \brief A PSI node defines a new value only if at least one of a set of
	predicates are set. */
class Psi : public Instruction
{
public:
	typedef RegisterOperand* RegisterOperandPointer;
	typedef std::vector<RegisterOperand*>  RegisterOperandVector;
	typedef std::vector<PredicateOperand*> PredicateOperandVector;
	
public:
	explicit Psi(BasicBlock* b = nullptr);

public:
	/*! \brief Set the destination, the instruction takes ownership */
	void setD(RegisterOperand* d);
	/*! \brief Add a new source, the instruction takes ownership */
	void addSource(PredicateOperand* predicate, RegisterOperand* reg);
	/*! \brief Remove a source associated with the specified predicate */
	void removeSource(PredicateOperand* predicate);
public:
	virtual Instruction* clone() const;

public:
	RegisterOperandPointer d();
	const RegisterOperandPointer d() const;
	
public:
	RegisterOperandVector  sources();
	PredicateOperandVector predicates();

};

}

}

