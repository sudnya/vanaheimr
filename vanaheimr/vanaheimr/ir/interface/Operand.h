/*	\file   Operand.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class Argument;        } }
namespace vanaheimr { namespace ir { class VirtualRegister; } }
namespace vanaheimr { namespace ir { class Variable;        } }
namespace vanaheimr { namespace ir { class Instruction;     } }
namespace vanaheimr { namespace ir { class Type;            } }
namespace vanaheimr { namespace ir { class Constant;        } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Represents a single instruction operand */
class Operand
{
public:
	/*! \brief An operand type */
	enum OperandMode
	{
		Register,
		Immediate,
		Predicate,
		Indirect,
		Address,
		Argument
	};
	
	/*! \brief A type to hold a register idenfitier */
	typedef uint64_t RegisterType;

public:
	Operand(OperandMode mode, Instruction* instruction);
	virtual ~Operand();

public:
	/*! \brief Is the operand a register */
	bool isRegister() const;
	
	/*! \brief Is the operand an indirect */
	bool isIndirect() const;
	
	/*! \brief Is the operand an address */
	bool isAddress() const;
	
	/*! \brief Is the operand an immediate */
	bool isImmediate() const;
	
	/*! \brief Is the operand a function argument */
	bool isArgument() const;
	
	/*! \brief Is the operand a basic block */
	bool isBasicBlock() const;

public:
	/*! \brief The mode of the operand determines how it is accessed */
	OperandMode mode() const;

public:
	/*! \brief Does the operand have a compile-time constant value? */
	virtual bool isConstant() const;

	/*! \brief Get a constant value of the operand if it isConstant. 
	    
	    Note that this function returns a new object owned by the caller. */
	virtual Constant* getConstantValue() const;

public:
	/*! \brief Get the operand type */
	virtual const Type* type() const = 0;

public:
	virtual Operand* clone() const = 0;
	virtual std::string toString() const = 0;
	
public:
	/*! \brief The owning instruction */
	Instruction* instruction;

private:
	OperandMode  _mode;
};

typedef Operand::RegisterType RegisterType;

/*! \brief A register operand */
class RegisterOperand : public Operand
{
public:
	RegisterOperand(VirtualRegister* reg, Instruction* i);

public:
	virtual const Type* type() const;
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The register being accessed */
	VirtualRegister* virtualRegister;

protected:
	RegisterOperand(VirtualRegister* reg, Instruction* i, OperandMode m);

};

/*! \brief An immediate operand */
class ImmediateOperand : public Operand
{
public:
	ImmediateOperand(uint64_t v, Instruction* i, const Type* t);
	ImmediateOperand(double   d, Instruction* i, const Type* t);

public:
	virtual bool isConstant() const;
	virtual Constant* getConstantValue() const;
	
public:
	virtual const Type* type() const;
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The immediate value */
	union
	{
		uint64_t uint;
		double   fp;
	};

	/*! \brief The data type */
	const Type* dataType;
};

/*! \brief A predicate operand */
class PredicateOperand : public RegisterOperand
{
public:
	/*! \brief The modifier on the predicate */
	enum PredicateModifier
	{
		StraightPredicate,
		InversePredicate,
		PredicateTrue,
		PredicateFalse
	};

public:
	PredicateOperand(VirtualRegister* reg,
		PredicateModifier mod, Instruction* i);
	PredicateOperand(PredicateModifier mod, Instruction* i);

public:
	bool isAlwaysTrue() const;

public:
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The predicate modifier */
	PredicateModifier modifier;
};

/*! \brief An indirect operand */
class IndirectOperand : public RegisterOperand
{
public:
	IndirectOperand(VirtualRegister* reg, int64_t offset, Instruction* i);

public:
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	/*! \brief The offset to add to the register */
	int64_t offset;
};

/*! \brief An address operand */
class AddressOperand : public Operand
{
public:
	AddressOperand(Variable* value, Instruction* i);

public:
	virtual const Type* type() const;
	virtual Operand* clone() const;
	virtual std::string toString() const;

public:
	Variable* globalValue;
};

class ArgumentOperand : public Operand
{
public:
	ArgumentOperand(ir::Argument* a, Instruction* i);

public:
	virtual const Type* type() const;
	virtual Operand* clone() const;
	virtual std::string toString() const;
	
public:
	ir::Argument* argument;

};

}

}

