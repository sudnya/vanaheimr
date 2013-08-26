/*! \file   Instruction.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the Instruction class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>

// Standard Library Includes
#include <vector>
#include <list>

// Forward Declaration
namespace vanaheimr { namespace machine { class Operation; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A model for an abstract machine instruction */
class Instruction : public vanaheimr::ir::Instruction
{
public:
	typedef ir::BasicBlock BasicBlock;

public:
	Instruction(const Operation* op, BasicBlock* block = 0, Id id = 0);

public:
	virtual bool isLoad()      const;
	virtual bool isStore()     const;
	virtual bool isBranch()    const;
	virtual bool isCall()      const;
	virtual bool isReturn()    const;
	
	virtual bool isMemoryBarrier() const;

public:
	virtual std::string opcodeString() const;

public:
	virtual vanaheimr::ir::Instruction* clone() const;

public:
	 /*! \brief The machine operation performed by the instruction. */
	const Operation* operation;

private:
	bool hasSpecialProperty(const std::string& property) const;

};

}

}


