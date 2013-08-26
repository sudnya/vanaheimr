/*! \file   BasicBlock.h
	\date   Friday February 10, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the BasicBlock class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>
#include <vanaheimr/ir/interface/Variable.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace vanaheimr { namespace ir { class Function; } }

namespace vanaheimr
{

namespace ir
{

/*! \brief A list of instructions ending with a terminator. */
class BasicBlock : public Variable
{
public:
	typedef std::list<Instruction*> InstructionList;

	typedef InstructionList::iterator       iterator;
	typedef InstructionList::const_iterator const_iterator;

	typedef unsigned int Id;

public:
	BasicBlock(Function* f, Id i, const std::string& name);
	~BasicBlock();
	BasicBlock(const BasicBlock& b);
	BasicBlock& operator=(const BasicBlock&);
	
public:
	/*! \brief Return the terminator instruction if there is one */
	      Instruction* terminator();
	/*! \brief Return the terminator instruction if there is one */
	const Instruction* terminator() const;

public:
	/*! \brief Inserts a new terminator instruction, possibly replacing */
	void setTerminator(Instruction* i);

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	Instruction*&        front();
	Instruction* const & front() const;

	Instruction*&        back();
	Instruction* const & back() const;

public:
	bool   empty() const;
	size_t size()  const;

public:
	Id id() const;
	
public:
	/*! \brief Pushes an instruction copy to the back of the block */
	void push_back(Instruction* i);
	/*! \brief Pushes an instruction copy to the front of the block */
	void push_front(Instruction* i);

public:
	/*! \brief Inserts an instruction copy into the block */
	iterator insert(iterator position, Instruction* i);

public:
	/*! \brief Delete all instructions within the block */
	void clear();
	/*! \brief Set the owning function */
	void setFunction(Function*);

private:
	Function*       _function;
	InstructionList _instructions;
	Id              _id;
};

}

}

