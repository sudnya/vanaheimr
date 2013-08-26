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
	
	typedef InstructionList::reverse_iterator       reverse_iterator;
	typedef InstructionList::const_reverse_iterator const_reverse_iterator;
	
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
	reverse_iterator       rbegin();
	const_reverse_iterator rbegin() const;

	reverse_iterator       rend();
	const_reverse_iterator rend() const;

public:
	Instruction*&        front();
	Instruction* const & front() const;

	Instruction*&        back();
	Instruction* const & back() const;

public:
	/*! \brief Get an iterator to a function in the block */
	iterator       getIterator(const Instruction*);
	/*! \brief Get an iterator to a function in the block */
	const_iterator getIterator(const Instruction*) const;

public:
	bool   empty() const;
	size_t size()  const;

public:
	Id        id()       const;
	Function* function() const;
	
public:
	/*! \brief Pushes an instruction to the back of the block */
	void push_back(Instruction* i);
	/*! \brief Pushes an instruction to the front of the block */
	void push_front(Instruction* i);

public:
	/*! \brief Remove the first instruction in the block */
	void pop_front();
	
	/*! \brief Remove the last instruction in the block */
	void pop_back();

public:
	/*! \brief Inserts an instruction into the block */
	iterator insert(iterator position, Instruction* i);
	/*! \brief Inserts an instruction into the block */
	iterator insert(const Instruction* position, Instruction* i);

public:
	/*! \brief Erase an instruction from the block, deleting it */
	iterator erase(iterator position);
	/*! \brief Erase an instruction from the block, deleting it */
	iterator erase(const Instruction* position);
	
public:
	/*! \brief Assign instructions to the block */
	template <typename Iterator>
	void assign(Iterator begin, Iterator end);
	
public:
	/*! \brief Delete all instructions within the block */
	void clear();
	/*! \brief Set the owning function */
	void setFunction(Function*);

public:
	/*! \brief Set the name of the basic block */
	void setName(const std::string& name);

private:
	Function*       _function;
	InstructionList _instructions;
	Id              _id;
	Instruction::Id _nextInstructionId;
};

template <typename Iterator>
void BasicBlock::assign(Iterator begin, Iterator end)
{
	_instructions.assign(begin, end);
}


}

}

