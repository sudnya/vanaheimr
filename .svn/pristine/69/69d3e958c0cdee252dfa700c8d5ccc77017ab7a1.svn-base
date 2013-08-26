/*! \file   BasicBlock.cpp
	\date   Friday February 10, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BasicBlock class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Function.h>

#include <vanaheimr/compiler/interface/Compiler.h>

// Standard Library Includes
#include <algorithm>

namespace vanaheimr
{

namespace ir
{

BasicBlock::BasicBlock(Function* f, Id i, const std::string& name)
: Variable(name, f->module(),
	compiler::Compiler::getSingleton()->getBasicBlockType(),
	InternalLinkage, HiddenVisibility), _function(f), _id(i)
{

}

BasicBlock::~BasicBlock()
{
	clear();
}

BasicBlock::BasicBlock(const BasicBlock& bb)
: Variable(bb), _function(bb.function()), _id(bb.id())
{
	for(auto instruction : bb)
	{
		push_back(instruction->clone());
	}
}

BasicBlock& BasicBlock::operator=(const BasicBlock& bb)
{
	if(this == &bb) return *this;
	
	clear();

	Variable::operator=(bb);

	_id       = bb.id();
	_function = bb.function();
	
	for(auto instruction : bb)
	{
		push_back(instruction->clone());
	}
	
	return *this;
}

Instruction* BasicBlock::terminator()
{
	if(empty()) return 0;
	
	if(back()->isBranch() || back()->isReturn()) return back();
	
	return 0;
}

const Instruction* BasicBlock::terminator() const
{
	if(empty()) return 0;
	
	if(back()->isBranch() || back()->isReturn()) return back();
	
	return 0;
}

void BasicBlock::setTerminator(Instruction* i)
{
	if(terminator() != 0)
	{
		delete back();
		back() = i->clone();
	}
	else
	{
		push_back(i);
	}
}

BasicBlock::iterator BasicBlock::begin()
{
	return _instructions.begin();
}

BasicBlock::const_iterator BasicBlock::begin() const
{
	return _instructions.begin();
}

BasicBlock::iterator BasicBlock::end()
{
	return _instructions.end();
}

BasicBlock::const_iterator BasicBlock::end() const
{
	return _instructions.end();
}

BasicBlock::reverse_iterator BasicBlock::rbegin()
{
	return _instructions.rbegin();
}

BasicBlock::const_reverse_iterator BasicBlock::rbegin() const
{
	return _instructions.rbegin();
}

BasicBlock::reverse_iterator BasicBlock::rend()
{
	return _instructions.rend();
}

BasicBlock::const_reverse_iterator BasicBlock::rend() const
{
	return _instructions.rend();
}

Instruction*& BasicBlock::front()
{
	return _instructions.front();
}

Instruction* const& BasicBlock::front() const
{
	return _instructions.front();
}

Instruction*& BasicBlock::back()
{
	return _instructions.back();
}

Instruction* const& BasicBlock::back() const
{
	return _instructions.back();
}


BasicBlock::iterator BasicBlock::getIterator(const Instruction* instruction)
{
	for(auto i = begin(); i != end(); ++i)
	{
		if(*i == instruction) return i;
	}
	
	return end();
}

BasicBlock::const_iterator BasicBlock::getIterator(
	const Instruction* instruction) const
{
	for(auto i = begin(); i != end(); ++i)
	{
		if(*i == instruction) return i;
	}
	
	return end();
}

bool BasicBlock::empty() const
{
	return _instructions.empty();
}

size_t BasicBlock::size() const
{
	return _instructions.size();
}

BasicBlock::Id BasicBlock::id() const
{
	return _id;
}

Function* BasicBlock::function() const
{
	return _function;
}

void BasicBlock::push_back(Instruction* i)
{
	i->block = this;
	
	_instructions.push_back(i);
}

void BasicBlock::push_front(Instruction* i)
{
	i->block = this;
	
	_instructions.push_front(i);
}

void BasicBlock::pop_back()
{
	_instructions.pop_back();
}

void BasicBlock::pop_front()
{
	_instructions.pop_front();
}

BasicBlock::iterator BasicBlock::insert(
	const Instruction* pointer, Instruction* i)
{
	auto position = std::find(begin(), end(), pointer);
	
	return insert(position, i);
}

BasicBlock::iterator BasicBlock::insert(iterator position, Instruction* i)
{
	return _instructions.insert(position, i);
}

BasicBlock::iterator BasicBlock::erase(iterator position)
{
	delete *position;
	return _instructions.erase(position);
}

BasicBlock::iterator BasicBlock::erase(const Instruction* pointer)
{
	auto position = std::find(begin(), end(), pointer);
	
	return erase(position);
}
	

void BasicBlock::clear()
{
	for(auto instruction : *this)
	{
		delete instruction;
	}
}

void BasicBlock::setFunction(Function* f)
{
	_function = f;
}

void BasicBlock::setName(const std::string& n)
{
	_setName(n);
}

}

}

