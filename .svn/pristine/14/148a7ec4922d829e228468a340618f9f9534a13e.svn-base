/*! \file   Function.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Function class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Argument.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

namespace vanaheimr
{

namespace ir
{

/*! \brief Describes a vanaheimr function */
class Function : public Variable
{
public:
	typedef std::list<BasicBlock>      BasicBlockList;
	typedef std::list<Argument>        ArgumentList;
	typedef std::list<VirtualRegister> VirtualRegisterList;

	typedef BasicBlockList::iterator       iterator;
	typedef BasicBlockList::const_iterator const_iterator;

	typedef ArgumentList::iterator       argument_iterator;
	typedef ArgumentList::const_iterator const_argument_iterator;

	typedef VirtualRegisterList::iterator       register_iterator;
	typedef VirtualRegisterList::const_iterator const_register_iterator;

public:
	Function(const std::string& name = "", Module* m = 0,
		Linkage l = InternalLinkage, Visibility v = HiddenVisibility,
		const Type* type = 0);
	Function(const Function& f);
	Function& operator=(const Function& f);
	
public:
	iterator       begin();
	const_iterator begin() const;
	
	iterator       end();
	const_iterator end() const;

	iterator       entry_block();
	const_iterator entry_block() const;

	iterator       exit_block();
	const_iterator exit_block() const;

public:
	size_t size()  const;
	bool   empty() const;

public:
	      BasicBlock& front();
	const BasicBlock& front() const;

	      BasicBlock& back();
	const BasicBlock& back() const;

public:
	iterator newBasicBlock(iterator position, const std::string& name);
	register_iterator newVirtualRegister(const Type* type,
		const std::string& name = "");
	argument_iterator newArgument(const Type* type,
		const std::string& name);
	argument_iterator newReturnValue(const Type* type,
		const std::string& name);

public:
	argument_iterator       argument_begin();
	const_argument_iterator argument_begin() const;
	
	argument_iterator       argument_end();
	const_argument_iterator argument_end() const;

public:
	size_t argument_size()  const;
	bool   argument_empty() const;

public:
	argument_iterator       returned_begin();
	const_argument_iterator returned_begin() const;
	
	argument_iterator       returned_end();
	const_argument_iterator returned_end() const;

public:
	size_t returned_size()  const;
	bool   returned_empty() const;
public:
	register_iterator       register_begin();
	const_register_iterator register_begin() const;
	
	register_iterator       register_end();
	const_register_iterator register_end() const;

public:
	size_t register_size()  const;
	bool   register_empty() const;

public:
	void clear();

private:
	BasicBlockList      _blocks;
	ArgumentList        _returnValues;
	ArgumentList        _arguments;
	VirtualRegisterList _registers;
	
	iterator _entry;
	iterator _exit;

	BasicBlock::Id      _nextBlockId;
	VirtualRegister::Id _nextRegisterId;
};

}

}


