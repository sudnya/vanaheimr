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
#include <vanaheimr/ir/interface/Local.h>

// Standard Library Includes
#include <list>
#include <set>

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
	typedef std::list<std::string>     StringList;
	typedef std::list<Local>           LocalList;
	
	typedef BasicBlockList::iterator       iterator;
	typedef BasicBlockList::const_iterator const_iterator;

	typedef ArgumentList::iterator       argument_iterator;
	typedef ArgumentList::const_iterator const_argument_iterator;
	
	typedef VirtualRegisterList::iterator       register_iterator;
	typedef VirtualRegisterList::const_iterator const_register_iterator;

	typedef LocalList::iterator       local_iterator;
	typedef LocalList::const_iterator const_local_iterator;
	
	typedef unsigned int Id;

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
	bool isPrototype() const;
	bool isIntrinsic() const;
	
	bool hasAttribute(const std::string& attribute) const;

	StringList attributes() const;

public:
	iterator newBasicBlock(iterator position, const std::string& name);
	register_iterator newVirtualRegister(const Type* type,
		const std::string& name = "");
	argument_iterator newArgument(const Type* type,
		const std::string& name);
	argument_iterator newReturnValue(const Type* type,
		const std::string& name);
	local_iterator newLocalValue(const std::string& name,
		const Type* t, Variable::Linkage l, ir::Global::Level le);

public:
	void addAttribute(const std::string& attribute);
	void removeAttribute(const std::string& attribute);

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
	register_iterator erase(const register_iterator&);
	register_iterator erase(const VirtualRegister*);

public:
	register_iterator findVirtualRegister(const std::string& name);
	         iterator findBasicBlock(const std::string& name);

public:
	/*! \brief Move a basic block to a new position */
	void moveBasicBlock(iterator position, iterator block);

public:
	local_iterator       local_begin();
	const_local_iterator local_begin() const;
	
	local_iterator       local_end();
	const_local_iterator local_end() const;

public:
	size_t local_size()  const;
	bool   local_empty() const;

public:
	local_iterator findLocalValue(const std::string& name);

public:
	size_t instruction_size()  const;
	bool   instruction_empty() const;
	
public:
	/*! \brief Get a unique ID for the function in the module */
	Id id() const;

public:
	void clear();

public:
	/*! \brief Set the type of the function by examining the arguments. */
	void interpretType();

private:
	typedef std::set<std::string> StringSet;

private:
	BasicBlockList      _blocks;
	ArgumentList        _returnValues;
	ArgumentList        _arguments;
	VirtualRegisterList _registers;
	StringSet           _attributes;
	LocalList           _locals;
	
	iterator _entry;
	iterator _exit;

	BasicBlock::Id      _nextBlockId;
	VirtualRegister::Id _nextRegisterId;
};

}

}


