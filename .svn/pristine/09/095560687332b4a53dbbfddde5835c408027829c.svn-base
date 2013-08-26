/*! \file   Function.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Function class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Function.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_map>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace ir
{

Function::Function(const std::string& n, Module* m, Linkage l, Visibility v,
	const Type* t)
: Variable(n, m, t, l, v), _nextBlockId(0), _nextRegisterId(0)
{
	_entry = newBasicBlock(end(), "__Entry");
	_exit  = newBasicBlock(end(), "__Exit");
}

Function::Function(const Function& f)
: Variable(f), _nextBlockId(0), _nextRegisterId(0)
{
	operator=(f);
}

Function& Function::operator=(const Function& f)
{
	typedef std::unordered_map<VirtualRegister::Id,
		VirtualRegister*> VirtualRegisterMap;
	typedef std::unordered_map<std::string, Argument*>
		ArgumentMap;
	typedef std::unordered_map<BasicBlock::Id,
		BasicBlock*> BasicBlockMap;

	if(&f == this) return *this;
	
	clear();
	
	_nextBlockId    = f._nextBlockId;

	BasicBlockMap basicBlockMapping;

	for(const_iterator block = f.begin(); block != f.end(); ++block)
	{
		if(block == f.exit_block())  continue;
		if(block == f.entry_block()) continue;
	
		auto newBlock = _blocks.insert(exit_block(), *block);
		newBlock->setFunction(this);
	
		basicBlockMapping.insert(std::make_pair(newBlock->id(), &*newBlock));
	}
	
	_arguments = f._arguments;
	ArgumentMap argumentMapping;

	for(argument_iterator argument = argument_begin();
		argument != argument_end(); ++argument)
	{
		argumentMapping.insert(std::make_pair(argument->name(), &*argument));

		argument->setFunction(this);
	}

	_returnValues = f._returnValues;
	
	for(argument_iterator value = returned_begin(); value != returned_end(); ++value)
	{
		argumentMapping.insert(std::make_pair(value->name(), &*value));

		value->setFunction(this);
	}
	
	// Virtual Registers
	VirtualRegisterMap registerMapping;

	for(const_register_iterator reg = f.register_begin(); reg != f.register_end(); ++reg)
	{
		registerMapping.insert(std::make_pair(
			reg->id, &*newVirtualRegister(reg->type, reg->name)));
	}
	
	report("Copying Function");

	for(iterator block = begin(); block != end(); ++block)
	{
		report(" " << block->name());
		for(auto instruction : *block)
		{
			report("  " << instruction->toString());
			for(auto operand : instruction->reads)
			{
				report("   read " << operand->toString());				

				if(operand->isRegister())
				{
					RegisterOperand* reg =
						static_cast<RegisterOperand*>(operand);
				
					VirtualRegisterMap::iterator mapping =
						registerMapping.find(reg->virtualRegister->id);
					assert(mapping != registerMapping.end());
					
					reg->virtualRegister = mapping->second;
				}

				if(operand->isArgument())
				{
					ArgumentOperand* argument =
						static_cast<ArgumentOperand*>(operand);	
				
					ArgumentMap::iterator mapping =
						argumentMapping.find(argument->argument->name());
					assert(mapping != argumentMapping.end());

					argument->argument = mapping->second;
				}

				if(operand->isBasicBlock())
				{
					AddressOperand* address =
						static_cast<AddressOperand*>(operand);
					
					BasicBlock* block = static_cast<BasicBlock*>(address->globalValue);

					BasicBlockMap::iterator mapping = basicBlockMapping.find(block->id());
					assert(mapping != basicBlockMapping.end());

					address->globalValue = mapping->second;
				}
			}
			
			for(auto operand : instruction->writes)
			{
				report("   write " << operand->toString());				
				
				if(operand->isRegister())
				{
					RegisterOperand* reg =
						static_cast<RegisterOperand*>(operand);
				
					VirtualRegisterMap::iterator mapping =
						registerMapping.find(reg->virtualRegister->id);
					assert(mapping != registerMapping.end());
					
					reg->virtualRegister = mapping->second;
				}
			}
		}
	}
	
	return *this;
}

Function::iterator Function::begin()
{
	return _blocks.begin();
}

Function::const_iterator Function::begin() const
{
	return _blocks.begin();
}

Function::iterator Function::end()
{
	return _blocks.end();
}

Function::const_iterator Function::end() const
{
	return _blocks.end();
}

Function::iterator Function::entry_block()
{
	return _entry;
}

Function::const_iterator Function::entry_block() const
{
	return _entry;
}

Function::iterator Function::exit_block()
{
	return _exit;
}

Function::const_iterator Function::exit_block() const
{
	return _exit;
}

size_t Function::size() const
{
	return _blocks.size();
}

bool Function::empty() const
{
	return _blocks.empty();
}

BasicBlock& Function::front()
{
	return _blocks.front();
}

const BasicBlock& Function::front() const
{
	return _blocks.front();
}

BasicBlock& Function::back()
{
	return _blocks.back();
}

const BasicBlock& Function::back() const
{
	return _blocks.back();
}

Function::iterator Function::newBasicBlock(iterator position,
	const std::string& name)
{
	return _blocks.insert(position, BasicBlock(this, _nextBlockId++, name));
}

Function::register_iterator Function::newVirtualRegister(const Type* type,
	const std::string& name)
{
	return _registers.insert(register_end(),
		VirtualRegister(name, _nextRegisterId++, this, type));	
}

Function::argument_iterator Function::newArgument(const Type* type,
	const std::string& name)
{
	return _arguments.insert(argument_end(), Argument(type, this, name));
}

Function::argument_iterator Function::newReturnValue(const Type* type,
	const std::string& name)
{
	return _returnValues.insert(returned_end(), Argument(type, this, name));
}

Function::argument_iterator Function::argument_begin()
{
	return _arguments.begin();
}

Function::const_argument_iterator Function::argument_begin() const
{
	return _arguments.begin();
}

Function::argument_iterator Function::argument_end()
{
	return _arguments.end();
}

Function::const_argument_iterator Function::argument_end() const
{
	return _arguments.end();
}

size_t Function::argument_size() const
{
	return _arguments.size();
}

bool Function::argument_empty() const
{
	return _arguments.empty();
}

Function::argument_iterator Function::returned_begin()
{
	return _returnValues.begin();
}

Function::const_argument_iterator Function::returned_begin() const
{
	return _returnValues.begin();
}

Function::argument_iterator Function::returned_end()
{
	return _returnValues.end();
}

Function::const_argument_iterator Function::returned_end() const
{
	return _returnValues.end();
}

size_t Function::returned_size() const
{
	return _returnValues.size();
}

bool Function::returned_empty() const
{
	return _returnValues.empty();
}

Function::register_iterator Function::register_begin()
{
	return _registers.begin();
}

Function::const_register_iterator Function::register_begin() const
{
	return _registers.begin();
}

Function::register_iterator Function::register_end()
{
	return _registers.end();
}

Function::const_register_iterator Function::register_end() const
{
	return _registers.end();
}

size_t Function::register_size() const
{
	return _registers.size();
}

bool Function::register_empty() const
{
	return _registers.empty();
}

void Function::clear()
{
	_blocks.clear();
	_arguments.clear();
	_registers.clear();
	
	_nextBlockId    = 0;
	_nextRegisterId = 0;

	_entry = newBasicBlock(end(), "__Entry");
	_exit  = newBasicBlock(end(), "__Exit");
}

}

}


