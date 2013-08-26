/*! \file   StaticTranslationTableEntry.cpp
	\date   Thursday February 23, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the StaticTranslationTableEntry class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/StaticTranslationTableEntry.h>

#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Instruction.h>
#include <vanaheimr/ir/interface/Constant.h>
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

StaticTranslationTableEntry::Translation::Argument::Argument(
	unsigned int i, bool s)
: immediate(nullptr), type(nullptr), index(i), isSource(s),
	_argumentType(Register)
{

}

StaticTranslationTableEntry::Translation::Argument::Argument(unsigned int i,
 const Type* t)
: immediate(nullptr), type(t), index(i), isSource(false),
	_argumentType(Temporary)
{

}

StaticTranslationTableEntry::Translation::Argument::Argument(const Constant* c)
: immediate(c), type(c->type()), index(0), isSource(false),
	_argumentType(Immediate)
{

}

bool StaticTranslationTableEntry::Translation::Argument::isTemporary() const
{
	return _argumentType == Temporary;
}

bool StaticTranslationTableEntry::Translation::Argument::isImmediate() const
{
	return _argumentType == Immediate;
}

bool StaticTranslationTableEntry::Translation::Argument::isRegister() const
{
	return _argumentType == Register;
}

StaticTranslationTableEntry::Translation::Translation(const Operation* l)
: operation(l)
{
	
}

StaticTranslationTableEntry::StaticTranslationTableEntry(const std::string& n)
: TranslationTableEntry(n)
{

}

typedef std::vector<ir::VirtualRegister*> RegisterVector;

static void mapOperands(ir::Instruction* newInstruction,
	const ir::Instruction* instruction,
	const StaticTranslationTableEntry::Translation& translation,
	RegisterVector& temporaries)
{
	auto operands = instruction->writes;
	operands.insert(operands.end(), instruction->reads.begin(),
		instruction->reads.end());
	
	for(auto argument : translation.arguments)
	{
		if(argument.isImmediate())
		{
			if(argument.immediate->type()->isFloatingPoint())
			{
				auto floatConstant =
					static_cast<const ir::FloatingPointConstant*>(
					argument.immediate);
				
				if(argument.immediate->type()->isSinglePrecisionFloat())
				{
					newInstruction->reads.push_back(new ir::ImmediateOperand(
						floatConstant->asFloat(), newInstruction,
						argument.immediate->type()));
				}
				else
				{
					newInstruction->reads.push_back(new ir::ImmediateOperand(
						floatConstant->asDouble(), newInstruction,
						argument.immediate->type()));
				}
			}
			else
			{
				auto integerConstant = static_cast<const ir::IntegerConstant*>(
					argument.immediate);
				
				newInstruction->reads.push_back(new ir::ImmediateOperand(
					(uint64_t)(*integerConstant), newInstruction,
					argument.immediate->type()));
			}
		}
		else if(argument.isRegister())
		{
			auto operand = operands[argument.index]->clone();
			
			if(argument.isSource)
			{		
				newInstruction->reads.push_back(operand);
			}
			else
			{
				newInstruction->writes.push_back(operand);
			}
		}
		else
		{
			assert(argument.isTemporary());
			
			newInstruction->reads.push_back(
				new ir::RegisterOperand(temporaries[argument.index],
				newInstruction));
		}
	}
}

StaticTranslationTableEntry::MachineInstructionVector
	StaticTranslationTableEntry::translateInstruction(
	const ir::Instruction* instruction) const
{
	MachineInstructionVector translatedInstructions;

	// Create temporary registers
	RegisterVector temporaries;

	auto temps = getTemporaries();
	
	auto function = instruction->block->function();
	
	for(auto temp : temps)
	{
		assert(temp.index == temporaries.size());
	
		temporaries.push_back(&*function->newVirtualRegister(temp.type));
	}
	
	// Translate instructions
	for(auto& entry : translations)
	{
		auto newInstruction = new Instruction(entry.operation);
		
		mapOperands(newInstruction, instruction, entry, temporaries);
	
		translatedInstructions.push_back(newInstruction);
	}

	return translatedInstructions;
}

TranslationTableEntry* StaticTranslationTableEntry::clone() const
{
	return new StaticTranslationTableEntry(*this);
}

unsigned int StaticTranslationTableEntry::totalArguments() const
{
	unsigned int args = 0;

	for(TranslationVector::const_iterator entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(ArgumentVector::const_iterator argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isRegister())
			{
				args = std::max(args, argument->index + 1);
			}
		}
	}
	
	return args;
}

unsigned int StaticTranslationTableEntry::totalTemporaries() const
{
	unsigned int temps = 0;

	for(auto entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(auto argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isTemporary())
			{
				temps = std::max(temps, argument->index + 1);
			}
		}
	}
	
	return temps;
}

StaticTranslationTableEntry::ArgumentVector
	StaticTranslationTableEntry::getTemporaries() const
{
	ArgumentVector temps(totalTemporaries(), Translation::Argument(nullptr));
	
	for(auto entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(auto argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isTemporary())
			{
				temps[argument->index] = *argument;
			}
		}
	}
	
	return temps;
}

StaticTranslationTableEntry::iterator StaticTranslationTableEntry::begin()
{
	return translations.begin();
}

StaticTranslationTableEntry::const_iterator
	StaticTranslationTableEntry::begin() const
{
	return translations.begin();
}

StaticTranslationTableEntry::iterator
	StaticTranslationTableEntry::end()
{
	return translations.end();
}

StaticTranslationTableEntry::const_iterator
	StaticTranslationTableEntry::end() const
{
	return translations.end();
}

size_t StaticTranslationTableEntry::size() const
{
	return translations.size();
}

bool StaticTranslationTableEntry::empty() const
{
	return translations.empty();
}

}

}


