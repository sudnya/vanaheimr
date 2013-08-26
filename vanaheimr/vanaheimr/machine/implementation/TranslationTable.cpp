/*! \file   TranslationTable.cpp
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the TranslationTable class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTable.h>
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

#include <vanaheimr/ir/interface/Instruction.h>

// Standard Library Includes
#include <map>
#include <cassert>

namespace vanaheimr
{

namespace machine
{

class TranslationTableMap
{
public:
	typedef std::map<std::string, TranslationTableEntry*> Map;

public:
	~TranslationTableMap();	

public:
	Map opcodeToTranslation;

};

TranslationTableMap::~TranslationTableMap()
{
	for(auto translation : opcodeToTranslation)
	{
		delete translation.second;
	}
}

TranslationTable::TranslationTable()
: _translations(new TranslationTableMap)
{

}

TranslationTable::~TranslationTable()
{
	delete _translations;
}

TranslationTable::MachineInstructionVector
	TranslationTable::translateInstruction(
	const ir::Instruction* instruction) const
{
	auto translation = getTranslation(instruction->opcodeString());
	
	if(translation == nullptr)
	{
		// Fail the translation by returning nothing
		return MachineInstructionVector();
	}

	return translation->translateInstruction(instruction);
}

const TranslationTableEntry* TranslationTable::getTranslation(
	const std::string& name) const
{
	auto translation = _translations->opcodeToTranslation.find(name);
	
	if(translation == _translations->opcodeToTranslation.end())
	{
		return nullptr;
	}
	
	return translation->second;
}

void TranslationTable::addTranslation(const TranslationTableEntry* entry)
{
	assert(_translations->opcodeToTranslation.count(entry->name) == 0);

	_translations->opcodeToTranslation.insert(
		std::make_pair(entry->name, entry->clone()));
}

}

}


