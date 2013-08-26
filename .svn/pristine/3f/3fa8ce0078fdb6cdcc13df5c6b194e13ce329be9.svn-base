/*! \file   BinaryReader.h
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the BinaryReader class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryHeader.h>
#include <vanaheimr/asm/interface/SymbolTableEntry.h>

#include <vanaheimr/ir/interface/Function.h>

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

// Standard Library Includes
#include <istream>
#include <vector>

namespace vanaheimr { namespace ir { class Constant; } }

namespace vanaheimr
{

namespace as
{

/*! \brief Reads in a vanaheimr bytecode file yielding a module. */
class BinaryReader
{
public:
	/*! \brief Attempts to read from a binary stream, returns a module */
	ir::Module* read(std::istream& stream, const std::string& name);

private:
	typedef archaeopteryx::ir::InstructionContainer InstructionContainer;
	typedef std::vector<InstructionContainer>       InstructionVector;
	typedef std::vector<char>                       DataVector;
	typedef std::vector<SymbolTableEntry>           SymbolVector;

	typedef SymbolVector::iterator symbol_iterator;

	class BasicBlockDescriptor
	{
	public:
		BasicBlockDescriptor(const std::string& name = "", uint64_t b = 0,
			uint64_t e = 0);

	public:
		std::string name;
		uint64_t    begin; // first instruction
		uint64_t    end; // last instruction + 1
	};

	typedef std::vector<BasicBlockDescriptor> BasicBlockDescriptorVector;

private:
	void _readHeader(std::istream& stream);
	void _readDataSection(std::istream& stream);
	void _readStringTable(std::istream& stream);
	void _readSymbolTable(std::istream& stream);
	void _readInstructions(std::istream& stream);

private:
	void _initializeModule(ir::Module& m) const;

	void _loadGlobals(ir::Module& m) const;
	void _loadFunctions(ir::Module& m) const;

private:
	std::string           _getSymbolName(const SymbolTableEntry& symbol)     const;
	std::string           _getSymbolTypeName(const SymbolTableEntry& symbol) const;
	ir::Type*             _getSymbolType(const SymbolTableEntry& symbol)     const;
	ir::Variable::Linkage _getSymbolLinkage(const SymbolTableEntry& symbol)  const;

	bool          _hasInitializer(const SymbolTableEntry& symbol) const;
	ir::Constant* _getInitializer(const SymbolTableEntry& symbol) const;

	BasicBlockDescriptorVector _getBasicBlocksInFunction(
		const SymbolTableEntry& name) const;

	void _addInstruction(ir::Function::iterator block,
		const InstructionContainer& container) const;

	bool _addSimpleBinaryInstruction(ir::Function::iterator block,
		const InstructionContainer& container) const;
	bool _addSimpleUnaryInstruction(ir::Function::iterator block,
		const InstructionContainer& container) const;
	bool _addComplexInstruction(ir::Function::iterator block,
		const InstructionContainer& container) const;

private:
	BinaryHeader _header;

	InstructionVector _instructions;
	DataVector        _dataSection;
	DataVector        _stringTable;
	SymbolVector      _symbolTable;

private:
	

};

}

}


