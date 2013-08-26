/*! \file   BinaryWriter.h
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the helper class that traslates compiler IR to a binary.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryHeader.h>

#include <vanaheimr/asm/interface/SymbolTableEntry.h>

#include <vanaheimr/asm/interface/Instruction.h>

// Standard Library Includes
#include <vector>
#include <unordered_map>
#include <ostream>

// Forward Declarations
namespace vanaheimr { namespace ir { class Module;      } }
namespace vanaheimr { namespace ir { class Global;      } }
namespace vanaheimr { namespace ir { class Instruction; } }
namespace vanaheimr { namespace ir { class Operand;     } }
namespace vanaheimr { namespace ir { class Argument;    } }
namespace vanaheimr { namespace ir { class Variable;    } }

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace as
{

/*! \brief Represents a single compilation unit. */
class BinaryWriter
{
public:
	typedef vanaheimr::ir::Instruction Instruction;
	typedef vanaheimr::ir::Operand     Operand;
	typedef std::vector<SymbolTableEntry> SymbolTableEntryVector;
	typedef SymbolTableEntryVector::iterator symbol_iterator;

public:
	static const unsigned int PageSize = BinaryHeader::PageSize;

public:
	BinaryWriter();
	void write(std::ostream& binary, const ir::Module& inputModule);

private:
	void writePage(std::ostream& binary, const void* data,
		uint64_t size);

	void populateHeader();
	void populateInstructions();
	void populateData();
	void linkSymbols();

private:
	size_t getHeaderOffset() const;
	size_t getInstructionOffset() const;
	size_t getDataOffset() const;
	size_t getSymbolTableOffset() const;
	size_t getStringTableOffset() const;

	size_t getSymbolTableSize() const;
	size_t getInstructionStreamSize() const;
	size_t getDataSize() const;
	size_t getStringTableSize() const;
	
	void convertComplexInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertUnaryInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertBinaryInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertComparisonInstruction(InstructionContainer& container,
		const Instruction& instruction);
	
	OperandContainer     convertOperand(const Operand&);
	InstructionContainer convertToContainer(const Instruction&); 

	size_t getSymbolTableOffset(const ir::Argument* a);
	size_t getSymbolTableOffset(const ir::Variable* g);
	size_t getSymbolTableOffset(const std::string& name);
	size_t getBasicBlockSymbolTableOffset(const ir::Variable* g);
	SymbolTableEntryVector::iterator getSymbol(const std::string& name);
	
	void addSymbol(unsigned int type, unsigned int linkage,
		unsigned int visibility, unsigned int level,
		const std::string& name, uint64_t offset, uint64_t size,
		const std::string& typeName, const std::string& attributeName = "");
	void addGlobal(const ir::Global&);
	void patchSymbol(const std::string& name, uint64_t offset, uint64_t size);

	void alignData(uint64_t alignment);

private:
	void convertStInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertBraInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertCallInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertRetInstruction(InstructionContainer& container,
		const Instruction& instruction);
	void convertPhiInstruction(InstructionContainer& container,
		const Instruction& instruction);

private:
	void addOperandToDataSection(const OperandContainer& operand);

private:
	static uint64_t align(uint64_t address, uint64_t alignment);
	static uint64_t pageAlign(uint64_t address);

private:
	typedef std::vector<InstructionContainer>         InstructionVector;
	typedef std::vector<char>                         DataVector;
	typedef std::vector<SymbolTableEntry>             SymbolVector;
	typedef std::unordered_map<std::string, uint64_t> OffsetMap;
	typedef std::unordered_map<uint64_t, uint64_t>    OffsetToSymbolMap;

private:
	const ir::Module*  m_module;
	
	BinaryHeader      m_header;
	InstructionVector m_instructions;
	DataVector        m_data;
	SymbolVector      m_symbolTable;
	DataVector        m_stringTable;

private:
	OffsetMap         m_basicBlockOffsets;
	OffsetToSymbolMap m_basicBlockSymbols;
};

}

}

