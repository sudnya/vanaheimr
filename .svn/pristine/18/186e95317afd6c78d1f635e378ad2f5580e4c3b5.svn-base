/*! \file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream, const std::string& name)
{
	_readHeader(stream);
	_readDataSection(stream);
	_readStringTable(stream);
	_readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(name,
		compiler::Compiler::getSingleton());

	_initializeModule(*module);
	
	return module;
}

void BinaryReader::_readHeader(std::istream& stream)
{
	report("Reading header...");
	stream.read((char*)&_header, sizeof(BinaryHeader));

	if(stream.gcount() != sizeof(BinaryHeader))
	{
		throw std::runtime_error("Failed to read binary "
			"header, hit EOF.");
	}

	report(" data pages:    " << _header.dataPages);
	report(" code pages:    " << _header.codePages);
	report(" symbols:       " << _header.symbols);
	report(" string pages:  " << _header.stringPages);
	report(" data offset:   " << _header.dataOffset);
	report(" code offset:   " << _header.codeOffset);
	report(" symbol offset: " << _header.symbolOffset);
	report(" string offset: " << _header.stringsOffset);
	report(" name offset:   " << _header.nameOffset);
}

void BinaryReader::_readDataSection(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.dataPages;

	stream.seekg(_header.dataOffset, std::ios::beg);

	_dataSection.resize(dataSize);

	stream.read((char*) _dataSection.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read binary data section, hit"
			" EOF."); 
	}
}

void BinaryReader::_readStringTable(std::istream& stream)
{
	size_t stringTableSize = BinaryHeader::PageSize * _header.stringPages;

	stream.seekg(_header.stringsOffset, std::ios::beg);

	_stringTable.resize(stringTableSize);

	stream.read((char*) _stringTable.data(), stringTableSize);

	if((size_t)stream.gcount() != stringTableSize)
	{
		throw std::runtime_error("Failed to read string table, hit EOF");
	}
}

void BinaryReader::_readSymbolTable(std::istream& stream)
{
	size_t symbolTableSize = sizeof(SymbolTableEntry) * _header.symbols;

	stream.seekg(_header.symbolOffset, std::ios::beg);

	_symbolTable.resize(_header.symbols);

	stream.read((char*) _symbolTable.data(), symbolTableSize);

	if((size_t)stream.gcount() != symbolTableSize)
	{
		throw std::runtime_error("Failed to read symbol table, hit EOF");
	}
}

void BinaryReader::_readInstructions(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.codePages;
	size_t sizeInInstructions = (dataSize + sizeof(InstructionContainer) - 1) /
		sizeof(InstructionContainer);

	_instructions.resize(sizeInInstructions);

	// TODO obey page alignment
	stream.read((char*) _instructions.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read code section, hit EOF.");
	}
}

void BinaryReader::_initializeModule(ir::Module& m) const
{
	_loadGlobals(m);
	_loadFunctions(m);
}

void BinaryReader::_loadGlobals(ir::Module& m) const
{
	for(auto symbol : _symbolTable)
	{
		if(symbol.type != SymbolTableEntry::VariableType)
		{
			continue;
		}

		auto global = m.newGlobal(_getSymbolName(symbol), _getSymbolType(symbol),
			_getSymbolLinkage(symbol));

		if(_hasInitializer(symbol))
		{
			global->setInitializer(_getInitializer(symbol));
		}
	}
}

void BinaryReader::_loadFunctions(ir::Module& m) const
{
	for(auto symbol : _symbolTable)
	{
		if(symbol.type != SymbolTableEntry::FunctionType)
		{
			continue;
		}

		ir::Module::iterator function = m.newFunction(_getSymbolName(symbol),
			_getSymbolLinkage(symbol));
		
		BasicBlockDescriptorVector blocks = _getBasicBlocksInFunction(symbol);
		
		for(auto blockOffset : blocks)
		{
			ir::Function::iterator block = function->newBasicBlock(
				function->end(), blockOffset.name);
			
			for(unsigned int i = blockOffset.begin; i != blockOffset.end; ++i)
			{
				_addInstruction(block, _instructions[i]);
			}
		}
	}
}

std::string BinaryReader::_getSymbolName(const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.stringOffset);
}

std::string BinaryReader::_getSymbolTypeName(const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.typeOffset);
}

ir::Type* BinaryReader::_getSymbolType(const SymbolTableEntry& symbol) const
{
	return compiler::Compiler::getSingleton()->getType(
		_getSymbolTypeName(symbol));
}

ir::Variable::Linkage BinaryReader::_getSymbolLinkage(const SymbolTableEntry& symbol) const
{
	return (ir::Variable::Linkage)(symbol.attributes.linkage);
}

bool BinaryReader::_hasInitializer(const SymbolTableEntry& symbol) const
{
	assertM(false, "Not implemented.");
	return false;
}

ir::Constant* BinaryReader::_getInitializer(const SymbolTableEntry& symbol) const
{
	assertM(false, "Not imeplemented.");
}

BinaryReader::BasicBlockDescriptorVector
	BinaryReader::_getBasicBlocksInFunction(const SymbolTableEntry& symbol) const
{
	typedef std::unordered_set<uint64_t> TargetSet;

	BasicBlockDescriptorVector blocks;
	
	// Get the first and last instruction in the function
	uint64_t begin = (symbol.offset - _header.codeOffset) /
		sizeof(InstructionContainer);
	
	uint64_t end = begin + symbol.size / sizeof(InstructionContainer);

	TargetSet targets;

	for(uint64_t i = begin; i != end; ++i)
	{
		const archaeopteryx::ir::InstructionContainer&
			instruction = _instructions[i];

		if(instruction.asInstruction.opcode == archaeopteryx::ir::Instruction::Bra)
		{
			if(instruction.asBra.target.asOperand.mode ==
				archaeopteryx::ir::Operand::Immediate)
			{
				targets.insert(instruction.asBra.target.asImmediate.uint);
			}
			else
			{
				assertM(false, "not implemented");
			}
		}
	}

	BasicBlockDescriptor block("BB_0", begin);

	for(uint64_t i = begin; i != end; ++i)
	{
		bool isTerminator = false;
		uint64_t blockEnd = i;

		const archaeopteryx::ir::InstructionContainer&
			instruction = _instructions[i];

		if(targets.count(i * sizeof(InstructionContainer)) != 0)
		{
			isTerminator = true;
		}
		else if(instruction.asInstruction.opcode == archaeopteryx::ir::Instruction::Bra)
		{
			isTerminator = true;
			blockEnd = i + 1;
		}

		if(isTerminator)
		{
			block.end = blockEnd;
			blocks.push_back(block);

			std::stringstream name;

			name << "BB_" << blocks.size();

			block = BasicBlockDescriptor(name.str(), blockEnd);
		}
	}

	if(block.begin != end)
	{
		block.end = end;
		blocks.push_back(block);
	}

	return blocks;
}

void BinaryReader::_addInstruction(ir::Function::iterator block,
	const InstructionContainer& container) const
{
	if(_addSimpleBinaryInstruction(block, container)) return;
	if(_addSimpleUnaryInstruction(block, container))  return;
	if(_addComplexInstruction(block, container))      return;

	assertM(false, "Translation for instruction not implemented.");
}

bool BinaryReader::_addSimpleBinaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container) const
{
	return false;
}

bool BinaryReader::_addSimpleUnaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container) const
{
	return false;
}

bool BinaryReader::_addComplexInstruction(ir::Function::iterator block,
	const InstructionContainer& container) const
{
	return false;
}

BinaryReader::BasicBlockDescriptor::BasicBlockDescriptor(const std::string& n, uint64_t b,
	uint64_t e)
: name(n), begin(b), end(e)
{

}

}

}

