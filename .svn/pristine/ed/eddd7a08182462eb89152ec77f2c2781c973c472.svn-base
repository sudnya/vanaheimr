/*! \file   Binary.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday September 9, 2011
	\brief  The source file the IR Binary class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Binary.h>
#include <archaeopteryx/ir/interface/Instruction.h>

#include <archaeopteryx/util/interface/File.h>
#include <archaeopteryx/util/interface/StlFunctions.h>

#include <archaeopteryx/util/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace ir
{

__device__ Binary::Binary(File* file)
: _file(file)
{
	Header header;

	_file->read(&header, sizeof(Header));

	dataPages          = header.dataPages;
	codePages          = header.codePages;
	symbolTableEntries = header.symbols;
	stringTableEntries = header.strings;
	
	dataSection = new PageDataType*[dataPages];
	codeSection = new PageDataType*[codePages];
	symbolTable = 0;
	stringTable = 0;

	std::memset(dataSection, 0, dataPages * sizeof(PageDataType*));
	std::memset(codeSection, 0, codePages * sizeof(PageDataType*));

	device_report("Loaded binary (%d data pages, %d code pages, "
		"%d symbols, %d strings)\n", dataPages, codePages, symbolTableEntries, 
		stringTableEntries);
}

__device__ Binary::~Binary()
{
	for(unsigned int c = 0; c != codePages; ++c)
	{
		delete[] codeSection[c];
	}
	
	for(unsigned int d = 0; d != dataPages; ++d)
	{
		delete[] dataSection[d];
	}
	
	delete[] stringTable;
	delete[] symbolTable;
	delete[] codeSection;
	delete[] dataSection;
}

__device__ Binary::PageDataType* Binary::getCodePage(page_iterator page)
{
	if(*page == 0)
	{
		size_t offset = _getCodePageOffset(page);

		device_report("Loading code page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getDataPage(page_iterator page)
{
	if(*page == 0)
	{
		size_t offset = _getDataPageOffset(page);

		device_report("Loading data page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::SymbolTableEntry* Binary::findSymbol(const char* name)
{
	_loadSymbolTable();
	
	for(unsigned int i = 0; i < symbolTableEntries; ++i)
	{
		SymbolTableEntry* symbol = symbolTable + i;
		const char* symbolName   = symbol->stringTableOffset + stringTable;
	
		if(util::strcmp(symbolName, name) != 0)
		{
			return symbol;
		}
	}
	
	return 0;
}

__device__ void Binary::findFunction(page_iterator& page, unsigned int& offset,
	const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0)
	{
		page   = 0;
		offset = 0;
		
		return;
	}
	
	device_assert(symbol->type == FunctionSymbolType);
	
	page   = codeSection + symbol->pageId;
	offset = symbol->pageOffset;
}

__device__ void Binary::findVariable(page_iterator& page, unsigned int& offset,
	const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0)
	{
		page   = 0;
		offset = 0;
		
		return;
	}
	
	device_assert(symbol->type == VariableSymbolType);
	
	page   = dataSection + symbol->pageId;
	offset = symbol->pageOffset;
}

__device__ Binary::PC Binary::findFunctionsPC(const char* name)
{
	page_iterator page  = 0;
	unsigned int offset = 0;

	findFunction(page, offset, name);
	
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(ir::InstructionContainer);
	
	return instructionsPerPage * (page - code_begin()) + offset;
}

__device__ Binary::page_iterator Binary::code_begin()
{
	return codeSection;
}

__device__ Binary::page_iterator Binary::code_end()
{
	return codeSection + codePages;
}

__device__ Binary::page_iterator Binary::data_begin()
{
	return dataSection;
}

__device__ Binary::page_iterator Binary::data_end()
{
	return dataSection + dataPages;
}

__device__ void Binary::copyCode(ir::InstructionContainer* code, PC pc,
	unsigned int instructions)
{
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(ir::InstructionContainer);
	
	size_t page       = pc / instructionsPerPage;
	size_t pageOffset = pc % instructionsPerPage;
	
	device_report("Copying %d instructions at PC %d\n", instructions, pc);

	while(instructions > 0)
	{
		size_t instructionsInThisPage =
			util::min(instructionsPerPage - pageOffset, (size_t)instructions);
	
		device_report(" copying %d instructions from page %d\n", 
			(int)instructionsInThisPage, (int)page);
		PageDataType* pageData = getCodePage(code_begin() + page);
		device_assert(pageData != 0);

		ir::InstructionContainer* container =
			reinterpret_cast<ir::InstructionContainer*>(pageData);
	
		std::memcpy(code, container + pageOffset,
			sizeof(ir::InstructionContainer) * instructionsInThisPage);
	
		instructions -= instructionsInThisPage;
		pageOffset    = 0;
		page         += 1;

		device_report("  %d instructions are remaining\n", instructions);
	}
}

size_t Binary::_getCodePageOffset(page_iterator page)
{
	return _getDataPageOffset(data_begin() + dataPages) +
		(page - code_begin()) * sizeof(PageDataType);
}

size_t Binary::_getDataPageOffset(page_iterator page)
{
	return sizeof(Header) + (page - data_begin()) * sizeof(PageDataType);
}

void Binary::_loadSymbolTable()
{
	if(symbolTableEntries == 0) return;
	if(symbolTable != 0)        return;

	device_report(" Loading symbol/string table now.\n");

	stringTable = new char[stringTableEntries];
	symbolTable = new SymbolTableEntry[symbolTableEntries];
	
	size_t symbolTableOffset = _getCodePageOffset(code_begin() + codePages);
	size_t stringTableOffset = symbolTableOffset +
		symbolTableEntries * sizeof(SymbolTableEntry);

	device_report("  symbol table offset %d, string table offset %d.\n", 
		(int)symbolTableOffset, (int)stringTableOffset);
	device_assert(_file != 0);

	_file->seekg(symbolTableOffset);

	device_report("  loading symbol table now.\n");

	_file->read(symbolTable, symbolTableEntries * sizeof(SymbolTableEntry));

	device_report("   loaded %d symbols...\n", symbolTableEntries);
	
	_file->seekg(stringTableOffset);

	device_report("  loading string table now.\n");

	_file->read(stringTable, stringTableEntries);

	device_report("   loaded %d bytes of strings...\n", stringTableEntries);
}

}


