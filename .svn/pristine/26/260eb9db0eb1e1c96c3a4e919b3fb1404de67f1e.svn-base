/*! \file   Binary.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday February 27, 2011
	\brief  The header file the IR Binary class.
*/

#pragma once

// Forward Declarations
namespace util { class File;                 }
namespace ir   { union InstructionContainer; }


namespace ir
{

/*! \brief A class representing a VIR binary, lazy loading is handled here */
class Binary
{
public:
	/*! \brief 32-bit unsigned int */
	typedef unsigned int uint32;
	/*! \brief 64-bit unsigned int */
	typedef long long unsigned int uint64;
	/*! \brief a 64-bit program counter */
	typedef uint64 PC;
	/*! \brief a file handle */
	typedef util::File File;

	/*! \brief A binary header */
	class Header
	{
	public:
		uint32 dataPages;
		uint32 codePages;
		uint32 symbols;
		uint32 strings;
	};

	/*! \brief A 32-KB page */
	typedef uint32 PageDataType[1 << 13];
	
	/*! \brief A symbol type */
	enum SymbolType
	{
		VariableSymbolType   = 0x1,
		FunctionSymbolType   = 0x2,
		ArgumentSymbolType   = 0x3,
		BasicBlockSymbolType = 0x4,
		InvalidSymbolType    = 0x0
	};

	/*! \brief A symbol attribute */
	enum SymbolAttribute
	{
		InvalidAttribute = 0x0	
	};

	/*! \brief A table mapping symbols to pages and offsets */
	class SymbolTableEntry
	{
	public:
		/*! \brief The type of symbol */
		uint32 type;
		/*! \brief The offset in the string table of the name */
		uint32 stringTableOffset;
		/*! \brief The page id it is stored in */
		uint32 pageId;
		/*! \brief The offset within the page */
		uint32 pageOffset;
		/*! \brief The set of attributes */
		uint64 attributes;
	};

	/*! \brief A symbol table iterator */
	typedef SymbolTableEntry* symbol_table_iterator;

	/*! \brief A page iterator */
	typedef PageDataType** page_iterator;

public:
	/*! \brief Construct a binary from an open file */
	__device__ Binary(File* file);
	/*! \brief Destroy the binary, free all memory */
	__device__ ~Binary();

public:
	/*! \brief Get a particular code page */
	__device__ PageDataType* getCodePage(page_iterator page);
	/*! \brief Get a pointer to a particular data page */
	__device__ PageDataType* getDataPage(page_iterator page);

	/*! \brief Find a symbol by name */
	__device__ SymbolTableEntry* findSymbol(const char* name);
	/*! \brief Find a function by name */
	__device__ void findFunction(page_iterator& page, unsigned int& offset,
		const char* name);
	/*! \brief Find a variable by name */
	__device__ void findVariable(page_iterator& page, unsigned int& offset,
		const char* name);

public:
	/*! \brief Get PC */
	__device__ PC findFunctionsPC(const char* name);

public:
	/*! \brief Get an iterator to the first code page */
	__device__ page_iterator code_begin();
	/*! \brief Get an iterator to one past the last code page */
	__device__ page_iterator code_end();

	/*! \brief Get an iterator to the first data page */
	__device__ page_iterator data_begin();
	/*! \brief Get an iterator to one past the last data page */
	__device__ page_iterator data_end();

public:
	/*! \brief Copy code from a PC */
	__device__ void copyCode(ir::InstructionContainer* code, PC pc,
		unsigned int instructions);

public:
	/*! \brief The number of pages in the data section */
	unsigned int dataPages;
	/*! \brief The list of data pages, lazily allocated */
	PageDataType** dataSection;
	/*! \brief The number of pages in the code section */
	unsigned int codePages;
	/*! \brief The list of instruction pages, lazily allocated */
	PageDataType** codeSection;
	/*! \brief The number of symbol table entries */
	unsigned int symbolTableEntries;
	/*! \brief The actual symbol table */
	SymbolTableEntry* symbolTable;
	/*! \brief The string table */
	char* stringTable;
	/*! \brief The number of string table entries */
	unsigned int stringTableEntries;

private:
	/*! \brief Get an offset in the file for a specific code page */
	__device__ size_t _getCodePageOffset(page_iterator page);
	/*! \brief Get an offset in the file for a specific data page */
	__device__ size_t _getDataPageOffset(page_iterator page);
	/*! \brief Load the symbol and string tables */
	__device__ void _loadSymbolTable();

private:
	/*! \brief A handle to the file */
	File* _file;

};

}

#include <archaeopteryx/ir/implementation/Binary.cpp>


