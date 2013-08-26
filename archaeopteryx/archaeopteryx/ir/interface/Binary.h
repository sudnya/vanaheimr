/*! \file   Binary.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday February 27, 2011
	\brief  The header file the IR Binary class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/string.h>
#include <archaeopteryx/util/interface/vector.h>
#include <archaeopteryx/util/interface/map.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryHeader.h>
#include <vanaheimr/asm/interface/SymbolTableEntry.h>

#include <vanaheimr/util/interface/IntTypes.h>

// Forward Declarations
namespace archaeopteryx { namespace util { class File;                 } }
namespace vanaheimr     { namespace as   { class InstructionContainer; } }

namespace archaeopteryx
{

namespace ir
{

/*! \brief A class representing a VIR binary, lazy loading is handled here */
class Binary
{
public:
	/*! \brief A generic type for a vector of strings */
	typedef util::vector<util::string> StringVector;

	/*! \brief An instruction object stored in the binary */
	typedef vanaheimr::as::InstructionContainer InstructionContainer;

	/*! \brief a 64-bit program counter */
	typedef uint64_t PC;
	/*! \brief a file handle */
	typedef util::File File;

	/*! \brief A binary header */
	typedef vanaheimr::as::BinaryHeader     Header;
	typedef vanaheimr::as::SymbolTableEntry SymbolTableEntry;

	/*! \brief A 32-KB page */
	static const unsigned int PageSize = Header::PageSize / sizeof(uint32_t);
	
	typedef uint32_t PageDataType[PageSize];
	
	/*! \brief A symbol table iterator */
	typedef SymbolTableEntry* symbol_table_iterator;

	/*! \brief A page iterator */
	typedef PageDataType* PagePointer;
	typedef PagePointer* page_iterator;

public:
	/*! \brief Construct a binary from a file name */
	__device__ Binary(const char* filename);
	/*! \brief Construct a binary from an open file */
	__device__ Binary(File* file);
	/*! \brief Destroy the binary, free all memory */
	__device__ ~Binary();

public:
	/*! \brief Copy code from a PC */
	__device__ void copyCode(InstructionContainer* code, PC pc,
		unsigned int instructions);
	/*! \brief Does a named funtion exist? */
	__device__ bool containsFunction(const char* name);
	/*! \brief Get PC */
	__device__ PC findFunctionsPC(const char* name);

public:	
	/*! \brief Find a symbol by name */
	__device__ SymbolTableEntry* findSymbol(const char* name);
	/*! \brief Find a function by name */
	__device__ void findFunction(page_iterator& page, unsigned int& offset,
		const char* name);
	/*! \brief Find a variable by name */
	__device__ void findVariable(page_iterator& page, unsigned int& offset,
		const char* name);

public:
	/*! \brief Get the name of a symbol */
	__device__ util::string getSymbolName(unsigned int symbolTableOffset);
	/*! \brief Get the name of a symbol */
	__device__ util::string getSymbolName(SymbolTableEntry* symbol);
	/*! \brief Get the size of a symbol */
	__device__ size_t getSymbolSize(const char* name);
	/*! \brief Find a symbol by name and return its data as a string */
	__device__ util::string getSymbolDataAsString(const char* symbolName);
	/*! \brief Find a symbol by name and copy its data to an address */
	__device__ void copySymbolDataToAddress(void* address,
		const char* symbolName);
	/*! \brief Get symbol names that match a substring */
	__device__ StringVector getSymbolNamesThatMatch(const char* substring);

public:
	/*! \brief Copy from the data section to an address */
	__device__ void copyDataToAddress(void* address, uint64_t offset,
		uint64_t bytes);

public:
	/*! \brief Get an iterator to the first code page */
	__device__ page_iterator code_begin();
	/*! \brief Get an iterator to one past the last code page */
	__device__ page_iterator code_end();

	/*! \brief Get an iterator to the first data page */
	__device__ page_iterator data_begin();
	/*! \brief Get an iterator to one past the last data page */
	__device__ page_iterator data_end();

	/*! \brief Get an iterator to the first string page */
	__device__ page_iterator string_begin();
	/*! \brief Get an iterator to one past the last string page */
	__device__ page_iterator string_end();

private:

	/*! \brief Get a particular code page */
	__device__ PageDataType* getCodePage(page_iterator page);
	/*! \brief Get a pointer to a particular data page */
	__device__ PageDataType* getDataPage(page_iterator page);
	/*! \brief Get a pointer to a particular string page */
	__device__ PageDataType* getStringPage(page_iterator page);


private:
	/*! \brief Load the binary header */
	__device__ void _loadHeader();

	/*! \brief Load the symbol table */
	__device__ void _loadSymbolTable();

	/*! \brief Get an offset in the file for a specific code page */
	__device__ size_t _getCodePageOffset(page_iterator page);
	/*! \brief Get an offset in the file for a specific data page */
	__device__ size_t _getDataPageOffset(page_iterator page);
	/*! \brief Get an offset in the file for a specific string page */
	__device__ size_t _getStringPageOffset(page_iterator page);


private:
	__device__ int _strcmp(unsigned int stringTableOffset, const char* string);
	__device__ void _strcpy(char* string, unsigned int stringTableOffset);
	__device__ int _strlen(unsigned int stringTableOffset);
	__device__ void _datacpy(char* string, unsigned int dataOffset,
		unsigned int size);

private:
	/*! \brief Get the page number for a specific offset in the file */
	__device__ unsigned int _getCodePageId(size_t offset);
	/*! \brief Get the page offset for a specific offset in the file */
	__device__ unsigned int _getCodePageOffset(size_t offset);
	/*! \brief Get the page number for a specific offset in the file */
	__device__ unsigned int _getDataPageId(size_t offset);
	/*! \brief Get the page offset for a specific offset in the file */
	__device__ unsigned int _getDataPageOffset(size_t offset);
	/*! \brief Get the page number for a specific offset in the file */
	__device__ unsigned int _getStringPageId(size_t offset);
	/*! \brief Get the page offset for a specific offset in the file */
	__device__ unsigned int _getStringPageOffset(size_t offset);

private:
	/*! \brief Attempt to lock a page */
	__device__ bool _lock(page_iterator);
	/*! \brief Attempt to unlock a page */
	__device__ bool _unlock(page_iterator);
	

private:
	/*! \brief A handle to the file */
	File* _file;
	/*! \brief A handle to a file owned by this binary */
	File* _ownedFile;

private:
	/*! \brief The header loaded from the file */
	Header _header;

private:
	/*! \brief The list of data pages, lazily allocated */
	PagePointer* _dataSection;
	/*! \brief The list of instruction pages, lazily allocated */
	PagePointer* _codeSection;
	/*! \brief The list of string pages, lazily allocated */
	PagePointer* _stringSection;

	/*! \brief The actual symbol table */
	SymbolTableEntry* _symbolTable;

private:
	class Lock
	{
	public:
		/*! \brief Construct the lock in the unlocked state */
		__device__ Lock();
	
	public:
		/*! \brief attempt to aquire the lock (may fail) */
		__device__ bool lock();
		/*! \brief attempt to release the lock */
		__device__ bool unlock();
	
	private:
		unsigned int _lock;	
	
	};

	typedef util::map<page_iterator, Lock> LockMap;

private:
	LockMap _locks;


};

}

}

