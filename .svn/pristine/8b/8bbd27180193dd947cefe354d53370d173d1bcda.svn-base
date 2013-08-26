/*! \file   Binary.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday September 9, 2011
	\brief  The source file the IR Binary class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/File.h>

#include <archaeopteryx/util/interface/debug.h>
#include <archaeopteryx/util/interface/cstring.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Instruction.h>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace ir
{

__device__ Binary::Binary(const char* filename)
: _file(0), _ownedFile(0)
{
	_ownedFile = new util::File(filename, "r");
	
	_file = _ownedFile;

	_loadHeader();
}

__device__ Binary::Binary(File* file)
: _file(file), _ownedFile(0)
{
	_loadHeader();
}

__device__ Binary::~Binary()
{
	device_report("Destroying binary.\n");

	device_report(" deleting copied pages from the binary file...\n");

	for(unsigned int c = 0; c != _header.codePages; ++c)
	{
		delete[] _codeSection[c];
	}
	
	for(unsigned int d = 0; d != _header.dataPages; ++d)
	{
		delete[] _dataSection[d];
	}
	
	for(unsigned int s = 0; s != _header.stringPages; ++s)
	{
		delete[] _stringSection[s];
	}
	
	device_report(" deleting symbol tables...\n");

	delete[] _symbolTable;
	delete[] _codeSection;
	delete[] _dataSection;
	delete[] _stringSection;
	
	device_report(" deleting the binary file...\n");
	delete _ownedFile;
	device_report(" finished...\n");
}

__device__ Binary::SymbolTableEntry* Binary::findSymbol(const char* name)
{
	_loadSymbolTable();
	
	for(unsigned int i = 0; i < _header.symbols; ++i)
	{
		SymbolTableEntry* symbol = _symbolTable + i;
		
		if(_strcmp(_header.stringsOffset + symbol->stringOffset, name) == 0)
		{
			return symbol;
		}
	}
	
	return 0;
}

__device__ void Binary::copyCode(InstructionContainer* code, PC pc,
	unsigned int instructions)
{
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(InstructionContainer);
	
	size_t page       = pc / instructionsPerPage;
	size_t pageOffset = pc % instructionsPerPage;
	
	device_report("Copying %d instructions at PC %d\n", instructions, pc);

	while(instructions > 0)
	{
		size_t instructionsInThisPage =
			util::min((size_t)(instructionsPerPage - pageOffset),
				(size_t)instructions);
	
		device_report(" copying %d instructions from page %d\n", 
			(int)instructionsInThisPage, (int)page);
		PageDataType* pageData = getCodePage(code_begin() + page);
		device_assert(pageData != 0);

		InstructionContainer* container =
			reinterpret_cast<InstructionContainer*>(pageData);
	
		util::memcpy(code, container + pageOffset,
			sizeof(InstructionContainer) * instructionsInThisPage);
	
		instructions -= instructionsInThisPage;
		pageOffset    = 0;
		page         += 1;

		device_report("  %d instructions are remaining\n", instructions);
	}
}

__device__ bool Binary::containsFunction(const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0) return false;
	
	return symbol->type == SymbolTableEntry::FunctionType;
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
	
	device_assert(symbol->type == SymbolTableEntry::FunctionType);
	
	page   = code_begin() + _getCodePageId(symbol->offset);
	offset = _getCodePageOffset(symbol->offset);
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
	
	device_assert(symbol->type == SymbolTableEntry::VariableType);
	
	page   = data_begin() + _getDataPageId(symbol->offset);
	offset = _getDataPageOffset(symbol->offset);
}

__device__ util::string Binary::getSymbolName(SymbolTableEntry* symbol)
{
	unsigned int length = _strlen(_header.stringsOffset + symbol->stringOffset);
	
	util::string name(length, '\0');
	
	_strcpy((char*)name.data(), _header.stringsOffset + symbol->stringOffset);
	
	return name;
}

__device__ util::string Binary::getSymbolName(unsigned int offset)
{
	SymbolTableEntry* symbol = _symbolTable +
		(offset - _header.symbolOffset) / sizeof(SymbolTableEntry);

	return getSymbolName(symbol);
}

__device__ size_t Binary::getSymbolSize(const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0) return 0;
	
	return symbol->size;
}
	
__device__ util::string Binary::getSymbolDataAsString(const char* symbolName)
{
	device_report("   getting data for symbol '%s'\n", symbolName);
	
	SymbolTableEntry* symbol = findSymbol(symbolName);
	
	device_assert(symbol != 0);
	
	device_assert(symbol->type == SymbolTableEntry::VariableType);
	
	device_report("    data size is '%d' bytes\n", symbol->size);
	
	util::string result(symbol->size, '\0');
	
	_datacpy((char*)result.data(), symbol->offset, symbol->size);
	
	return result;
}

__device__ void Binary::copySymbolDataToAddress(void* address,
	const char* symbolName)
{
	device_report("   copying data for symbol '%s'\n", symbolName);
	
	SymbolTableEntry* symbol = findSymbol(symbolName);
	
	device_assert(symbol != 0);
	
	device_assert(symbol->type == SymbolTableEntry::VariableType);
	
	device_report("    data size is '%d' bytes\n", symbol->size);
	
	_datacpy((char*)address, symbol->offset, symbol->size);
}

__device__ Binary::StringVector Binary::getSymbolNamesThatMatch(
	const char* substring)
{
	_loadSymbolTable();
	
	StringVector matches;
	
	for(unsigned int i = 0; i < _header.symbols; ++i)
	{
		SymbolTableEntry* symbol = _symbolTable + i;
		
		util::string name = getSymbolName(symbol);
		
		if(name.find(substring) == 0)
		{
			matches.push_back(name);
		}
	}
	
	return matches;
}

__device__ void Binary::copyDataToAddress(void* address, uint64_t offset,
	uint64_t bytes)
{
	_datacpy((char*)address, _header.dataOffset + offset, bytes);
}

__device__ Binary::PC Binary::findFunctionsPC(const char* name)
{
	page_iterator page  = 0;
	unsigned int offset = 0;

	findFunction(page, offset, name);
	
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(InstructionContainer);
	
	return instructionsPerPage * (page - code_begin()) + offset;
}

__device__ Binary::page_iterator Binary::code_begin()
{
	return _codeSection;
}

__device__ Binary::page_iterator Binary::code_end()
{
	return _codeSection + _header.codePages;
}

__device__ Binary::page_iterator Binary::data_begin()
{
	return _dataSection;
}

__device__ Binary::page_iterator Binary::data_end()
{
	return _dataSection + _header.dataPages;
}

__device__ Binary::page_iterator Binary::string_begin()
{
	return _stringSection;
}

__device__ Binary::page_iterator Binary::string_end()
{
	return _stringSection + _header.stringPages;
}

__device__ Binary::PageDataType* Binary::getCodePage(page_iterator page)
{
	while(*page == 0)
	{
		if(!_lock(page)) continue;
	
		size_t offset = _getCodePageOffset(page);

		device_report("Loading code page (%p) at offset (%p) now...\n",
			page, offset);
	
		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));

		_unlock(page);
	
		break;
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getDataPage(page_iterator page)
{
	while(*page == 0)
	{
		if(!_lock(page)) continue;
		
		size_t offset = _getDataPageOffset(page);
	
		device_report("Loading data page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
		
		_unlock(page);

		break;
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getStringPage(page_iterator page)
{
	device_assert(page < string_end());

	while(*page == 0)
	{
		if(!_lock(page)) continue;
		
		size_t offset = _getStringPageOffset(page);

		device_report("Loading string page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
		
		_unlock(page);
		
		break;
	}
	
	return *page;
}


__device__ void Binary::_loadHeader()
{
	_file->read(&_header, sizeof(Header));
	
	device_report("Loading header (%p magic)\n", _header.magic);
	
	device_assert(_header.magic == Header::MagicNumber);
	
	_dataSection   = new PagePointer[_header.dataPages];
	_codeSection   = new PagePointer[_header.codePages];
	_stringSection = new PagePointer[_header.stringPages];

	_symbolTable = 0;

	util::memset(_dataSection,   0, _header.dataPages   * sizeof(PagePointer));
	util::memset(_codeSection,   0, _header.codePages   * sizeof(PagePointer));
	util::memset(_stringSection, 0, _header.stringPages * sizeof(PagePointer));
	
	for(page_iterator page = code_begin(); page != code_end(); ++page)
	{
		_locks.insert(util::make_pair(page, Lock()));
	}

	for(page_iterator page = data_begin(); page != data_end(); ++page)
	{
		_locks.insert(util::make_pair(page, Lock()));
	}

	for(page_iterator page = string_begin(); page != string_end(); ++page)
	{
		_locks.insert(util::make_pair(page, Lock()));
	}
	
	device_report("Loaded binary (%d data pages, %d code pages, "
		"%d symbols, %d string pages)\n", _header.dataPages, _header.codePages,
		_header.symbols, _header.stringPages);
}

__device__ void Binary::_loadSymbolTable()
{
	if(_header.symbols == 0) return;
	if(_symbolTable != 0)    return;

	device_report(" Loading symbol table now.\n");

	_symbolTable = new SymbolTableEntry[_header.symbols];
	
	device_report("  symbol table offset %d.\n", (int)_header.symbolOffset);
	device_assert(_file != 0);

	_file->seekg(_header.symbolOffset);

	device_report("  loading symbol table now.\n");

	_file->read(_symbolTable, _header.symbols * sizeof(SymbolTableEntry));

	device_report("   loaded %d symbols...\n", _header.symbols);
}

__device__ size_t Binary::_getCodePageOffset(page_iterator page)
{
	return _header.codeOffset +	(page - code_begin()) * sizeof(PageDataType);
}

__device__ size_t Binary::_getDataPageOffset(page_iterator page)
{
	return _header.dataOffset + (page - data_begin()) * sizeof(PageDataType);
}

__device__ size_t Binary::_getStringPageOffset(page_iterator page)
{
	return _header.stringsOffset +
		(page - string_begin()) * sizeof(PageDataType);
}

__device__ int Binary::_strcmp(unsigned int stringTableOffset,
	const char* string)
{
	page_iterator page  = string_begin() + _getStringPageId(stringTableOffset);
	unsigned int offset = _getStringPageOffset(stringTableOffset);

	for(; page != string_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getStringPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++string)
		{
			if(data[offset] != *string)
			{
				return -1;
			}
			
			if(data[offset] == '\0')
			{
				if(*string == '\0')
				{
					return 0;
				}
				
				return -1;
			}
			else if(*string == '\n')
			{
				return -1;
			}
		}
	}
	
	return 0;
}

__device__ void Binary::_datacpy(char* string, unsigned int dataOffset,
	unsigned int size)
{
	page_iterator page  = data_begin() + _getDataPageId(dataOffset);
	unsigned int offset = _getDataPageOffset(dataOffset);
	unsigned int bytesCopied = 0;

	//device_report("copying data from file offset (0x%x) to (%p)\n",
	//	dataOffset, string);
	
	for(; page != data_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getDataPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++string)
		{
			if(bytesCopied++ >= size)
			{
				device_report(" copied %d bytes\n", bytesCopied);
				return;
			}
			
			*string = data[offset];
		}
	}
}

__device__ void Binary::_strcpy(char* string, unsigned int stringTableOffset)
{
	page_iterator page  = string_begin() + _getStringPageId(stringTableOffset);
	unsigned int offset = _getStringPageOffset(stringTableOffset);

	//device_report("copying string from file offset (0x%x) to (%p)\n",
	//	stringTableOffset, string);
	
	for(; page != string_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getStringPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++string)
		{
			if(data[offset] == '\0') return;
			
			*string = data[offset];
		}
	}
}

__device__ int Binary::_strlen(unsigned int stringTableOffset)
{
	page_iterator page  = string_begin() + _getStringPageId(stringTableOffset);
	unsigned int offset = _getStringPageOffset(stringTableOffset);

	unsigned int length = 0;
	
	for(; page != string_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getStringPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++length)
		{
			if(data[offset] == '\0')
			{
				return length;
			}
		}
	}
	
	return length;
}

__device__ unsigned int Binary::_getCodePageId(size_t offset)
{
	device_assert(offset >= _header.codeOffset);
	
	size_t codeOffset = offset - _header.codeOffset;
	
	unsigned int page = codeOffset / sizeof(PageDataType);

	device_assert(page < _header.codePages);

	return page;
}

__device__ unsigned int Binary::_getCodePageOffset(size_t offset)
{
	device_assert(offset >= _header.codeOffset);
	
	size_t codeOffset = offset - _header.codeOffset;
	
	return codeOffset % sizeof(PageDataType);
}

__device__ unsigned int Binary::_getDataPageId(size_t offset)
{
	device_assert(offset >= _header.dataOffset);
	
	size_t dataOffset = offset - _header.dataOffset;
	
	unsigned int page = dataOffset / sizeof(PageDataType);

	device_assert(page < _header.dataPages);

	return page;	
}

__device__ unsigned int Binary::_getDataPageOffset(size_t offset)
{
	device_assert(offset >= _header.dataOffset);
	
	size_t dataOffset = offset - _header.dataOffset;
	
	return dataOffset % sizeof(PageDataType);
}

__device__ unsigned int Binary::_getStringPageId(size_t offset)
{
	device_assert(offset >= _header.stringsOffset);
	
	size_t stringsOffset = offset - _header.stringsOffset;
	
	unsigned int page = stringsOffset / sizeof(PageDataType);

	device_assert(page < _header.stringPages);

	return page;
}

__device__ unsigned int Binary::_getStringPageOffset(size_t offset)
{
	device_assert(offset >= _header.stringsOffset);
	
	size_t stringsOffset = offset - _header.stringsOffset;
	
	return stringsOffset % sizeof(PageDataType);
}

__device__ bool Binary::_lock(page_iterator page)
{
	LockMap::iterator lock = _locks.find(page);
	
	device_assert(lock != _locks.end());
	
	return lock->second.lock();
}

__device__ bool Binary::_unlock(page_iterator page)
{
	LockMap::iterator lock = _locks.find(page);
	
	device_assert(lock != _locks.end());
	
	return lock->second.unlock();
}

__device__ Binary::Lock::Lock()
{
	_lock = 0xffffffff;
}

__device__ bool Binary::Lock::lock()
{
	unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int original = atomicCAS(&_lock, 0xffffffff, gid);

	return original == 0xffffffff;
}

__device__ bool Binary::Lock::unlock()
{
	unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int original = atomicCAS(&_lock, gid, 0xffffffff);

	return original == gid;
}

}

}

