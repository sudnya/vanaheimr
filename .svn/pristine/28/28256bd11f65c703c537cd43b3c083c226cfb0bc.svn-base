/*! \file   FetchUnit.cpp
	\date   Tuesday April 26, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	        Sudnya  Diamos <mailsudnya@gmail.com>
	\brief  The source file for the FetchUnit class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeoteryx/executive/interface/FetchUnit.h>

namespace executive
{

__device__ FetchUnit::FetchUnit(ir::Binary* b)
: _cache(0), _cacheSize(0), _tag(-1), _binary(b)
{

}

__device__ void FetchUnit::setCache(const void* cache, unsigned int size)
{
	unsigned int elements = size / sizeof(ir::InstructionContainer);
	
	_cache = const ir::InstructionContainer* cache;
	_cacheSize = elements;
	_tag = -1;
}
	
__device__ const ir::InstructionContainer*
	FetchUnit::getInstruction(ir::Binary::PC pc)
{
	bool hit = pc >= _tag && pc < _tag + _cacheSize;

	// If we hit the cache, we are done
	if(hit) return _cache + pc - _tag;

	// Otherwise we trigger a miss
	ir::Binary::page_iterator page = 0;
	unsigned int offset = 0;
	_binary->copyCode(_cache, pc, _cacheSize);
	_tag = pc;
	
	return _cache;
}

}

