/*! \file   FetchUnit.h
	\date   Saturday April 23, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	        Sudnya  Diamos <mailsudnya@gmail.com>
	\brief  The header file for the FetchUnit class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeoteryx/executive/interface/Binary.h>

namespace executive
{

/*! \brief The simulator's interface to a binary file */
class FetchUnit
{
public:
	/*! \brief Create a new fetch unit */
	__device__ FetchUnit(ir::Binary* binary);

public:
	/*! \brief Set the size of the cache and the memory that should be used */
	__device__ void setCache(const void* cache, unsigned int size);
	
public:
	/*! \brief Given a PC, return the instruction container */
	__device__ const ir::InstructionContainer*
		getInstruction(ir::Binary::PC pc);

private:
	/*! \brief The cache array */
	const ir::InstructionContainer* _cache;
	/*! \brief The cache size in instruction container units */
	unsigned int _cacheSize;
	/*! \brief The PC stored in the cache */
	ir::Binary::PC _tag;

	/*! \brief A pointer to the binary being fetched from */
	ir::Binary* _binary;
};

}

