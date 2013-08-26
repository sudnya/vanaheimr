/*! \file   CoreSimThread.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/IntTypes.h>

// Forward Declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }
namespace vanaheimr     { namespace as        { class Instruction;  } }

namespace archaeopteryx
{

namespace executive
{
class CoreSimThread
{
    public:
		typedef vanaheimr::as::Instruction Instruction;
        typedef ir::Binary Binary;
		typedef Binary::PC PC;
        
		typedef uint64_t Value;
        typedef int64_t  SValue;
        typedef float    FValue;
        typedef double   DValue;
        
        typedef long long unsigned int Address;
    
	public:
        __device__ CoreSimThread(CoreSimBlock* parentBlock = 0,
        	unsigned threadId = 0, unsigned priority = 1, bool barrier = false);
        __device__ PC executeInstruction(Instruction*, PC);

	public:
		__device__ void setParentBlock(CoreSimBlock* parentBlock);
		__device__ void setThreadId(unsigned id);

    public:
        PC   pc;
        bool finished;
        unsigned instructionPriority;
        bool barrierBit; //we may later want to support multiple barriers

    private:
        CoreSimBlock* m_parentBlock;
        unsigned m_tId;
};

}


}

