/*! \file   CoreSimThread.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread class.
*/

#pragma once

#include <archaeopteryx/ir/interface/Binary.h>

// Forward Declarations
namespace executive { class CoreSimBlock; }
namespace ir        { class Instruction;}

/*! \brief A namespace for program execution */
namespace executive
{
class CoreSimThread
{
    public:
        typedef ir::Binary::PC PC;
        typedef long long unsigned int Value;
        typedef long long signed int SValue;
        typedef long long unsigned int Address;
    public:
        __device__ CoreSimThread(CoreSimBlock* parentBlock = 0,
        	unsigned threadId = 0, unsigned priority = 1, bool barrier = false);
        __device__ PC executeInstruction(ir::Instruction*, PC);

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


