/*! \file   CoreSimKernel.h
 *  \date   Thursday September 15, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  The implementation file for the Core simulator of the Kernel class.
 *   */

#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/runtime/interface/Runtime.h>
#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/Knob.h>

namespace archaeopteryx
{

namespace executive
{

__device__ void CoreSimKernel::launchKernel(CoreSimBlock* blocks,
	ir::Binary* binary)
{
	unsigned int registerCount = util::KnobDatabase::getKnob<unsigned int>(
			"simulator-registers-per-thread");

	for (unsigned int simulatedBlock = blockIdx.x;
		simulatedBlock < simulatedBlocks; simulatedBlock += gridDim.x)
	{
		if(threadIdx.x == 0)
		{
			blocks[blockIdx.x].setupBinary(binary);
			blocks[blockIdx.x].setupCoreSimBlock(simulatedBlock,
				registerCount, this);
		}

		__syncthreads();

		blocks[blockIdx.x].runBlock();
	}
}

__device__ CoreSimKernel::Address
	CoreSimKernel::translateVirtualToPhysicalAddress(Address va) const
{
    return rt::Runtime::translateVirtualToPhysicalAddress(va);
}

}

}

