/*! \file   ThreadGroup.h
	\date   Saturday Feburary 26, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the ThreadGroup class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/Thread.h>
#include <archaeopteryx/executive/interface/CudaUtilities.h>

namespace executive
{

/*! \brief A class representing the state of a thread group */
class ThreadGroup
{
public:
    Thread*       threads;
    const Kernel* kernel;
};

__device__ void executeThreadGroup(void* parameters)
{
    ThreadGroup* threadGroup = util::getParameter<ThreadGroup*>(parameters);
    unsigned int id   = threadIdx.x;
    unsigned int step = blockDim.x;

    unsigned int threadCount = threadGroup->kernel->threadsPerGroup;

    for(unsigned int i = id; i < threadCount; i += step)
    {
        Thread* thread = threadGroup->threads + i;
        
        thread->pc           = 0;
        thread->localMemory  = 0;
        thread->threadGroup  = threadGroup;
        thread->registerFile =
        util::SharedMemoryWrapper::at<Thread::Register>(i);
    }

    
}

}

