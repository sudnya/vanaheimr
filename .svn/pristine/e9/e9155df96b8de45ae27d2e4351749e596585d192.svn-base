/*! \file   CoreSimBlock.cpp
	\date   Sunday August, 7th 2011
	\author Sudnya Padalikar
		<mailsudnya@gmail.com>
	\brief  The implementation file for the Core simulator of the thread block class.
*/

#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/ir/interface/Instruction.h>
#include <archaeopteryx/executive/interface/CoreSimKernel.h>

namespace executive
{

__device__ void CoreSimBlock::setupCoreSimBlock(unsigned int blockId,
	unsigned int registers, const CoreSimKernel* kernel)
{
    m_blockState.blockId = blockId;
    m_blockState.registersPerThread = registers;
    m_kernel = kernel;
    
    printf("Setting up core sim block %p, %d threads, %d registers\n", this, m_blockState.threadsPerBlock, m_blockState.registersPerThread);

    m_registerFiles  = new Register[m_blockState.registersPerThread *
    	m_blockState.threadsPerBlock];
    m_sharedMemory   = new SharedMemory[m_blockState.sharedMemoryPerBlock];
    m_localMemory    = new LocalMemory[m_blockState.localMemoryPerThread];

    m_threads        = new CoreSimThread[m_blockState.threadsPerBlock];
    m_warp           = m_threads + threadIdx.x - getThreadIdInWarp();
    
    for(unsigned i = 0; i < m_blockState.threadsPerBlock; ++i)
    {
    	m_threads[i].setParentBlock(this);
    	m_threads[i].setThreadId(i);
    }
}

__device__ void CoreSimBlock::setupBinary(ir::Binary* binary)
{
    m_blockState.binary = binary;
}

__device__ bool CoreSimBlock::areAllThreadsFinished()
{
    //TODO evaluate some bool 'returned' for all threads
    bool finished = m_warp[getThreadIdInWarp()].finished;
    __shared__ bool tempFinished[WARP_SIZE];

    tempFinished[getThreadIdInWarp()] = finished;
    // barrier

    for (unsigned int i = 2; i < WARP_SIZE; i*=2)
    {
        if (getThreadIdInWarp() % i == 0)
        {
            finished = finished & tempFinished[getThreadIdInWarp() + i/2];
        }
        // barrier
        
        if (getThreadIdInWarp() % i == 0)
        {
            tempFinished[getThreadIdInWarp()] = finished;
        }

        // barrier
    }
    
    finished = tempFinished[0];

    return finished;
}

__device__ void CoreSimBlock::roundRobinScheduler()
{
    if (getThreadIdInWarp() == 0)
    {
    	unsigned int currentWarp = m_warp - m_threads;
    	cta_report("Running round robin scheduler, current warp is [%d, %d]\n",
    		currentWarp, currentWarp + WARP_SIZE);
        if (currentWarp + WARP_SIZE >= m_blockState.threadsPerBlock)
        {
            m_warp = m_threads;
        }
        else
        {
            m_warp += WARP_SIZE;
        }

    	cta_report(" selected warp [%d, %d]\n", (int)(m_warp - m_threads),
    		(int)(m_warp - m_threads) + WARP_SIZE);
    }
    //barrier
}

__device__ unsigned int CoreSimBlock::findNextPC(unsigned int& returnPriority)
{
    __shared__ uint2 priority[WARP_SIZE];
    unsigned int localThreadPriority = 0;
    unsigned int localThreadPC       = 0;

    // only give threads a non-zero priority if they are NOT waiting at a barrier
    if (m_warp[getThreadIdInWarp()].barrierBit == false)
    {
        localThreadPriority = m_warp[getThreadIdInWarp()].instructionPriority;
        localThreadPC       = m_warp[getThreadIdInWarp()].pc;

        priority[getThreadIdInWarp()].x = localThreadPriority;
        priority[getThreadIdInWarp()].y = localThreadPC;
    }
 
    device_report("FindNextPC for threadId %d, input priority %d, threadIdInWarp: %d \n", threadIdx.x, localThreadPriority, getThreadIdInWarp());
    
    // warp_barrier

    for (unsigned int i = 2; i < WARP_SIZE; i*=2)
    {
        if (getThreadIdInWarp() % i == 0)
        {
        	unsigned int neighborsThreadId = getThreadIdInWarp() + i/2;
            unsigned int neighborsPriority = priority[neighborsThreadId].x;
            unsigned int neighborsPC       = priority[neighborsThreadId].y;

            bool local = localThreadPriority > neighborsPriority;

            localThreadPriority = local ? localThreadPriority : neighborsPriority;
            localThreadPC       = local ? localThreadPC       : neighborsPC;
            device_report("\tThread [%d]: LocalThreadPriority: %d, neighborsPriority[%d]: %d \n", threadIdx.x, localThreadPriority, neighborsThreadId, neighborsPriority);
        }
        // warp_barrier
        if (getThreadIdInWarp() % i == 0)
        {
            priority[getThreadIdInWarp()].x = localThreadPriority;
            priority[getThreadIdInWarp()].y = localThreadPC;
        }
        // warp_barrier
    }

    unsigned int maxPriority = priority[0].x;
    unsigned int maxPC       = priority[0].y;
 
    cta_report(" max priority is %d, max pc is %d\n", maxPriority, maxPC);
 
    returnPriority = maxPriority;

    return maxPC;
}

__device__ bool CoreSimBlock::setPredicateMaskForWarp(PC pc)
{
    //TO DO - evaluate a predicate over the entire warp
    return pc == m_warp[getThreadIdInWarp()].pc;
}

__device__ CoreSimBlock::InstructionContainer CoreSimBlock::fetchInstruction(PC pc)
{
    __shared__ InstructionContainer instruction;
    
    if (getThreadIdInWarp() == 0)
    {
        m_blockState.binary->copyCode(&instruction, pc, 1);
    }
    // barrier
    return instruction;
}

__device__ void CoreSimBlock::executeWarp(InstructionContainer* instruction, PC pc)
{
    bool predicateMask = setPredicateMaskForWarp(pc);    
    
    //some function for all threads if predicateMask is true
    if (predicateMask)
    {
        PC newPC = m_warp[getThreadIdInWarp()].executeInstruction(
        	&instruction->asInstruction, pc);
        m_warp[getThreadIdInWarp()].pc = newPC;
        m_warp[getThreadIdInWarp()].instructionPriority = newPC + 1;
    }
}

__device__ unsigned int CoreSimBlock::getThreadIdInWarp()
{
    return (threadIdx.x % WARP_SIZE);
}

__device__ void CoreSimBlock::initializeSpecialRegisters()
{
    cta_report("Intializing special registers for %d threads\n", 
        m_blockState.threadsPerBlock);
    for(unsigned int tid = threadIdx.x; tid < m_blockState.threadsPerBlock;
        tid += blockDim.x)
    {
        // r32 is parameter memory (0x00000000 for now)
        setRegister(tid, 32, 0);
        // r33 is the global thread id 
        setRegister(tid, 33, tid);
    }

    cta_report(" done\n");
}

// Entry point to the block simulation
// It performs the following operations
//   1) Schedule group of simulated threads onto CUDA warps (static/round-robin)
//   2) Pick the next PC to execute (the one with the highest priority using a reduction)
//   3) Set the predicate mask (true if threadPC == next PC, else false)
//   4) Fetch the instruction at the selected PC
//   5) Execute all threads with true predicate masks
//   6) Save the new PC, goto 1 if all threads are not done
 __device__ void CoreSimBlock::runBlock()
{
    m_warp = m_threads + threadIdx.x - getThreadIdInWarp();

    initializeSpecialRegisters();

    cta_report("Running core-sim-block loop for simulated cta %d\n", 
        m_blockState.blockId);

    unsigned int executedCount  = 0;
    unsigned int scheduledCount = 0;
    unsigned int priority       = 1;

    while (!areAllThreadsFinished())
    {
        ++scheduledCount;
        PC nextPC = findNextPC(priority);

        cta_report(" next PC is %d, priority %d\n", (int)nextPC, priority);

        // only execute if all threads in this warp are NOT waiting on a barrier
        if (priority != 0)
        {
             InstructionContainer instruction = fetchInstruction(nextPC);
             executeWarp(&instruction, nextPC);
             ++executedCount;
        }

        if (scheduledCount == m_blockState.threadsPerBlock / WARP_SIZE)
        {
            if (executedCount == 0)
            {
                clearAllBarrierBits();
            }
            scheduledCount = 0;
            executedCount  = 0;
        }

        roundRobinScheduler();
    }
}

__device__ CoreSimThread::Value CoreSimBlock::getRegister(unsigned int threadId, unsigned int reg)
{
    Value v = m_registerFiles[(m_blockState.registersPerThread * threadId)+reg];

    device_report("(%d): reading register r%d, (%p)\n", threadId, reg, v);

    return v;
}

__device__ void CoreSimBlock::setRegister(unsigned int threadId,
	unsigned int reg, const CoreSimThread::Value& result)
{
    device_report("(%d): setting register r%d, (%p)\n",
        threadId, reg, result);

    m_registerFiles[(m_blockState.registersPerThread*threadId)+reg] = result;
}

__device__ CoreSimThread::Value CoreSimBlock::translateVirtualToPhysical(const CoreSimThread::Value v)
{
    return m_kernel->translateVirtualToPhysicalAddress(v);
}


__device__ void CoreSimBlock::barrier(unsigned int threadId)
{
    m_threads[threadId].barrierBit = true;
}

__device__ unsigned int CoreSimBlock::returned(unsigned int threadId,
	unsigned int pc)
{
    m_threads[threadId].finished = true;

    // TODO return the PC from the stack
    return 0;
}

__device__ void CoreSimBlock::clearAllBarrierBits()
{
    for (unsigned int i = 0 ; i < (m_blockState.threadsPerBlock)/WARP_SIZE ; ++i)
    {
        unsigned int logicalThread = i * WARP_SIZE + getThreadIdInWarp();
	m_threads[logicalThread].barrierBit = false;
        //barrier should be here but it is slow (every warp)
    } 
    //barrier -> we gurantee that we wont clobber values (blocks are not overlapping)
}

__device__ void CoreSimBlock::setNumberOfThreadsPerBlock(unsigned int threads)
{
    m_blockState.threadsPerBlock = threads;
}

__device__ void CoreSimBlock::setMemoryState(unsigned int localMemory, unsigned int sharedMemory)
{
    m_blockState.localMemoryPerThread = localMemory;
    m_blockState.sharedMemoryPerBlock = sharedMemory;
}

__device__ void CoreSimBlock::setBlockState(const BlockState& blockState)
{
    m_blockState = blockState;
}

}

