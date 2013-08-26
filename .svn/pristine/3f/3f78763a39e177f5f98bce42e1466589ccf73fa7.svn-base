/*! \file   Runtime.cpp
 *   \date   Tuesday Sept, 13th 2011
 *   \author Sudnya Padalikar
 *   <mailsudnya@gmail.com>
 *   \brief  The implementation file for the runtime API.
 **/

#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/runtime/interface/Runtime.h>

#define NUMBER_OF_HW_THREADS_PER_BLOCK        32 
#define NUMBER_OF_HW_BLOCKS                   64
#define PHYSICAL_MEMORY_SIZE           (1 << 14)
#define PARAMETER_MEMORY_SIZE          (1 << 10)

__device__ rt::Runtime::RuntimeState g_runtimeState;

namespace rt
{

__device__ void Runtime::create()
{
	printf("Creating runtime with %d blocks, %d bytes of memory\n", 
		NUMBER_OF_HW_BLOCKS, PHYSICAL_MEMORY_SIZE);
    
    g_runtimeState.m_kernel         = new executive::CoreSimKernel;
    g_runtimeState.m_blocks         =
    	new executive::CoreSimBlock[NUMBER_OF_HW_BLOCKS];
    g_runtimeState.m_physicalMemory = malloc(PHYSICAL_MEMORY_SIZE);
    g_runtimeState.m_loadedBinary   = 0;
}

__device__ void Runtime::destroy()
{
   delete []g_runtimeState.m_blocks;
   delete g_runtimeState.m_loadedBinary;
   delete g_runtimeState.m_kernel;
}

// We will need a list/map of open binaries
//  a) maybe just one to start with
//  b) create a binary object using the filename in the constructor, it
//     will read the data for us
__device__ void Runtime::loadBinary(const char* fileName)
{
    g_runtimeState.m_loadedBinary = new ir::Binary(new util::File(fileName));
    //TODO: eventually m_loadedBinary.push_back(new Binary(fileName));
}


// We want a window of allocated global memory (malloced)
//   a) This window contains all allocations
//   b) The base of the window starts at the address of the first allocation
//   c) All other allocations are offsets from the base
//
//   It looks like this
//   
//   base 
//   <------------------------------------------------------------>
//   <------>                      <------------------->
//   allocation1 (address=base)    allocation2 (address=base+offset)
//
__device__ bool Runtime::allocateMemoryChunk(size_t bytes, size_t address)
{
    return !(address+bytes > ((size_t)g_runtimeState.m_physicalMemory+PHYSICAL_MEMORY_SIZE));
}

__device__ void* Runtime::translateSimulatedAddressToCudaAddress(void* simAddress)
{
    void* cudaAddress = (void*)((size_t)g_runtimeState.m_physicalMemory + (size_t)simAddress);
	printf("Translated simulated address %p to cuda address %p\n", simAddress, cudaAddress);
	
	return cudaAddress;
}

__device__ void* Runtime::translateCudaAddressToSimulatedAddress(void* cudaAddress)
{
    return (void*)((size_t)cudaAddress - (size_t)g_runtimeState.m_physicalMemory);
}

// The Runtime class owns all of the simulator state, it should have allocated it in the constructor
//  a) simulated state is CoreSimKernel/Block/Thread and other classes
//  b) this call changes the number of CoreSimBlock/Thread
__device__ void Runtime::setupLaunchConfig(unsigned int totalCtas, unsigned int threadsPerCta)
{
    g_runtimeState.m_simulatedBlocks = totalCtas;
    
    for (unsigned int i = 0; i < NUMBER_OF_HW_BLOCKS; ++i)
    {
        g_runtimeState.m_blocks[i].setNumberOfThreadsPerBlock(threadsPerCta);
    }
}

// Similar to the previous call, this sets the memory sizes
__device__ void Runtime::setupMemoryConfig(unsigned int localMemoryPerThread, unsigned int sharedMemoryPerCta)
{
    for (unsigned int i = 0; i < NUMBER_OF_HW_BLOCKS; ++i) 
    {
       g_runtimeState.m_blocks[i].setMemoryState(localMemoryPerThread, sharedMemoryPerCta);
    }
}

__device__ void Runtime::setupArgument(const void* data, size_t size, size_t offset)
{
	char* parameterBase = (char*)translateSimulatedAddressToCudaAddress(0);
	
	std::memcpy(parameterBase + offset, data, size);
}

__device__ size_t Runtime::baseOfUserMemory()
{
	// parameter base + parameter size
	return 0 + PARAMETER_MEMORY_SIZE;
}

// Set the PC of all threads to the PC of the specified function
//   Call into the binary to get the PC
__device__ void Runtime::setupKernelEntryPoint(const char* functionName)
{
    g_runtimeState.m_launchSimulationAtPC = g_runtimeState.m_loadedBinary->findFunctionsPC(functionName);    
}

// Start a new asynchronous kernel with the right number of HW CTAs/threads
__device__ void Runtime::launchSimulation()
{
    util::HostReflection::launch(NUMBER_OF_HW_BLOCKS,
    	NUMBER_OF_HW_THREADS_PER_BLOCK, "launchSimulation");
}

__device__ void Runtime::launchSimulationInParallel()
{
    g_runtimeState.m_kernel->launchKernel(g_runtimeState.m_simulatedBlocks, 	
        g_runtimeState.m_blocks, g_runtimeState.m_loadedBinary);
}

extern "C" __global__ void launchSimulation(
	util::HostReflection::Payload payload)
{
	Runtime::launchSimulationInParallel();
}

__device__ void Runtime::munmap(size_t address)
{

}

__device__ void Runtime::unloadBinary()
{
    delete g_runtimeState.m_loadedBinary;
}


}

