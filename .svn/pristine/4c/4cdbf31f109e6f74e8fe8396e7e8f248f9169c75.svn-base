/* !   \file Runtime.h
 *     \date Tuesday Sept, 13th 2011
        \author Sudnya Padalikar
                <mailsudnya@gmail.com>
        \brief  The header file for the runtime API.
*/
#pragma once

#include <archaeopteryx/ir/interface/Binary.h>

// Forward Declarations
namespace util { class File; }

namespace executive { class CoreSimBlock; }
namespace executive { class CoreSimKernel; }

namespace rt
{
class Runtime
{
    public:
        __device__ static void create();
        __device__ static void destroy();

        __device__ static void loadBinary(const char* fileName);
        __device__ static void loadBinarr(ir::Binary* binary);
        __device__ static bool allocateMemoryChunk(size_t bytes, size_t address);
        __device__ static void* translateSimulatedAddressToCudaAddress(void* simAddress);
        __device__ static void* translateCudaAddressToSimulatedAddress(void* CudaAddress);
        __device__ static void setupLaunchConfig(unsigned int totalCtas, unsigned int threadsPerCta);
        __device__ static void setupMemoryConfig(unsigned int localMemoryPerThread, unsigned int sharedMemoryPerCta);
        __device__ static void setupArgument(const void* data, size_t size,
        	size_t offset);
        __device__ static size_t baseOfUserMemory();
        __device__ static void setupKernelEntryPoint(const char* functionName);
        __device__ static void launchSimulation();
        __device__ static void munmap(size_t address);
        __device__ static void unloadBinary();
        __device__ static void memcpy(void* src, void* dest, size_t dataSize);

    public:
        struct RuntimeState 
        {
            ir::Binary*               m_loadedBinary;
            void*                     m_physicalMemory;
            executive::CoreSimKernel* m_kernel;
            executive::CoreSimBlock*  m_blocks;//an array of blocks?
            unsigned int              m_simulatedBlocks;
            ir::Binary::PC            m_launchSimulationAtPC;
        };
        
    public:
        __device__ static void launchSimulationInParallel();
};

}
#include <archaeopteryx/runtime/implementation/Runtime.cpp>
