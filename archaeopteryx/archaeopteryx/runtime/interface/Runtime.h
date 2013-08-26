/*!	\file Runtime.h
	\date Tuesday Sept, 13th 2011
	\author Sudnya Padalikar
		    <mailsudnya@gmail.com>
	\brief  The header file for the runtime API.
*/
#pragma once

#include <archaeopteryx/ir/interface/Binary.h>

// Forward Declarations
namespace archaeopteryx { namespace util { class File; } }

namespace archaeopteryx { namespace executive { class CoreSimBlock;  } }
namespace archaeopteryx { namespace executive { class CoreSimKernel; } }

namespace archaeopteryx
{

namespace rt
{

class Runtime
{
public:
	typedef uint64_t Address;

public:
	__device__ static void create();
	__device__ static void destroy();

	__device__ static void loadKnobs();

public:
	__device__ static void loadBinary(const char* fileName);
	__device__ static void loadBinary(ir::Binary* binary);
	__device__ static void unloadBinaries();

public:
	__device__ static Address mmap(uint64_t bytes);
	__device__ static bool mmap(size_t bytes, Address address);
	__device__ static void munmap(Address address);

	__device__ static void memcpy(Address src, Address dest, size_t dataSize);

public:
	__device__ static Address translateVirtualToPhysicalAddress(
		Address VirtualAddress);

public:
	__device__ static void setupLaunchConfig(unsigned int totalCtas,
		unsigned int threadsPerCta);
	__device__ static void setupMemoryConfig(unsigned int threadStackSize);
	__device__ static void setupArgument(const void* data, size_t size,
		size_t offset);
	__device__ static void setupKernelEntryPoint(const char* functionName);

public:
	__device__ static void launchSimulation();

public:
	__device__ static size_t findFunctionsPC(const char* functionName);
	__device__ static ir::Binary* getSelectedBinary();

};

}

}

