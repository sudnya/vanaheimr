/*! \file   Runtime.cpp
 *   \date   Tuesday Sept, 13th 2011
 *   \author Sudnya Padalikar
 *   <mailsudnya@gmail.com>
 *   \brief  The implementation file for the runtime API.
 **/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/executive/interface/Intrinsics.h>

#include <archaeopteryx/runtime/interface/Runtime.h>
#include <archaeopteryx/runtime/interface/MemoryPool.h>

#include <archaeopteryx/util/interface/Knob.h>
#include <archaeopteryx/util/interface/debug.h>

// Preprocessor Defines
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace rt
{

class RuntimeState
{
public:
	typedef util::vector<executive::CoreSimBlock> CTAVector;
	typedef util::map<util::string, ir::Binary*>  BinaryMap;
	typedef executive::CoreSimKernel              Kernel;

public:
	Kernel     kernel;
	CTAVector  hardwareCTAs;
	BinaryMap  binaries;
	MemoryPool memory;
	
public:
	size_t parameterMemoryAddress;
	
public:
	size_t simulatedBlockCount;
	size_t programEntryPointAddress;

};

__device__ static RuntimeState* state = 0;

__device__ void Runtime::create()
{
	device_report("Creating runtime.\n");

	state = new RuntimeState;

	executive::Intrinsics::loadIntrinsics();
}

__device__ void Runtime::destroy()
{
	device_report("Destroying runtime.\n");
	device_report(" unloading binaries...\n");
	
	unloadBinaries();

	device_report(" destroying runtime state..\n");

	executive::Intrinsics::unloadIntrinsics();
	
	delete state; state = 0;

	device_report(" destroyed runtime state..\n");
}

__device__ void Runtime::loadBinary(const char* fileName)
{
    state->binaries.insert(util::make_pair(fileName, new ir::Binary(fileName)));
}

__device__ bool Runtime::mmap(size_t bytes, Address address)
{
	return state->memory.allocate(bytes, address);
}

__device__ Runtime::Address Runtime::mmap(uint64_t bytes)
{
	return state->memory.allocate(bytes);
}

__device__ void Runtime::munmap(Address address)
{
	state->memory.deallocate(address);
}

__device__ Runtime::Address
	Runtime::translateVirtualToPhysicalAddress(Address virtualAddress)
{
	return state->memory.translate((size_t) virtualAddress);
}

__device__ void Runtime::loadKnobs()
{
	unsigned int ctas =
		util::KnobDatabase::getKnob<unsigned int>("simulator-ctas");
	state->hardwareCTAs.resize(ctas);

	state->kernel.simulatedBlocks = ctas;
	state->kernel.linkRegister =
		util::KnobDatabase::getKnob<unsigned int>("simulated-link-register");

	Address parameterMemoryAddress = 
		util::KnobDatabase::getKnob<Address>(
			"simulated-parameter-memory-address");
	
	device_report("Allocating parameter memory at address %p\n",
		parameterMemoryAddress);

	state->parameterMemoryAddress = parameterMemoryAddress;

	bool success = mmap(util::KnobDatabase::getKnob<size_t>(
			"simulated-parameter-memory-size"),
			parameterMemoryAddress);

	device_assert(success);
			
	device_report(" Allocated parameter memory at address %p\n",
		state->parameterMemoryAddress);
}

// The Runtime class owns all of the simulator state, it should have allocated
//       it in the constructor
//  a) simulated state is CoreSimKernel/Block/Thread and other classes
//  b) this call changes the number of CoreSimBlock/Thread
__device__ void Runtime::setupLaunchConfig(unsigned int totalCtas,
	unsigned int threadsPerCta)
{
	state->simulatedBlockCount = totalCtas;
   
	// TODO: run in a kernel 
    for(RuntimeState::CTAVector::iterator cta = state->hardwareCTAs.begin();
		cta != state->hardwareCTAs.end(); ++cta)
    {
        cta->setNumberOfThreadsPerBlock(threadsPerCta);
    }
}

// Similar to the previous call, this sets the memory sizes
__device__ void Runtime::setupMemoryConfig(unsigned int threadStackSize)
{
	unsigned int sharedMemoryPerCta =
		util::KnobDatabase::getKnob<unsigned int>(
			"simulator-shared-memory-per-cta");

	// TODO: run in a kernel 
    for(RuntimeState::CTAVector::iterator cta = state->hardwareCTAs.begin();
		cta != state->hardwareCTAs.end(); ++cta)
    {
        cta->setMemoryState(threadStackSize, sharedMemoryPerCta);
    }
}

__device__ void Runtime::setupArgument(const void* data,
	size_t size, size_t offset)
{
	device_report("Adding argument (address 0x%p, %d bytes, at offset %d) "
		"to parameter memory (%p)\n", data, (int)size, (int)offset,
		state->parameterMemoryAddress);
	char* parameterBase =
		(char*)translateVirtualToPhysicalAddress(state->parameterMemoryAddress);
	
	std::memcpy(parameterBase + offset, data, size);
}

// Set the PC of all threads to the PC of the specified function
//   Call into the binary to get the PC
__device__ void Runtime::setupKernelEntryPoint(const char* functionName)
{
    state->programEntryPointAddress = findFunctionsPC(functionName);
}

__global__ void launchSimulationInParallel()
{
    kernel_report("Booting up parallel simulation entry point with "
    	"(%d ctas, %d threads)\n", gridDim.x, blockDim.x);
    
    state->kernel.launchKernel(&state->hardwareCTAs[0],
    	Runtime::getSelectedBinary());    
}

// Start a new asynchronous kernel with the right number of HW CTAs/threads
__device__ void Runtime::launchSimulation()
{
	unsigned int ctas    =
		util::KnobDatabase::getKnob<unsigned int>("simulator-ctas");
	unsigned int threads =
		util::KnobDatabase::getKnob<unsigned int>("simulator-threads-per-cta");
	
	state->kernel.simulatedBlocks = ctas;
	launchSimulationInParallel<<<ctas, threads>>>();
	cudaDeviceSynchronize();

    kernel_report("Parallel simulation finished.\n");
}

__device__ void Runtime::unloadBinaries()
{
	for(RuntimeState::BinaryMap::iterator binary = state->binaries.begin();
		binary != state->binaries.end(); ++binary)
	{
		delete binary->second;
	}
	
	device_report("   clearing binary map...\n");
	
	device_assert(state != 0);
	state->binaries.clear();
	
	device_report("    finished...\n");
}

__device__ size_t Runtime::findFunctionsPC(const char* functionName)
{
	for(RuntimeState::BinaryMap::iterator binary = state->binaries.begin();
		binary != state->binaries.end(); ++binary)
	{
		if(!binary->second->containsFunction(functionName)) continue;

		return binary->second->findFunctionsPC(functionName);
	}

	//assertM(false, "Function name not found.");

	return 0;
}

__device__ ir::Binary* Runtime::getSelectedBinary()
{
	//TODO support multiple binaries (requires linking)
	return state->binaries.begin()->second;
}

}

}

