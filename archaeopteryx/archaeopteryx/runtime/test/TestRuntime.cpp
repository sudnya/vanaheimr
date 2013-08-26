/*! \file TestRuntime.cu
 *  \date   Tuesday September 27, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  A test file to test the implementation of Runtime.
 *   */

#include <archaeopteryx/runtime/interface/Runtime.h>

// High level:
// Create i/p data, load binary, launch threads/run kernel, read outputs and verify outputs 
// Detailed steps:
// allocate memory for i/p data 
// maybe add memset/memcpy in runtime?
// Runtime loads binary from a given filename 
// Runtime finds the entry pt function
// simulation is launched by kernel for that PC
// runtime should memcpy outputs
// test should have a reference function to compare the output against

#define ARRAY_SIZE 1024

extern "C" __global__ void runTest()
{
    unsigned int* refX = 0;
    unsigned int* refY = 0;
    unsigned int a     = 5;

	size_t arrayBytes = ARRAY_SIZE * sizeof(unsigned int);

    refX = (unsigned int*)malloc(arrayBytes);
    refY = (unsigned int*)malloc(arrayBytes);

	rt::Runtime::create();

	rt::Runtime::loadBinary("BinarySaxpy.exe");

    util::HostReflection::launch(1, ARRAY_SIZE, "initValues",
    	util::HostReflection::createPayload(refX));
    util::HostReflection::launch(1, ARRAY_SIZE, "initValues",
    	util::HostReflection::createPayload(refY));
    util::HostReflection::launch(1, ARRAY_SIZE, "refCudaSaxpy",
    	util::HostReflection::createPayload(refX, refY, a));

    //allocate memory for arrays used by saxpy
    size_t baseX = rt::Runtime::baseOfUserMemory();
    size_t baseY = baseX + arrayBytes;
    bool allocX  = rt::Runtime::allocateMemoryChunk(arrayBytes, baseX);
    bool allocY  = rt::Runtime::allocateMemoryChunk(arrayBytes, baseY);

    if (allocX && allocY)
    {
        util::HostReflection::launch(1, ARRAY_SIZE, "initValues", 
        	util::HostReflection::createPayload(
        	rt::Runtime::translateSimulatedAddressToCudaAddress((void*)baseX)));
        util::HostReflection::launch(1, ARRAY_SIZE, "initValues", 
        	util::HostReflection::createPayload(
        	rt::Runtime::translateSimulatedAddressToCudaAddress((void*)baseY)));
		
		rt::Runtime::setupArgument(&baseX, sizeof(size_t),       0 );
		rt::Runtime::setupArgument(&baseY, sizeof(size_t),       8 );
		rt::Runtime::setupArgument(&a,     sizeof(unsigned int), 16);
		rt::Runtime::setupLaunchConfig(ARRAY_SIZE/128, 128);
        rt::Runtime::setupKernelEntryPoint("main");
        rt::Runtime::launchSimulation();

        util::HostReflection::launch(1, 1, "compareMemory",
        	util::HostReflection::createPayload
        	(rt::Runtime::translateSimulatedAddressToCudaAddress((void*)baseY),
        	refY, ARRAY_SIZE));
    }
} 

extern "C" __global__ void compareMemory(util::HostReflection::Payload payload)
{
	unsigned int* result       = payload.get<unsigned int*>(0);
    unsigned int* ref          = payload.get<unsigned int*>(1);
    unsigned int  memBlockSize = payload.get<unsigned int >(2);

    printf("Checking memory block of size %d, %p result, %p ref\n",
    	memBlockSize, result, ref);
    
    for (unsigned int i = 0; i < memBlockSize; ++i)
    {
        if (ref[i] != result[i])
        {
            printf(" memory not equal\n");
            return;
        }
    }

    printf(" test passed!\n");

	rt::Runtime::destroy();
}

extern "C" __global__ void initValues(util::HostReflection::Payload payload)
{
	unsigned int* array = payload.get<unsigned int*>(0);

    array[threadIdx.x] = threadIdx.x;
}

extern "C" __global__ void refCudaSaxpy(util::HostReflection::Payload payload)
{
	unsigned int* y = payload.get<unsigned int*>(0);
	unsigned int* x = payload.get<unsigned int*>(1);
	unsigned int  a = payload.get<unsigned int >(2);
	
    y[threadIdx.x] = a*x[threadIdx.x] + y[threadIdx.x];
}

int main(int argc, char** argv)
{
    util::HostReflection::create(__FILE__);

	runTest<<<1, 1>>>();
	
    util::HostReflection::destroy();
}


