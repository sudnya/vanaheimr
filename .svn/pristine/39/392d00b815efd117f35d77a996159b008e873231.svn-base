/*! \file   TestFileAccesses.h
	\date   Tuesday June 28, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the TestFileAccesses series of units tests for
		CUDA file accesses.
*/

// Ocelot Includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Archaeopteryx Includes
#include <archaeopteryx/util/host-interface/HostReflectionHost.h>

// Autogen files
const char TestFileAccessesKernel[] = {
	#include <TestFileAccessesKernel.inc>
};


// Standard Library Includes
#include <string>
#include <iostream>
#include <sstream>

namespace test
{

static unsigned int align(unsigned int size, unsigned int alignment)
{
	unsigned int remainder = size % alignment;
	return remainder == 0 ? size : size + alignment - remainder;
}

bool testReadWriteFile(const std::string& filename, unsigned int size)
{
	std::stringstream stream(TestFileAccessesKernel);
	ocelot::registerPTXModule(stream, "ArchaeopteryxModule");
	
	archaeopteryx::util::HostReflectionHost::create("ArchaeopteryxModule");

	size = align(size, sizeof(unsigned int));

	char* hostFilename = 0;
	char* deviceFilename = 0;
	cudaHostAlloc((void**)&hostFilename, filename.size() + 1,
		cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&deviceFilename, hostFilename, 0);

	strcpy(hostFilename, filename.c_str());

	unsigned int* hostData = 0;
	unsigned int* deviceData = 0;
	cudaHostAlloc((void**)&hostData, size, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&deviceData, hostData, 0);

	unsigned int* hostResult = 0;
	unsigned int* deviceResult = 0;
	cudaHostAlloc((void**)&hostResult, size, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&deviceResult, hostResult, 0);

	for(unsigned int i = 0; i < size/sizeof(unsigned int); ++i)
	{
		hostData[i] = std::rand();
	}
	
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);

	cudaSetupArgument(&deviceFilename, 8, 0 );
	cudaSetupArgument(&deviceResult,   8, 8 );
	cudaSetupArgument(&deviceData,     8, 16);
	cudaSetupArgument(&size,           4, 24);

	ocelot::launch("ArchaeopteryxModule", "deviceMain");
	
	bool pass = std::memcmp(hostData, hostResult, size) == 0;

	if(!pass)
	{
		unsigned int errors = 0;
		for(unsigned int i = 0; i < size; ++i)
		{
			if(hostData[i] != hostResult[i])
			{
				std::cout << " at [" << i << "] original (" << std::hex
					<< hostData[i] << std::dec << ") != copied (" << std::hex
					<< hostResult[i] << std::dec << ")\n";
			
				if(errors++ > 10) break; 
			}
		}
	}
			
	cudaFreeHost(hostFilename);
	cudaFreeHost(hostData);
	cudaFreeHost(hostResult);
	
	archaeopteryx::util::HostReflectionHost::destroy();
	
	return pass;
}

}

int main(int argc, char** argv)
{
	if(test::testReadWriteFile("Archaeopteryx_Test_File", 1000))
	{
		std::cout << "Pass/Fail: Pass\n";
	}
	else
	{
		std::cout << "Pass/Fail: Fail\n";
	}
}


