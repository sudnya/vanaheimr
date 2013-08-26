/*	\file   ArchaeopteryxDeviceDriver.cu
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The source file for the ArchaeopteryxDeviceDriver class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/driver/interface/ArchaeopteryxDeviceDriver.h>
#include <archaeopteryx/driver/interface/SimulatorKnobs.h>

#include <archaeopteryx/runtime/interface/Runtime.h>

#include <archaeopteryx/util/interface/Knob.h>
#include <archaeopteryx/util/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace driver
{
	
__device__ ArchaeopteryxDeviceDriver::ArchaeopteryxDeviceDriver()
{
	device_report("creating simulator driver\n");

	util::KnobDatabase::create();
	rt::Runtime::create();
}

__device__ ArchaeopteryxDeviceDriver::~ArchaeopteryxDeviceDriver()
{
	device_report("destroying simulator driver\n");
	
	rt::Runtime::destroy();
	util::KnobDatabase::destroy();
}

__device__ static void loadDefaultKnobs()
{
	util::KnobDatabase::addKnob(
		new util::Knob("simulator-thread-stack-size", "128"));
	util::KnobDatabase::addKnob(
		new util::Knob("simulator-shared-memory-per-cta", "1024"));

	util::KnobDatabase::addKnob(new util::Knob("simulator-ctas", "64" ));
	util::KnobDatabase::addKnob(
		new util::Knob("simulator-threads-per-cta", "32"));
	util::KnobDatabase::addKnob(
		new util::Knob("simulator-registers-per-thread", "64"));
	util::KnobDatabase::addKnob(
		new util::Knob("simulated-link-register", "63"));
}

__device__ void ArchaeopteryxDeviceDriver::loadKnobs(
	const void* serializedKnobs)
{
	loadDefaultKnobs();

	const char* base     = (const char*) serializedKnobs;
	const char* iterator = base;	

	const SimulatorKnobs* header = (const SimulatorKnobs*)iterator;
	iterator += sizeof(SimulatorKnobs);

	for(unsigned int knob = 0; knob != header->knobCount; ++knob)
	{
		const SimulatorKnobs::KnobOffsetPair* offsets = 
			(const SimulatorKnobs::KnobOffsetPair*) iterator;
		iterator += sizeof(SimulatorKnobs::KnobOffsetPair);

		const char* knobName  = base + offsets->first;
		const char* knobValue = base + offsets->second;

		device_report("Loaded knob (%s, %s)\n", knobName, knobValue);

		util::KnobDatabase::addKnob(new util::Knob(knobName, knobValue));
	}
}

__device__ void ArchaeopteryxDeviceDriver::runSimulation()
{
	_loadFile();
	_extractSimulatorParameters();
	_loadInitialMemoryContents();
	_runSimulation();
	_verifyMemoryContents();
}

__device__ void ArchaeopteryxDeviceDriver::_loadFile()
{
	util::string fileName =
		util::KnobDatabase::getKnob<util::string>("TraceFileName");

	rt::Runtime::loadBinary(fileName.c_str());
	
	device_report("loaded binary...\n");
}

__device__ static void addKnobFromBinary(ir::Binary* binary,
	const char* knobName)
{
	device_report(" Getting symbol (%s)\n", knobName);

	util::string value = binary->getSymbolDataAsString(knobName);
	
	device_report("  Loaded knob (%s, %s)\n", knobName, value.c_str());

	util::KnobDatabase::addKnob(new util::Knob(knobName, value));
}

__device__ void ArchaeopteryxDeviceDriver::_extractSimulatorParameters()
{
	device_report("Extracting simulator knobs from binary.\n");
	
	ir::Binary* binary = rt::Runtime::getSelectedBinary();

	addKnobFromBinary(binary, "simulated-ctas"                    );
	addKnobFromBinary(binary, "simulated-parameter-memory-size"   );
	addKnobFromBinary(binary, "simulated-parameter-memory"        );
	addKnobFromBinary(binary, "simulated-parameter-memory-address");
	addKnobFromBinary(binary, "simulated-threads-per-cta"         );
	addKnobFromBinary(binary, "simulated-shared-memory-per-cta"   );
	addKnobFromBinary(binary, "simulated-kernel-name"             );
}

__device__ rt::Runtime::Address getAddress(const util::string& symbol)
{
	device_report(" getting address for symbol %s.\n", symbol.c_str());
	
	rt::Runtime::Address address = 0;
	
	unsigned int position = symbol.find("0x");
	
	if(position != util::string::npos)
	{
		util::string value = symbol.substr(position);
		
		address = util::atoi(value.c_str());
	}
	
	return address;
}

__device__ void ArchaeopteryxDeviceDriver::_loadInitialMemoryContents()
{
	device_report("Loading the initial contents of memory.\n");
	
	ir::Binary* binary = rt::Runtime::getSelectedBinary();

	ir::Binary::StringVector allocationNames =
		binary->getSymbolNamesThatMatch("simulated-allocation");

	for(ir::Binary::StringVector::iterator name = allocationNames.begin();
		name != allocationNames.end(); ++name)
	{
		rt::Runtime::Address address = getAddress(*name);
		
		device_report(" loading address 0x%x.\n", address);
		
		size_t bytes = binary->getSymbolSize(name->c_str());
		
		device_report("  allocating %d bytes.\n", bytes);
		bool success = rt::Runtime::mmap(bytes, address);
		
		if(!success)
		{
			device_report("  allocation failed!\n");
			continue;
		}
		
		rt::Runtime::Address physicalAddress =
			rt::Runtime::translateVirtualToPhysicalAddress(address);
			
		device_report("  allocated at %p, starting copy.\n", physicalAddress);
		
		binary->copySymbolDataToAddress((void*)physicalAddress, name->c_str());
	}
}

__device__ void ArchaeopteryxDeviceDriver::_runSimulation()
{
	rt::Runtime::loadKnobs();
	
	device_report("Launching simulation.\n");
	
	device_report(" launch config (%d ctas, %d threads).\n",
		util::KnobDatabase::getKnob<unsigned int>("simulated-ctas"),
		util::KnobDatabase::getKnob<unsigned int>("simulated-threads-per-cta"));
	
	rt::Runtime::setupLaunchConfig(
		util::KnobDatabase::getKnob<unsigned int>("simulated-ctas"),
		util::KnobDatabase::getKnob<unsigned int>("simulated-threads-per-cta"));
		
	rt::Runtime::setupMemoryConfig(util::KnobDatabase::getKnob<unsigned int>(
		"simulator-thread-stack-size"));

	util::string argumentMemory = util::KnobDatabase::getKnob<util::string>(
		"simulated-parameter-memory");

	rt::Runtime::setupArgument(argumentMemory.data(), argumentMemory.size(), 0);
	
	device_report(" parameter memory (%d bytes).\n", argumentMemory.size());
	
	util::string kernelEntryPoint = util::KnobDatabase::getKnob<util::string>(
		"simulated-kernel-name");
	
	device_report(" kernel name '%s'.\n", kernelEntryPoint.c_str());
	
	rt::Runtime::setupKernelEntryPoint(kernelEntryPoint.c_str());
	
	rt::Runtime::launchSimulation();
	
	device_report(" simulation completed...\n");
}

__device__ static bool verifyAllocation(const util::string& data,
	rt::Runtime::Address address, rt::Runtime::Address virtualAddress)
{
	// TODO do this in parallel
	const unsigned int chunkSize = 32;
	
	unsigned int chunks = (data.size() + chunkSize - 1) / chunkSize;
	
	const char* base = (const char*)address;
	
	std::printf(" Checking allocation at address 0x%p (%d bytes):",
		virtualAddress, data.size());
	
	bool anyErrors = false;
	
	for(unsigned int chunk = 0; chunk != chunks; ++chunk)
	{
		bool error = false;
		
		unsigned int index = chunkSize * chunk;
		
		unsigned int chunkLength = util::min(chunkSize,
			(unsigned int)(data.size() - index));
		
		for(unsigned int i = 0; i < chunkLength; ++i, ++index)
		{
			if(data[index] != base[index])
			{
				error = true;
				break;
			}
		}
		
		if(error)
		{
			if(!anyErrors)
			{
				anyErrors = true;
				
				std::printf("\n");
			}
			
			unsigned int index = chunkSize * chunk;
			
			std::printf("  Mismatch at (0x%x): reference:",
				(index + virtualAddress));
			
			for(unsigned int i = 0; i < chunkLength; ++i, ++index)
			{
				std::printf(" 0x%x", (int)(unsigned char)data[index]);				
			}
			
			std::printf("\n");
			
			std::printf("                           computed: ");
			
			index = chunkSize * chunk;
			
			for(unsigned int i = 0; i < chunkLength; ++i, ++index)
			{
				std::printf(" 0x%x", (int)(unsigned char)base[index]);				
			}
			
			std::printf("\n");
		}
	}
	
	if(!anyErrors)
	{
		std::printf(" matched.\n");
	}
	
	return !anyErrors;
}

__device__ void ArchaeopteryxDeviceDriver::_verifyMemoryContents()
{
	std::printf("Verifying the final contents of memory.\n");
	
	ir::Binary* binary = rt::Runtime::getSelectedBinary();

	ir::Binary::StringVector allocationNames =
		binary->getSymbolNamesThatMatch("simulated-verify-allocation");

	bool anyErrors = false;

	for(ir::Binary::StringVector::iterator name = allocationNames.begin();
		name != allocationNames.end(); ++name)
	{
		rt::Runtime::Address address = getAddress(*name);
		
		rt::Runtime::Address physicalAddress =
			rt::Runtime::translateVirtualToPhysicalAddress(address);
		
		util::string contents = binary->getSymbolDataAsString(name->c_str());
		
		anyErrors |= verifyAllocation(contents, physicalAddress, address);
	}
	
	if(anyErrors)
	{
		std::printf("Memory check passed!\n");
	}
	else
	{
		std::printf("Memory check failed.\n");
	}
}

}

}

extern "C" __global__ void archaeopteryxDriver(const void* knobs)
{
	archaeopteryx::driver::ArchaeopteryxDeviceDriver driver;

	driver.loadKnobs(knobs);
	driver.runSimulation();
}

