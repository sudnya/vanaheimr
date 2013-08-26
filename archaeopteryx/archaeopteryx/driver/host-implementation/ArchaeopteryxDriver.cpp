/*	\file   ArchaeopteryxDriver.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The source file for the ArchaeopteryxDriver class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/driver/interface/SimulatorKnobs.h>
#include <archaeopteryx/driver/host-interface/ArchaeopteryxDriver.h>
#include <archaeopteryx/util/host-interface/HostReflectionHost.h>

// Ocelot Includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Standard Library Includes
const char ArchaeopteryxModule[] = {
	#include <ArchaeopteryxModule.inc>
};

namespace archaeopteryx
{

namespace driver
{

void ArchaeopteryxDriver::runSimulation(const std::string& traceFileName,
	const KnobList& knobs)
{
	_knobs = knobs;
	
	_loadTraceFile(traceFileName);
	
	_loadArchaeopteryxDeviceCode();
	
	_runSimulation();
	
	_unloadArchaeopteryxDeviceCode();
}
	
void ArchaeopteryxDriver::_loadTraceFile(const std::string& traceFileName)
{
	_knobs.push_back(std::make_pair("TraceFileName", traceFileName));
}

void ArchaeopteryxDriver::_loadArchaeopteryxDeviceCode()
{
	std::stringstream stream(ArchaeopteryxModule);
	ocelot::registerPTXModule(stream, "ArchaeopteryxModule");
	
	archaeopteryx::util::HostReflectionHost::create("ArchaeopteryxModule");
}

void ArchaeopteryxDriver::_runSimulation()
{
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);

	SimulatorKnobs* deviceKnobs = _createDeviceKnobs();

	cudaSetupArgument(&deviceKnobs, 8, 0 );
	
	ocelot::launch("ArchaeopteryxModule", "archaeopteryxDriver");

	_freeDeviceKnobs(deviceKnobs);
}

void ArchaeopteryxDriver::_unloadArchaeopteryxDeviceCode()
{
	archaeopteryx::util::HostReflectionHost::destroy();
	
	ocelot::unregisterModule("ArchaeopteryxModule");
}

SimulatorKnobs* ArchaeopteryxDriver::_createDeviceKnobs()
{
	typedef std::vector<SimulatorKnobs::KnobOffsetPair> OffsetVector;

	OffsetVector offsets;

	// Allocate memory for knobs
	size_t size = sizeof(SimulatorKnobs);
	
	size += _knobs.size() * sizeof(SimulatorKnobs::KnobOffsetPair);

	for(auto knob = _knobs.begin(); knob != _knobs.end(); ++knob)
	{
		SimulatorKnobs::KnobOffsetPair pair;

		pair.first  = size;
		pair.second = size + knob->first.size() + 1;

		offsets.push_back(pair);

		size += knob->first.size() + knob->second.size() + 2;
	}

	SimulatorKnobs* devicePointer = 0;

	cudaMalloc((void**)&devicePointer, size);

	// serialize the knobs
	SimulatorKnobs simulatorKnobs;

	simulatorKnobs.knobCount = _knobs.size();

	// 1) serialize the header
	char* deviceIterator = (char*) devicePointer;

	cudaMemcpy(deviceIterator, &simulatorKnobs, sizeof(SimulatorKnobs),
		cudaMemcpyHostToDevice);
	deviceIterator += sizeof(SimulatorKnobs);

	// 2) serialize the offsets
	cudaMemcpy(deviceIterator, offsets.data(),
		sizeof(SimulatorKnobs::KnobOffsetPair) * offsets.size(),
		cudaMemcpyHostToDevice);
	deviceIterator += sizeof(SimulatorKnobs::KnobOffsetPair) * offsets.size();
	
	// 3) serialize the knobs themselves
	for(auto knob = _knobs.begin(); knob != _knobs.end(); ++knob)
	{
			cudaMemcpy(deviceIterator, knob->first.c_str(),
				knob->first.size() + 1, cudaMemcpyHostToDevice);
			deviceIterator += knob->first.size() + 1;
		
			cudaMemcpy(deviceIterator, knob->second.c_str(),
				knob->second.size() + 1, cudaMemcpyHostToDevice);
			deviceIterator += knob->second.size() + 1;
	}

	return devicePointer;
}

void ArchaeopteryxDriver::_freeDeviceKnobs(SimulatorKnobs* knobs)
{
	cudaFree(knobs);
}

}

}


