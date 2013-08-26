/*	\file   archaeopteryx-simulator.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The source file for the archaeopetryx simulator entry point
*/

// Archaeopteryx Includes
#include <archaeopteryx/driver/host-interface/ArchaeopteryxDriver.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/string.h>

// Standard Library Includes
#include <stdexcept>

namespace archaeopteryx
{

static driver::ArchaeopteryxDriver::KnobList extractKnobs(
	const std::string& commaSeparatedKnobs)
{
	auto individualKnobs = hydrazine::split(commaSeparatedKnobs, ",");
	
	driver::ArchaeopteryxDriver::KnobList knobs;
	
	for(auto knob : individualKnobs)
	{
		auto keyValuePair = hydrazine::split(knob, "=");
		
		if(keyValuePair.empty()) continue;
		
		if(keyValuePair.size() == 1)
		{
			knobs.push_back(std::make_pair(keyValuePair[0], "0"));
			continue;
		}
		
		knobs.push_back(std::make_pair(keyValuePair[0], keyValuePair[1]));
	}
	
	return knobs;
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("The Archaeopteryx simulator for the "
		"Vanaheimr processor architecture.");
	
	std::string input;
	std::string knobs;
	
	parser.parse( "-i", "--input", input, "",
		"The input trace file to be simulated." );
	parser.parse( "-k", "--knobs", knobs, "",
		"Comma separated list of knobs "
		"(e.g. 'key1=value1,key2=value2, etc')." );
	
	parser.parse();
	
	archaeopteryx::driver::ArchaeopteryxDriver driver;
	
	try
	{
		driver.runSimulation(input, archaeopteryx::extractKnobs(knobs));
	}
	catch(const std::exception& e)
	{
		std::cout << "Simulation Error:\n";
		std::cout << " Message: " << e.what() << "\n";
	}
	
	return 0;
}




