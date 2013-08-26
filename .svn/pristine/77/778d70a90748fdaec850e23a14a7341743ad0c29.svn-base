/*	\file   ArchaeopteryxDriver.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The header file for the ArchaeopteryxDriver class.
*/

#pragma once

// Standard Library Includes
#include <list>
#include <utility>
#include <string>

// Forward Declarations
namespace archaeopteryx { namespace driver { class SimulatorKnobs; } }

namespace archaeopteryx
{

namespace driver
{

class ArchaeopteryxDriver
{
public:
	typedef std::pair<std::string, std::string> Knob;
	typedef std::list<Knob> KnobList;

public:
	void runSimulation(const std::string& traceFileName, const KnobList& knobs);

private:
	void _loadTraceFile(const std::string& traceFileName);
	void _loadArchaeopteryxDeviceCode();
	void _runSimulation();
	void _unloadArchaeopteryxDeviceCode();

private:
	SimulatorKnobs* _createDeviceKnobs();
	void _freeDeviceKnobs(SimulatorKnobs*);

private:
	KnobList _knobs;


};

}

}


