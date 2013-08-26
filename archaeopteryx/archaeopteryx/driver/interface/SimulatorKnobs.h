/*	\file   SimulatorKnobs.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The header file for the SimulatorKnobs class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/IntTypes.h>

namespace archaeopteryx
{

namespace driver
{

class SimulatorKnobs
{
public:
	class KnobOffsetPair
	{
	public:
		uint32_t first;
		uint32_t second;
	};

public:
		uint32_t knobCount;

};

}

}

