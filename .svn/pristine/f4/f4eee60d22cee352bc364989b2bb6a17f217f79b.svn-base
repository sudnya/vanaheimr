/*	\file   ArchaeopteryxDeviceDriver.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The header file for the ArchaeopteryxDeviceDriver class.
*/

#pragma once

namespace archaeopteryx
{

namespace driver
{

class ArchaeopteryxDeviceDriver
{
public:
	__device__ ArchaeopteryxDeviceDriver();
	__device__ ~ArchaeopteryxDeviceDriver();

public:
	__device__ void loadKnobs(const void* serializedKnobs);
	__device__ void runSimulation();

private:
	__device__ void _loadFile();
	__device__ void _extractSimulatorParameters();
	__device__ void _loadInitialMemoryContents();
	__device__ void _runSimulation();
	__device__ void _verifyMemoryContents();

};

}

}

