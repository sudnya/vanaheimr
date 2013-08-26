/*! \file   ArchaeopteryxSimulatorMachineModel.cpp
	\date   Tuesday January 22, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ArchaeopteryxSimulatorMachineModel class.
*/

// Vanahieimr Includes
#include <vanaheimr/machine/interface/ArchaeopteryxSimulatorMachineModel.h>

namespace vanaheimr
{

namespace machine
{

ArchaeopteryxSimulatorMachineModel::ArchaeopteryxSimulatorMachineModel()
: MachineModel("ArchaeopteryxSimulator")
{
	addRegisterFile("rf", 64);
}

MachineModel* ArchaeopteryxSimulatorMachineModel::clone() const
{
	return new ArchaeopteryxSimulatorMachineModel;
}

}

}


