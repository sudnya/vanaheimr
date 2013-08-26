/*! \file   MachineModelFactory.h
	\date   Wednesday January 16, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineModelFactory class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/MachineModelFactory.h>

#include <vanaheimr/machine/interface/MachineModel.h>
#include <vanaheimr/machine/interface/ArchaeopteryxSimulatorMachineModel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

class MachineDatabase
{
public:
	typedef std::map<std::string, MachineModel*> MachineMap;

public:
	MachineMap machines;

public:
	~MachineDatabase();
};

MachineDatabase::~MachineDatabase()
{
	for(auto machine : machines)
	{
		delete machine.second;
	}
}

static MachineDatabase machineDatabase;


MachineModel* MachineModelFactory::createMachineModel(const std::string& name,
	const StringVector& options)
{
	MachineModel* machine = nullptr;

	if(name == "ArchaeopteryxSimulator")
	{
		machine = new ArchaeopteryxSimulatorMachineModel;
	}

	auto databaseEntry = machineDatabase.machines.find(name);

	if(databaseEntry != machineDatabase.machines.end())
	{
		machine = databaseEntry->second->clone();
	}

	if(machine != nullptr)
	{
		machine->configure(options);
	}

	return machine;
}

MachineModel* MachineModelFactory::createDefaultMachine()
{
	return createMachineModel("ArchaeopteryxSimulator");
}

void MachineModelFactory::registerMachineModel(const MachineModel* machine)
{
	assert(machineDatabase.machines.count(machine->name) == 0);

	machineDatabase.machines.insert(std::make_pair(machine->name,
		machine->clone()));
}

}

}

