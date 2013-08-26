/*! \file   MachineModel.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineModel class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/MachineModel.h>
#include <vanaheimr/machine/interface/TranslationTable.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

MachineModel::MachineModel(const std::string& n)
: name(n), _translationTable(nullptr)
{

}

MachineModel::~MachineModel()
{
	delete _translationTable;
}

const PhysicalRegister* MachineModel::getPhysicalRegister(RegisterId id) const
{
	auto reg = _idToRegisters.find(id);
	
	if(reg == _idToRegisters.end()) return nullptr;
	
	return reg->second;
}

const Operation* MachineModel::getOperation(const std::string& name) const
{
	auto operation = _machineOperations.find(name);
	
	if(operation == _machineOperations.end()) return nullptr;
	
	return &operation->second;
}

unsigned int MachineModel::totalRegisterCount() const
{
	return _idToRegisters.size();
}

const TranslationTable* MachineModel::translationTable() const
{
	return _translationTable;
}

void MachineModel::addOperation(const Operation& op)
{
	assert(_machineOperations.count(op.name) == 0);

	_machineOperations.insert(std::make_pair(op.name, op));
}

std::string makeRegisterName(const RegisterFile& file, unsigned int id)
{
	std::stringstream stream;
	
	stream << file.name() << id;
	
	return stream.str();
}

void MachineModel::addRegisterFile(const std::string& name,
	unsigned int registers)
{
	auto file = _registerFiles.insert(std::make_pair(name,
		RegisterFile(this, name))).first;

	for(unsigned int i = 0; i < registers; ++i)
	{
		file->second.registers.push_back(PhysicalRegister(&file->second, i,
			_idToRegisters.size() + i, makeRegisterName(file->second, i)));
	}
	
	for(auto reg = file->second.registers.begin();
		reg != file->second.registers.end(); ++reg)
	{
		_idToRegisters.insert(std::make_pair(reg->uniqueId(), &*reg));
	}
}

void MachineModel::configure(const StringVector& )
{
	// blank for the base class
}

}

}


