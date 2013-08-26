/*! \file   RegisterFile.cpp
	\date   Tuesday January 22, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the RegisterFile class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/RegisterFile.h>

namespace vanaheimr
{

namespace machine
{

RegisterFile::RegisterFile(const MachineModel* mm, const std::string& n)
: _machineModel(mm), _name(n)
{

}

const MachineModel* RegisterFile::machineModel() const
{
	return _machineModel;
}

const std::string& RegisterFile::name() const
{
	return _name;
}

}

}



