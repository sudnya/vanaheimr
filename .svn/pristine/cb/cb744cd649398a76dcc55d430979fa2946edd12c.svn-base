/*! \file   RegisterFile.h
	\date   Tuesday January 22, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the RegisterFile class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/machine/interface/PhysicalRegister.h>

// Forward Declaration
namespace vanaheimr { namespace machine { class MachineModel; } }

namespace vanaheimr
{

namespace machine
{

class RegisterFile
{
public:
	typedef std::vector<PhysicalRegister> RegisterVector;

public:
	RegisterFile(const MachineModel* mm, const std::string& name);

public:
	const MachineModel* machineModel() const;
	const std::string&  name()         const;

public:
	RegisterVector registers;

private:
	const MachineModel* _machineModel;
	const std::string   _name;
	
};

}

}



