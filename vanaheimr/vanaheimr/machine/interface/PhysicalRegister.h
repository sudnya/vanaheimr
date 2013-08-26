/*! \file   PhysicalRegister.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the PhysicalRegister class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class RegisterFile; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A physical register in the machine model */
class PhysicalRegister
{
public:
	typedef unsigned int Id;

public:
	/*! \brief Construct a register associated with a register file */
	PhysicalRegister(const RegisterFile* file,
		Id id = 0, Id uid = 0, const std::string& name = "");

public:
	const RegisterFile* registerFile() const;
	Id                  id()           const;
	Id                  uniqueId()     const;
	const std::string&  name()         const;

private:
	const RegisterFile* _registerFile;
	Id                  _id;
	Id                  _uniqueId;
	std::string         _name;
};

}

}


