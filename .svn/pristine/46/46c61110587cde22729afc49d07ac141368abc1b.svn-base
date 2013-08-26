/*! \file   PhysicalRegister.cpp
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the PhysicalRegister class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/PhysicalRegister.h>

namespace vanaheimr
{

namespace machine
{

PhysicalRegister::PhysicalRegister(const RegisterFile* file,
	Id id, Id uid, const std::string& n)
: _registerFile(file), _id(id), _uniqueId(uid), _name(n)
{

}

const RegisterFile* PhysicalRegister::registerFile() const
{
	return _registerFile;
}

PhysicalRegister::Id PhysicalRegister::id() const
{
	return _id;
}

PhysicalRegister::Id PhysicalRegister::uniqueId() const
{
	return _uniqueId;
}

const std::string& PhysicalRegister::name() const
{
	return _name;
}

}

}


