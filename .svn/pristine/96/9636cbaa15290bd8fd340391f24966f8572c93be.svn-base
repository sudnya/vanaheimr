/*	\file   VirtualRegister.h
	\date   Thursday March 1, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the VirtualRegister class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class Function; } }
namespace vanaheimr { namespace ir { class Type;     } }

namespace vanaheimr
{

namespace ir
{

/*! \brief A virtual register in the vanaheimr IR */
class VirtualRegister
{
public:
	typedef unsigned int Id;

public:
	VirtualRegister(const std::string& name, Id id,
		Function* function, const Type* t);

public:
	std::string toString() const;

public:
	std::string name;
	Id          id;
	Function*   function;
	const Type* type;
};

}

}


