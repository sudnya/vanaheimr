/*! \file   Argument.h
	\date   Saturday February 11, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Argument class.
	
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class Type;     } }
namespace vanaheimr { namespace ir { class Function; } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Describe a function argument */
class Argument
{
public:
	Argument(const Type* type, Function* f, const std::string& name = "");

public:
	const std::string& name() const;
	      std::string  mangledName() const;
	const Type& type() const;

public:
	void setFunction(Function* f);

private:
	const Type* _type;
	Function*   _function;
	std::string _name;
};

}

}


