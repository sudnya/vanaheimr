/*! \file   Global.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Global class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Variable.h>

// Forward  Declarations
namespace vanaheimr { namespace ir { class Module;   } }
namespace vanaheimr { namespace ir { class Constant; } }
namespace vanaheimr { namespace ir { class Type;     } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Describes a vanaheimr globally-scoped variable */
class Global : public Variable
{
public:
	Global(const std::string& name = "", Module* m = 0,
		const Type* t = 0, Linkage l = InternalLinkage,
		Visibility v = HiddenVisibility, Constant* c = 0);
	~Global();

public:
	Global(const Global&);
	Global& operator=(const Global&);

public:
	bool hasInitializer() const;

	Constant*       intializer();
	const Constant* initializer() const;

public:
	size_t bytes() const;

public:
	void setInitializer(Constant* c);
	
private:
	Constant* _initializer; // owned by the global

};

}

}


