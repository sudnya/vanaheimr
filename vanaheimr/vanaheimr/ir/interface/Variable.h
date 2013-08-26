/*! \file   Variable.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Variable class.
*/

#pragma once 

// Standard Library Includes
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class Module; } }
namespace vanaheimr { namespace ir { class Type;   } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Anything that can be defined at global scope */
class Variable
{
public:
	enum Linkage
	{
		ExternalLinkage = 0,//! Externally visible function
		LinkOnceAnyLinkage, //! Keep one copy of function when linking (inline)
		LinkOnceODRLinkage, //! Same, but only replaced by something equivalent.
		WeakAnyLinkage,     //! Keep one copy of named function when linking (weak)
		InternalLinkage,    //! Rename collisions when linking (static functions).
		PrivateLinkage      //! Like Internal, but omit from symbol table.
	};

	enum Visibility
	{
		HiddenVisibility,
		VisibleVisibility,
		ProtectedVisibility
	};

public:
	Variable(const std::string& name, Module* module,
		const Type* type, Linkage linkage, Visibility visibility);

public:
	void setModule(Module* m);

public:
	const std::string& name() const;
	Module*            module() const;
	Linkage            linkage() const;
	Visibility         visibility() const;
	const Type&        type() const;

protected:
	void _setType(const Type*);
	void _setName(const std::string& name);

private:
	std::string _name;
	Module*     _module;
	Linkage     _linkage;
	Visibility  _visibility;
	const Type* _type;
};

}

}

