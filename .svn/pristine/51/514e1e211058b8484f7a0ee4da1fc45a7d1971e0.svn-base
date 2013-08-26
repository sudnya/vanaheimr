/*! \file   Local.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Local class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Global.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class Function; } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Describes a vanaheimr function-scoped variable */
class Local : public Global
{

public:
	Local(const std::string& name = "", Module* m = 0, Function* f = 0,
		const Type* t = 0, Linkage l = InternalLinkage,
		Visibility v = HiddenVisibility, Constant* c = 0,
		unsigned int level = Shared);

public:
	Function* function() const;

private:
	Function* _function;
};

}

}


