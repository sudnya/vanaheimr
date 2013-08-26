/*! \file   Local.cpp
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Local class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Local.h>

namespace vanaheimr
{

namespace ir
{

Local::Local(const std::string& n, Module* m, Function* f, const Type* t,
	Linkage l, Visibility v, Constant* c, unsigned int level)
: Global(n, m, t, l, v, c, level), _function(f)
{

}

Function* Local::function() const
{
	return _function;
}

}

}


