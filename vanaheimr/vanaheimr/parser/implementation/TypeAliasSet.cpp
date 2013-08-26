/*! \file   TypeAliasSet.cpp
	\date   March 25, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the TypeAliasSet class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/TypeAliasSet.h>

namespace vanaheimr
{

namespace parser
{

const ir::Type* TypeAliasSet::getType(const std::string& name) const
{
	auto type = _types.find(name);

	if(type == _types.end()) return nullptr;

	return type->second;
}

void TypeAliasSet::addAlias(const std::string& name, const ir::Type* type)
{
	_types[name] = type;
}

void TypeAliasSet::clear()
{
	_types.clear();
}

}

}



