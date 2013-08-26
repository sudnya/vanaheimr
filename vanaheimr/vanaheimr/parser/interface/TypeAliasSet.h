/*! \file   TypeAliasSet.h
	\date   March 25, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TypeAliasSet class.
*/

// Standard Library Includes
#include <string>
#include <unordered_map>

// Forward Declarations
namespace vanaheimr { namespace ir { class Type; } }

namespace vanaheimr
{

namespace parser
{

class TypeAliasSet
{
public:
	const ir::Type* getType(const std::string& name) const;

	void addAlias(const std::string& name, const ir::Type* type);

	void clear();

private:
	typedef std::unordered_map<std::string, const ir::Type*> TypeMap;

private:
	TypeMap _types;

};

}

}


