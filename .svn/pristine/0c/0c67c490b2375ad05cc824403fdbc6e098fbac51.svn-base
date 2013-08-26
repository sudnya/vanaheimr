/*! \file  SymbolTableEntry.h
	\date   Saturday March 4, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the specification of the symbol table of the binary
*/

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{

/*! \brief A namespace for the internal representation */
namespace as
{

class SymbolTableEntry
{
public:
	enum Type
	{
		VariableType   = 0x1,
		FunctionType   = 0x2,
		ArgumentType   = 0x3,
		BasicBlockType = 0x4,
		InvalidType    = 0x0
	};
	
	class Attributes
	{
	public:
		uint32_t linkage    : 3;
		uint32_t visibility : 2;
		uint32_t unused     : 27;
	};

public:
    uint32_t type         : 32;
    
	union
	{
		uint32_t   attributeData : 32;
		Attributes attributes;
	};

	uint64_t stringOffset : 64;
    uint64_t offset       : 64;
	uint64_t typeOffset   : 64;
	uint64_t size         : 64;
};

}

}

