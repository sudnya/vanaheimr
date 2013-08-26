/*!	\file   BinaryHeader.h
	\date   Saturday March 4, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the specification of the header of the binary
*/

// Standard Library Includes
#include <cstdint>

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace as
{

class BinaryHeader
{
public:
	static const unsigned int PageSize = (1 << 15); // 32 KB

public:
	uint32_t dataPages     : 32;
	uint32_t codePages     : 32;
	uint32_t symbols       : 32;
	uint32_t stringPages   : 32;
	uint64_t dataOffset    : 64;
	uint64_t codeOffset    : 64;
	uint64_t symbolOffset  : 64;
	uint64_t stringsOffset : 64;
	uint64_t nameOffset    : 64;
};

}

}

