/*!	\file   BinaryHeader.h
	\date   Saturday March 4, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the specification of the header of the binary
*/

// Vanaheimr Includes
#include <vanaheimr/util/interface/IntTypes.h>

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{

/*! \brief A namespace for the assembly representation */
namespace as
{

class BinaryHeader
{
public:
	static const unsigned int PageSize    = (1 << 15); // 32 KB
	static const uint64_t     MagicNumber = 0x2E5649527F454C46ULL;

public:
	uint64_t magic          : 64;
	uint32_t dataPages      : 32;
	uint32_t codePages      : 32;
	uint32_t symbols        : 32;
	uint32_t stringPages    : 32;
	uint64_t dataOffset     : 64;
	uint64_t codeOffset     : 64;
	uint64_t symbolOffset   : 64;
	uint64_t stringsOffset  : 64;
	uint64_t nameOffset     : 64;
};

}

}

