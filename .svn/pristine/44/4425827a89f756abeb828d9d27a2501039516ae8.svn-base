/*! \file   string.cpp
	\date   Tuesday June 28, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device string functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/string.h>

namespace util
{

__device__ void strlcpy(char* destination, const char* source, size_t max)
{
	max = max == 0 ? 1 : max;
	const char* end = source + (max - 1);
	for( ; source != end; ++source, ++destination)
	{
		*destination = *source;
		if( *source == '\0' ) return;
	}
	*destination = '\0';
}

__device__ int strcmp(const char* left, const char* right)
{
	while(*left != '\0' && *right != '\0')
	{
		if(*left != *right) return -1;
		
		++left; ++right;
	}

	if(*left != *right) return -1;

	return 0;
}

}

