/*! \file   StlFunctions.cpp
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device equivalents of STL functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/StlFunctions.h>

namespace util
{

template<typename Type>
__host__ __device__ Type min(Type a, Type b)
{
	return a < b ? a : b;
}

template<typename Type>
__host__ __device__ Type max(Type a, Type b)
{
	return a > b ? a : b;
}

__host__ __device__ size_t strlen(const char* str)
{
	unsigned int size = 0;
	
	while(*str != '\0') { ++size; ++str; }
	
	return size;
}

}


