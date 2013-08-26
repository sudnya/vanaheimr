/*! \file   cstring.h
	\date   Tuesday June 28, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for device string functions.
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

/*! \brief Safe string copy
	
	\param destination The target string
	\param source The source string
	\param max The max number of characters to copy
*/
__device__ void strlcpy(char* destination, const char* source, size_t max);

/*! \brief string compare
	
	\param left The target string
	\param right The source string
	
	\return 0 if all bytes match, some random int otherwise
*/
__device__ int strcmp(const char* left, const char* right);

__device__ int memcmp(const void* s1, const void* s2, size_t n);
__device__ size_t strlen(const char* s);
__device__ const void* memchr(const void* s, int c, size_t n);
__device__ void* memchr(      void* s, int c, size_t n);
__device__ void* memmove(void* s1, const void* s2, size_t n);

__device__ void* memcpy(void* s1, const void* s2, size_t n);
__device__ void* memset(void* s, int c, size_t n);

}

}

