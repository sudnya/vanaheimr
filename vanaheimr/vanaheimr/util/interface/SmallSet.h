/*! \file   SmallSet.h
	\date   Tuesday September 11, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SmallSet class.
*/

#pragma once

// Standard Library Includes
#include <set>

namespace vanaheimr
{

namespace util
{


/*! \brief A class optimized to store a small unique set of objects with
	zero mallocs, it falls back on a std::set if the set grows large */
template<typename T>
class SmallSet : public std::set<T>
{
// TODO specialize
};

}

}


