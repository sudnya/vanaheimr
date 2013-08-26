/*! \file   TranslationTableEntry.cpp
	\date   Thursday February 23, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

#include <vanaheimr/ir/interface/Constant.h>

namespace vanaheimr
{

namespace machine
{

TranslationTableEntry::TranslationTableEntry(const std::string& n)
: name(n)
{

}

TranslationTableEntry::~TranslationTableEntry()
{

}

}

}


