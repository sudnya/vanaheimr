/*! \file   LLVMParser.h
	\date   March 3, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LLVM parser class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }

namespace vanaheimr
{

namespace parser
{

/*! \brief A parser for the low level virtual machine assembly language */
class LLVMParser
{
public:
	typedef compiler::Compiler Compiler;

public:
	LLVMParser(Compiler* compiler);

public:
	void parse(const std::string& filename);

public:
	std::string getParsedModuleName() const;

private:
	compiler::Compiler* _compiler;
	std::string         _moduleName;

};

}


}



