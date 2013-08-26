/*!	\file   OcelotToVIRTraceTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday December 12, 2012
	\brief  The header file for the OcelotToVIRTraceTranslator class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }

// Standard Library Includes
#include <string>

namespace vanaheimr
{

namespace translation
{

class OcelotToVIRTraceTranslator
{
public:
	OcelotToVIRTraceTranslator(compiler::Compiler* compiler);

public:
	void translate(const std::string& traceFileName);

public:
	std::string translatedModuleName() const;

private:
	compiler::Compiler* _compiler;
	
	std::string _translatedModuleName;
};

}

}


