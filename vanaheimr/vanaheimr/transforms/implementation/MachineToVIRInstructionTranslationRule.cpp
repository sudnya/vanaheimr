/*! \file   MachineToVIRInstructionTranslationRule.h
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineToVIRInstructionTranslationRule class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/MachineToVIRInstructionTranslationRule.h>

namespace vanaheimr
{

namespace transforms
{


MachineToVIRInstructionTranslationRule::MachineToVIRInstructionTranslationRule(
	const std::string& opcodeName)
: opcode(opcodeName)
{

}

MachineToVIRInstructionTranslationRule::~MachineToVIRInstructionTranslationRule()
{

}

}

}



