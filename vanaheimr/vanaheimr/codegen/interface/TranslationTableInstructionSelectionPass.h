/*! \file   TranslationTableInstructionSelectionPass.h
	\date   Tuesday February 26, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the
		    TranslationTableInstructionSelectionPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace codegen
{

/*! \brief Perform instruction selection directly from the translation table */
class TranslationTableInstructionSelectionPass : public transforms::FunctionPass
{
public:
	TranslationTableInstructionSelectionPass();

public:
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;

private:
	void _lowerBlock(BasicBlock& b);

};

}

}


