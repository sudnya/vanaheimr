/*! \file   ListInstructionSchedulerPass.h
	\date   Sunday December 23, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ListInstructionSchedulerPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace codegen
{

/*! \brief Perform instruction scheduling using the list algorithm */
class ListInstructionSchedulerPass : public transforms::FunctionPass
{
public:
	ListInstructionSchedulerPass();

public:
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;


};

}

}


