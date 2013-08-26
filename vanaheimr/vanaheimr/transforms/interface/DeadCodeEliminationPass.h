/*! \file   DeadCodeEliminationPass.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the DeadCodeEliminationPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/include/Pass.h>

namespace vanaheimr
{

namespace transforms
{

class DeadCodeEliminationPass : public FunctionPass
{
public:
	DeadCodeEliminationPass();

public:
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;

};

}

}

