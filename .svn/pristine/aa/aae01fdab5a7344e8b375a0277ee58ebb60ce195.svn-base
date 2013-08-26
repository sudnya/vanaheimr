/*! \file   ThreadFrontierAnalysis.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ThreadFrontierAnalysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/Analysis.h>

#include <vanaheimr/util/includes/SmallSet.h>

namespace vanaheimr
{

namespace analysis
{

class ThreadFrontierAnalysis : public FunctionAnalysis
{
public:
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;
	typedef unsigned int Priority;
	
public:
	/*! \brief Get the blocks in the thread frontier of a specified block */
	BasicBlockSet getThreadFrontier(const BasicBlock& block) const;
	/*! \brief Get the scheduling priorty of a specified block */
	Priority getPriority(const BasicBlock& block) const;
	/*! \brief Test if a block is in the thread frontier of another block */
	bool isInThreadFrontier(const BasicBlock& block,
		const BasicBlock& potentialBlockInFrontier) const;
	
public:
	virtual void analyze(Function& function);

};

}

}

