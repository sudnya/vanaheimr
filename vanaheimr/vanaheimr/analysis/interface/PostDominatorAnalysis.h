/*! \file   PostDominatorAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2009
	\file   The header file for the PostDominatorAnalysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/Analysis.h>

namespace vanaheimr
{

namespace analysis
{

class PostDominatorAnalysis : public FunctionAnalysis
{
public:
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;
	
public:
	/*! \brief Is a block post-dominated by another? */
	bool postDominates(const BasicBlock& b,
		const BasicBlock& potentialPostDominator);

	/*! \brief Find the immediate post dominator of a given block */
	BasicBlock* getPostDominator(const BasicBlock& b);
	
	/*! \brief Get the set of blocks immediately post-dominated by this block */
	BlockSet getPostDominatedBlocks(const BasicBlock& b);
	
public:
	virtual void analyze(Function& function);

};

}

}


