/*! \file   DominatorAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2009
	\file   The header file for the DominatorAnalysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Forward Declaration
namespace vanaheimr { namespace ir { class BasicBlock; } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief Dominator analysis using the algorithm described in:

	"A simple and fast dominance algorithm" by 
		Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
		
	Simple extensions for parallelism.
 */
class DominatorAnalysis : public FunctionAnalysis
{
public:
	typedef              ir::BasicBlock BasicBlock;
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;

public:
	DominatorAnalysis();

public:
	/*! \brief Is a block dominated by another? */
	bool dominates(const BasicBlock& b, const BasicBlock& potentialDominator);

	/*! \brief Find the immediate dominator of a given block */
	BasicBlock* getDominator(const BasicBlock& b);
	
	/*! \brief Get the set of blocks immediately dominated by this block */
	const BasicBlockSet& getDominatedBlocks(const BasicBlock& b);

	/*! \brief Get the set of blocks in the dominance frontier of
		a specified block */
	const BasicBlockSet& getDominanceFrontier(const BasicBlock& b);
		
public:
	virtual void analyze(Function& function);

private:
	void _determineImmediateDominators(Function& function);
	void _determineDominatedSets(Function& function);
	void _determineDominanceFrontiers(Function& function);

private:
	typedef std::vector<BasicBlock*>   BasicBlockVector;
	typedef std::vector<BasicBlockSet> BasicBlockSetVector;
	
private:
	BasicBlockVector    _immediateDominators;
	BasicBlockSetVector _dominatedBlocks;
	BasicBlockSetVector _dominanceFrontiers;

};

}

}


