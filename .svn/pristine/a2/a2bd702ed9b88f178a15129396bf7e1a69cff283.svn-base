/*! \file   ReversePostOrderTraversal.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2012
	\file   The header file for the ReversePostOrderTraversal class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class BasicBlock; } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief Computes a post order traversal of the CFG.

	Ideally, it does so in parallel 
*/
class ReversePostOrderTraversal : public FunctionAnalysis
{
public:
	typedef ir::BasicBlock           BasicBlock;
	typedef std::vector<BasicBlock*> BasicBlockVector;

public:
	ReversePostOrderTraversal();

public:
	virtual void analyze(Function& function);

public:
	BasicBlockVector order;


};

}

}



