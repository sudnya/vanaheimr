/*! \file   ControlFlowGraph.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the control flow graph class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class BasicBlock; } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief Maintains predecessor and successor information for basic blocks.

	Average Case Complexity:
		O(N) in number of nodes.
		O(N) in number of fallthrough edges.
		
		O(N log N) in number of branch edges. // TODO revisit this

	Worst Case Complexity:
		Same as average.

*/
class ControlFlowGraph : public FunctionAnalysis
{
public:
	typedef ir::BasicBlock BasicBlock;
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;

public:
	ControlFlowGraph();

public:
	BasicBlockSet   getSuccessors(const BasicBlock&);
	BasicBlockSet getPredecessors(const BasicBlock&);

public:
	bool            isEdge(const BasicBlock& head, const BasicBlock& tail);
	bool      isBranchEdge(const BasicBlock& head, const BasicBlock& tail);
	bool isFallthroughEdge(const BasicBlock& head, const BasicBlock& tail);

public:
	      Function* function();
	const Function* function() const;

public:
	virtual void analyze(Function& function);

private:
	void _initializePredecessorsAndSuccessors(BasicBlock* block,
		BasicBlock* next);

private:
	typedef std::vector<BasicBlockSet> BasicBlockSetVector;

private:
	BasicBlockSetVector _successors;
	BasicBlockSetVector _predecessors;

private:
	Function* _function;
};

}


}

