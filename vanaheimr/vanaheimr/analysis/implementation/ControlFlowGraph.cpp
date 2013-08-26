/*! \file   ControlFlowGraph.cpp
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the control flow graph class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Instruction.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace analysis
{

ControlFlowGraph::ControlFlowGraph()
: FunctionAnalysis("ControlFlowGraph"), _function(nullptr)
{

}

ControlFlowGraph::BasicBlockSet
	ControlFlowGraph::getSuccessors(const BasicBlock& b)
{
	assert(b.id() < _successors.size());
	return _successors[b.id()];
}

ControlFlowGraph::BasicBlockSet
	ControlFlowGraph::getPredecessors(const BasicBlock& b)
{
	assert(b.id() < _predecessors.size());
	return _predecessors[b.id()];
}

bool ControlFlowGraph::isEdge(const BasicBlock& head, const BasicBlock& tail)
{
	BasicBlockSet successors = getSuccessors(head);
	
	return successors.count(const_cast<BasicBlock*>(&tail)) != 0;
}

bool ControlFlowGraph::isBranchEdge(const BasicBlock& head,
	const BasicBlock& tail)
{
	const ir::Instruction* terminator = head.terminator();
	
	if  (terminator == nullptr) return false;
	if(!terminator->isBranch()) return false;
	if(terminator->isCall())    return false;
	
	const ir::Bra* branch = static_cast<const ir::Bra*>(terminator);
	
	return &tail == branch->targetBasicBlock();
}

bool ControlFlowGraph::isFallthroughEdge(const BasicBlock& head,
	const BasicBlock& tail)
{
	if(isBranchEdge(head, tail)) return false;
	
	return isEdge(head, tail);
}

ir::Function* ControlFlowGraph::function()
{
	return _function;
}

const ir::Function* ControlFlowGraph::function() const
{
	return _function;
}

void ControlFlowGraph::analyze(Function& function)
{
	// perform the analysis sequentially
	  _successors.resize(function.size());
	_predecessors.resize(function.size());
		
	_function = &function;
		
	// should be for-all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		BasicBlock* nextBlock = nullptr;
		
		auto next = block; ++next;
		
		if(next != function.end())
		{
			nextBlock = &*next;
		}
			
		_initializePredecessorsAndSuccessors(&*block, nextBlock);
	}
}

void ControlFlowGraph::_initializePredecessorsAndSuccessors(BasicBlock* block,
	BasicBlock* next)
{
	BasicBlock* fallthrough = nullptr;
	BasicBlock* target      = nullptr;
	
	// find target and fallthrough
	ir::Instruction* terminator = block->terminator();
	
	ir::Bra* branch = nullptr;
	ir::Ret* ret    = nullptr;
	
	if(terminator != nullptr)
	{
		if(terminator->isBranch() && !terminator->isCall() &&
			!terminator->isMachineInstruction())
		{
			branch = static_cast<ir::Bra*>(terminator);
		}
		else if(terminator->isReturn())
		{
			ret = static_cast<ir::Ret*>(terminator);
		}
	}
	
	if(next != nullptr)
	{
		if(branch == nullptr)
		{
			fallthrough = next;
		}
		else if(!branch->isUnconditional())
		{
			fallthrough = next;
		}
	}
	
	if(ret != nullptr)
	{
		target = &*function()->exit_block();
	}
	
	if(branch != nullptr)
	{
		target = branch->targetBasicBlock();
	}
	
	// populate successor set
	BasicBlockSet& successors = _successors[block->id()];
	
	if(fallthrough != nullptr)
	{
		successors.insert(fallthrough);
	}
	
	if(target != nullptr)
	{
		successors.insert(target);
	}
	
	// populate predecessor sets, for a parallel algorithm, this 
	//  would need to create lists of predecessors for each block and
	//  then convert them into sets.
	// 
	// One of these algorithms would be possible:
	// 1) add (head, tail) pairs to an array, sort them by tail,
	//    then group by tail, each group becomes a set
	// 2) atomically add pairs by tail
	//
	//
	// For all of these, fallthroughs can go in parallel first since no
	// collisions are possible
	
	// in parallel
	if(fallthrough != nullptr)
	{
		BasicBlockSet& predecessors = _predecessors[fallthrough->id()];
		
		predecessors.insert(&*block);
	}
	
	// TODO: handle conflicts
	if(target != nullptr)
	{
		BasicBlockSet& predecessors = _predecessors[target->id()];
		
		predecessors.insert(&*block);
	}
}

}


}

