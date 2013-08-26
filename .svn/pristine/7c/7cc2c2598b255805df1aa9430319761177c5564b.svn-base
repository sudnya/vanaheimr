/*! \file   ReversePostOrderTraversal.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2012
	\file   The source file for the ReversePostOrderTraversal class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/ReversePostOrderTraversal.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>

#include <vanaheimr/util/interface/LargeSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Include
#include <stack>
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace analysis
{

ReversePostOrderTraversal::ReversePostOrderTraversal()
: FunctionAnalysis("ReversePostOrderTraversal",
  StringVector(1, "ControlFlowGraph"))
{

}

void ReversePostOrderTraversal::analyze(Function& function)
{
	typedef util::LargeSet<BasicBlock*> BlockSet;
	typedef std::stack<BasicBlock*>     BlockStack;

	order.clear();
	
	BlockSet   visited;
	BlockStack stack;
	
	auto cfgAnalysis = getAnalysis("ControlFlowGraph");
	auto cfg         = static_cast<ControlFlowGraph*>(cfgAnalysis);	

	report("Creating reverse post order traversal over function '" +
		function.name() + "'");

	// reverse post order is reversed topological order
	stack.push(&*function.entry_block());
	
	while(order.size() != function.size())
	{
		if(stack.empty())
		{
			for(auto block : order)
			{
				auto successors = cfg->getSuccessors(*block);
				
				for(auto successor : successors)
				{
					if(visited.insert(successor).second)
					{
						stack.push(successor);
						break;
					}
				}
				
				if(!stack.empty()) break;
			}
		}
		
		assertM(!stack.empty(), (function.size() - order.size())
			<< " blocks are not connected.");
		
		while(!stack.empty())
		{
			BasicBlock* top = stack.top();
			stack.pop();
		
			auto successors = cfg->getSuccessors(*top);
			
			for(auto successor : successors)
			{
				assert(successor != nullptr);
				
				auto predecessors = cfg->getPredecessors(*successor);
				
				bool allPredecessorsVisited = true;
		
				for(auto predecessor : predecessors)
				{
					if(visited.count(predecessor) == 0)
					{
						allPredecessorsVisited = false;
						break;
					}
				}
				
				if(!allPredecessorsVisited) continue;
				
				if(visited.insert(successor).second)
				{
					stack.push(successor);
				}
			}

			order.push_back(top);
		
			report(" " << top->name());
		}
	}
	
	// reverse the order
	std::reverse(order.begin(), order.end());
}

}

}



