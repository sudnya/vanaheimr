/*! \file   ListInstructionSchedulerPass.cpp
	\date   Sunday December 23, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ListInstructionSchedulerPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ListInstructionSchedulerPass.h>

#include <vanaheimr/analysis/interface/DependenceAnalysis.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

#include <vanaheimr/util/interface/LargeSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace codegen
{

ListInstructionSchedulerPass::ListInstructionSchedulerPass()
: FunctionPass({"DependenceAnalysis"}, "ListInstructionSchedulerPass")
{

}

typedef util::LargeSet<ir::Instruction*> InstructionSet;
	
static bool anyDependencies(ir::Instruction* instruction,
	analysis::DependenceAnalysis& dep, const InstructionSet& remaining)
{
	auto predecessors = dep.getLocalPredecessors(*instruction);

	for(auto writer : predecessors)
	{
		if(writer->block != instruction->block) continue;
		       if(remaining.count(writer) == 0) continue;
		
		assertM(writer->index() < instruction->index(), "Instruction '"
			<< writer->toString() << "' has a higher sequence number than '"
			<< instruction->toString() << "'");

		return true;
	}
	
	return false;
}

static void schedule(ir::BasicBlock& block, analysis::DependenceAnalysis& dep)
{
	report(" Scheduling basic block '" << block.name() << "'");

	// TODO sort by priority, sort in parallel
	ir::BasicBlock::InstructionList newInstructions;
	
	InstructionSet readyInstructions;
	
	InstructionSet remainingInstructions;
	
	remainingInstructions.insert(block.begin(), block.end());

	report("  Getting instructions with no dependencies...");
	
	for(auto instruction : block)
	{
		if(!anyDependencies(instruction, dep, remainingInstructions))
		{
			report("   " << instruction->toString());
		
			readyInstructions.insert(instruction);
		}
	}
	
	// Remove them from the set of remaining instructions
	for(auto instruction : readyInstructions)
	{
		auto remaining = remainingInstructions.find(instruction);

		assert(remaining != remainingInstructions.end());

		remainingInstructions.erase(remaining);
	}

	report("  Scheduling remaining instructions...");
	
	while(!readyInstructions.empty())
	{
		auto next = *readyInstructions.begin();
		readyInstructions.erase(readyInstructions.begin());

		report("   " << next->toString());

		newInstructions.push_back(next);

		// free dependent instructions
		auto successors = dep.getLocalSuccessors(*next);

		for(auto successor : successors)
		{
			auto remaining = remainingInstructions.find(successor);

			if(remaining == remainingInstructions.end()) continue;

			if(!anyDependencies(successor, dep, remainingInstructions))
			{
				remainingInstructions.erase(remaining);
				
				report("    released '" << successor->toString() << "'");

				readyInstructions.insert(successor);
			}
		}
	}

	assert(newInstructions.size() == block.size());

	block.assign(newInstructions.begin(), newInstructions.end());
}

void ListInstructionSchedulerPass::runOnFunction(Function& f)
{
	auto dep = static_cast<analysis::DependenceAnalysis*>(
		getAnalysis("DependenceAnalysis"));
	
	report("Running list scheduling on '" << f.name() << "'");
	
	// for all blocks
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		schedule(*block, *dep);
	}
}

transforms::Pass* ListInstructionSchedulerPass::clone() const
{
	return new ListInstructionSchedulerPass;
}

}

}



