/*! \file   LiveRangeAnalysis.cpp
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the live-range analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/LiveRangeAnalysis.h>

#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace analysis
{

LiveRange::LiveRange(LiveRangeAnalysis* liveRangeAnalysis, VirtualRegister* vr)
: _analysis(liveRangeAnalysis), _virtualRegister(vr)
{
	hydrazine::log("LiveRangeAnalysis") << "  Live range for VR - '"
		<< vr->toString() << "'\n";
}

LiveRangeAnalysis* LiveRangeAnalysis::LiveRange::liveRangeAnalysis() const
{
	return _analysis;
}

ir::VirtualRegister* LiveRangeAnalysis::LiveRange::virtualRegister() const
{
	return _virtualRegister;
}

LiveRangeAnalysis::BasicBlockSet
	LiveRangeAnalysis::LiveRange::allBlocksWithLiveValue() const
{
	BasicBlockSet blocks = fullyCoveredBlocks;
	
	for(auto instruction : definingInstructions)
	{
		blocks.insert(instruction->block);
	}

	for(auto instruction : usingInstructions)
	{
		blocks.insert(instruction->block);
	}
	
	return blocks;
}

bool LiveRangeAnalysis::LiveRange::interferesWith(const LiveRange& range) const
{
	// easy case, live ranges intersect in fully covered blocks
	for(auto block : range.fullyCoveredBlocks)
	{
		if(fullyCoveredBlocks.count(block) != 0) return true;
	}
	
	// hard case, live ranges intersect in a partially covered block
	for(auto instruction : range.usingInstructions)
	{
		auto block = instruction->block;
	
		if(fullyCoveredBlocks.count(block) != 0) return true;
		
		auto user = ir::BasicBlock::reverse_iterator(
			block->getIterator(instruction));
		
		while(user != block->rend())
		{
			if(range.definingInstructions.count(*user) != 0) break;
		
			if(usingInstructions.count(*user) != 0)    return true;
			if(definingInstructions.count(*user) != 0) return true;

			++user;
		}
	}

	for(auto instruction : range.definingInstructions)
	{
		auto block = instruction->block;
	
		if(fullyCoveredBlocks.count(block) != 0) return true;
		
		auto definer = block->getIterator(instruction);
		
		while(definer != block->end())
		{
			if(range.usingInstructions.count(*definer) != 0) break;
		
			if(usingInstructions.count(*definer) != 0)    return true;
			if(definingInstructions.count(*definer) != 0) return true;
		
			++definer;
		}
	}
	
	return false;
}

LiveRangeAnalysis::LiveRangeAnalysis()
: FunctionAnalysis("LiveRangeAnalysis",
  {"DataflowAnalysis", "ControlFlowGraph"})
{

}

const LiveRange* LiveRangeAnalysis::getLiveRange(
	const VirtualRegister& virtualRegister) const
{
	assert(virtualRegister.id < _liveRanges.size());

	return &_liveRanges[virtualRegister.id];
}

LiveRangeAnalysis::LiveRange* LiveRangeAnalysis::getLiveRange(
	const VirtualRegister& virtualRegister)
{
	assert(virtualRegister.id < _liveRanges.size());

	return &_liveRanges[virtualRegister.id];
}

static void findLiveRange(LiveRangeAnalysis::LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* );

void LiveRangeAnalysis::analyze(Function& function)
{
	hydrazine::log("LiveRangeAnalysis") << "Running analysis on function '"
		<< function.name() << "'\n";
	
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
	assert(dfg != nullptr);

	auto cfg = static_cast<ControlFlowGraph*>(getAnalysis("ControlFlowGraph"));
	assert(cfg != nullptr);

	_initializeLiveRanges(function);

	hydrazine::log("LiveRangeAnalysis") << " Discovering live ranges\n";
	
	// compute the live range for each variable in parallel (for all)
	// TODO: use an algorithm that merges partial results
	for(auto virtualRegister = function.register_begin();
		virtualRegister != function.register_end(); ++virtualRegister)
	{
		findLiveRange(*getLiveRange(*virtualRegister), dfg, cfg);
	}
}

LiveRangeAnalysis::iterator LiveRangeAnalysis::begin()
{
	return _liveRanges.begin();
}

LiveRangeAnalysis::const_iterator LiveRangeAnalysis::begin() const
{
	return _liveRanges.begin();
}

LiveRangeAnalysis::iterator LiveRangeAnalysis::end()
{
	return _liveRanges.end();
}

LiveRangeAnalysis::const_iterator LiveRangeAnalysis::end() const
{
	return _liveRanges.end();
}

bool LiveRangeAnalysis::empty() const
{
	return _liveRanges.empty();
}

size_t LiveRangeAnalysis::size() const
{
	return _liveRanges.size();
}

void LiveRangeAnalysis::_initializeLiveRanges(Function& function)
{
	_liveRanges.clear();
	_liveRanges.reserve(function.register_size());

	hydrazine::log("LiveRangeAnalysis") << " Creating live ranges\n";
	
	for(auto virtualRegister = function.register_begin();
		virtualRegister != function.register_end(); ++virtualRegister)
	{
		_liveRanges.push_back(LiveRange(this, &*virtualRegister));
	}
}

typedef ir::BasicBlock BasicBlock;

static bool isLiveOut(LiveRange& liveRange, BasicBlock* block,
	DataflowAnalysis* dfg)
{
	auto liveOuts = dfg->getLiveOuts(*block);

	return liveOuts.count(liveRange.virtualRegister()) != 0;
}

static bool blockHasDefinitions(BasicBlock* block, const LiveRange& liveRange)
{
	for(auto definition : liveRange.definingInstructions)
	{
		if(definition->block == block) return true;
	}

	return false;
}

typedef LiveRangeAnalysis::BasicBlockSet BasicBlockSet;

static void walkUpPredecessor(BasicBlockSet& visited, LiveRange& liveRange,
	BasicBlock* block, DataflowAnalysis* dfg, ControlFlowGraph* cfg)
{
	// early exit when a node is already visited
	if(!visited.insert(block).second) return;

	// skip nodes for which the value is not live-out
	if(!isLiveOut(liveRange, block, dfg)) return;

	// skip nodes that define the value, starting the live range
	if(blockHasDefinitions(block, liveRange)) return;

	// add the block to the live range
	liveRange.fullyCoveredBlocks.insert(block);

	hydrazine::log("LiveRangeAnalysis") << "   fully covered block '"
		<< block->name() << "'\n";

	// recurse on predecessors with the value as a live out
	auto predecessors = cfg->getPredecessors(*block);
	
	for(auto predecessor : predecessors)
	{
		walkUpPredecessor(visited, liveRange, predecessor, dfg, cfg);
	}
	
}

typedef ir::Instruction Instruction;

static bool blockHasPriorDefinitions(LiveRange& liveRange, Instruction* user)
{
	for(auto definition : liveRange.definingInstructions)
	{
		// Is there a definition in the same block
		if(definition->block != user->block) continue;
		
		// Does the definition occur prior to this use?
		if(definition->index() < user->index()) return true;
	}

	return false;
}

static void walkUpDataflowGraph(LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* cfg, Instruction* user)
{
	BasicBlockSet visited;

	visited.insert(user->block);

	// skip blocks that start the live range
	if(blockHasPriorDefinitions(liveRange, user)) return;

	auto predecessors = cfg->getPredecessors(*user->block);
	
	for(auto predecessor : predecessors)
	{
		walkUpPredecessor(visited, liveRange, predecessor, dfg, cfg);
	}
}

static void findLiveRange(LiveRangeAnalysis::LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* cfg)
{
	auto vr = liveRange.virtualRegister();

	liveRange.definingInstructions = dfg->getReachingDefinitions(*vr);
	liveRange.usingInstructions    = dfg->getReachedUses(*vr);

	hydrazine::log("LiveRangeAnalysis") << "  Discovering live range for '"
		<< liveRange.virtualRegister()->toString() << "'\n";

	// in parallel
	for(auto use : liveRange.usingInstructions)
	{
		walkUpDataflowGraph(liveRange, dfg, cfg, use);
	}
}

}

}

