/*! \file   DependenceAnalysis.cpp
	\date   Tuesday January 8, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the dependence analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DependenceAnalysis.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Instruction.h>

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

namespace analysis
{

DependenceAnalysis::DependenceAnalysis()
: FunctionAnalysis("DependenceAnalysis", {"ControlFlowGraph"})
{

}

bool DependenceAnalysis::hasLocalDependence(const Instruction& predecessor,
	const Instruction& successor) const
{
	auto predecessors = getLocalPredecessors(successor);
	
	return predecessors.count(const_cast<Instruction*>(&predecessor)) != 0;
}

bool DependenceAnalysis::hasDependence(const Instruction& predecessor,
	const Instruction& successor) const
{
	assertM(false, "not implemented");
}

DependenceAnalysis::InstructionSet DependenceAnalysis::getLocalPredecessors(
	const Instruction& successor) const
{
	auto block = _localPredecessors.find(successor.block->id());
	
	if(block == _localPredecessors.end()) return InstructionSet();
	
	assert(successor.index() < block->second.size());
	
	return block->second[successor.index()];
}

DependenceAnalysis::InstructionSet DependenceAnalysis::getLocalSuccessors(
	const Instruction& predecessor) const
{
	auto block = _localSuccessors.find(predecessor.block->id());
	
	if(block == _localSuccessors.end()) return InstructionSet();
	
	assert(predecessor.index() < block->second.size());
	
	return block->second[predecessor.index()];
}

void DependenceAnalysis::analyze(Function& function)
{
	report("Running dependence analysis on '" << function.name() << "'");

	// for all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		_setLocalDependences(*block);
	}
}

typedef DependenceAnalysis::InstructionSet InstructionSet;

static void addPredecessors(InstructionSet& predecessors,
	ir::BasicBlock::const_iterator instruction);

void DependenceAnalysis::_setLocalDependences(BasicBlock& block)
{
	report(" for basic block '" << block.name() << "'");

	auto predecessor = _localPredecessors.insert(std::make_pair(block.id(),
			InstructionSetVector())).first;
	auto successor   =   _localSuccessors.insert(std::make_pair(block.id(),
		InstructionSetVector())).first;
		
	predecessor->second.resize(block.size());
	  successor->second.resize(block.size());
	
	if(block.empty()) return;
	
	// TODO: do this with a prefix scan
	auto instruction = block.begin();
	for(++instruction; instruction != block.end(); ++instruction)
	{
		InstructionSet& instructionPredecessors =
			predecessor->second[(*instruction)->index()];
	
		addPredecessors(instructionPredecessors, instruction);
	}
	
	// TODO: collect successors in parallel
	for(auto instruction : block)
	{
		InstructionSet& instructionPredecessors =
			predecessor->second[instruction->index()];
		
		for(auto predecessor : instructionPredecessors)
		{
			InstructionSet& instructionSuccessors =
				successor->second[predecessor->index()];
		
			instructionSuccessors.insert(instruction);
		}
	}
}

static bool hasDataflowDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	typedef util::SmallSet<ir::VirtualRegister*> VirtualRegisterSet;

	VirtualRegisterSet writes;

	for(auto write : predecessor.writes)
	{
		if(!write->isRegister()) continue;
	
		auto registerOperand = static_cast<ir::RegisterOperand*>(write);
	
		writes.insert(registerOperand->virtualRegister);
	}
	
	for(auto read : successor.reads)
	{
		if(!read->isRegister()) continue;
	
		auto registerOperand = static_cast<ir::RegisterOperand*>(read);
	
		if(writes.count(registerOperand->virtualRegister) != 0) return true;
	}
	
	return false;
}

static bool hasControlflowDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return (predecessor.isBranch() && !predecessor.isIntrinsic()) ||
		(successor.isBranch() && !successor.isIntrinsic());
}

static bool hasBarrierDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return (predecessor.isMemoryBarrier() && successor.accessesMemory()) ||
		(predecessor.isMemoryBarrier() && successor.accessesMemory());
}

static bool hasMemoryDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return (predecessor.accessesMemory() && successor.isStore()) ||
		(predecessor.isStore() && successor.accessesMemory());
}

static bool hasDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	if(hasControlflowDependence(predecessor, successor)) return true;
	if(    hasBarrierDependence(predecessor, successor)) return true;
	if(     hasMemoryDependence(predecessor, successor)) return true;
	if(   hasDataflowDependence(predecessor, successor)) return true;
	
	return false;
}

static void addPredecessors(InstructionSet& predecessors,
	ir::BasicBlock::const_iterator instruction)
{
	auto      end = (*instruction)->block->rend();
	auto position = ir::BasicBlock::const_reverse_iterator(instruction);
	
	for(; position != end; ++position)
	{
		if(!hasDependence(**position, **instruction)) continue;
		
		report("  " << (*position)->toString() << " (" << (*position)->index()
			<< ") -> " << (*instruction)->toString() << " ("
			<< (*instruction)->index() << ")");
		
		predecessors.insert(*position);
	}
}

}

}


