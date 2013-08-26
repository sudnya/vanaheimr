/*! \file   DataflowAnalysis.cpp
	\date   Friday September 14, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the dataflow analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

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

DataflowAnalysis::DataflowAnalysis()
: FunctionAnalysis("DataflowAnalysis", StringVector(1, "ControlFlowGraph"))
{

}

DataflowAnalysis::VirtualRegisterSet
	DataflowAnalysis::getLiveIns(const BasicBlock& block)
{
	assert(block.id() < _liveins.size());
	
	return _liveins[block.id()];
}

DataflowAnalysis::VirtualRegisterSet
	DataflowAnalysis::getLiveOuts(const BasicBlock& block)
{
	assert(block.id() < _liveouts.size());
	
	return _liveouts[block.id()];
}

DataflowAnalysis::InstructionSet
	DataflowAnalysis::getReachingDefinitions(const Instruction& instruction)
{
	DataflowAnalysis::InstructionSet definitions;
	
	for(auto write : instruction.writes)
	{
		auto writeOperand = static_cast<ir::RegisterOperand*>(write);
	
		auto localDefinitions = getReachingDefinitions(
			*writeOperand->virtualRegister);
	
		definitions.insert(localDefinitions.begin(), localDefinitions.end());
	}
	
	return definitions;
}

DataflowAnalysis::InstructionSet 
	DataflowAnalysis::getReachedUses(const Instruction& instruction)
{
	DataflowAnalysis::InstructionSet uses;
	
	for(auto read : instruction.reads)
	{
		if(!read->isRegister()) continue;
	
		auto readOperand = static_cast<ir::RegisterOperand*>(read);
	
		auto localUses = getReachedUses(*readOperand->virtualRegister);
	
		uses.insert(localUses.begin(), localUses.end());
	}
	
	return uses;
}

void DataflowAnalysis::setLiveOuts(const BasicBlock& block,
	const VirtualRegisterSet& liveOuts)
{
	assert(block.id() < _liveouts.size());
	
	_liveouts[block.id()] = liveOuts;
}

void DataflowAnalysis::addReachingDefinition(VirtualRegister& value,
	Instruction& instruction)
{
	_reachingDefinitions[value.id].insert(&instruction);
}

DataflowAnalysis::InstructionSet DataflowAnalysis::getReachingDefinitions(
	const VirtualRegister& value)
{
	assert(value.id < _reachingDefinitions.size());

	return _reachingDefinitions[value.id];
}

DataflowAnalysis::InstructionSet
	DataflowAnalysis::getReachedUses(const VirtualRegister& value)
{
	assert(value.id < _reachedUses.size());
	
	return _reachedUses[value.id];
}

void DataflowAnalysis::analyze(Function& function)
{
	     _analyzeLiveInsAndOuts(function);
	_analyzeReachingDefinitions(function);
}

void DataflowAnalysis::_analyzeLiveInsAndOuts(Function& function)
{
	 _liveins.resize(function.size());
	_liveouts.resize(function.size());
	
	BasicBlockSet worklist;
	
	// should be for-all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		worklist.insert(&*block);
	}
	
	while(!worklist.empty())
	{
		_computeLocalLiveInsAndOuts(worklist);
	}
}

void DataflowAnalysis::_analyzeReachingDefinitions(Function& function)
{
	// For each instruction, create a writer set and a reader set
	//  gather them together 
	
	_reachingDefinitions.clear();
	        _reachedUses.clear();
	
	_reachingDefinitions.resize(function.register_size());
	        _reachedUses.resize(function.register_size());
	
	
	// parallel for-all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		for(auto instruction : *block)
		{
			VirtualRegisterSet localDefinitions;
			VirtualRegisterSet localUses;
		
			// find values that are written to
			for(auto write : instruction->writes)
			{
				if(!write->isRegister()) continue;
			
				auto registerWrite = static_cast<ir::RegisterOperand*>(write);
			
				localDefinitions.insert(registerWrite->virtualRegister);
			}

			// find values that are read from
			for(auto read : instruction->reads)
			{
				if(!read->isRegister()) continue;
			
				auto registerRead = static_cast<ir::RegisterOperand*>(read);
			
				localUses.insert(registerRead->virtualRegister);
			}
		
			// insert the instruction into the read value's user set
			for(auto readValue : localUses)
			{
				_reachedUses[readValue->id].insert(instruction);
			}
		
			// insert the instruction into the written value's def set
			for(auto writtenValue : localDefinitions)
			{
				_reachingDefinitions[writtenValue->id].insert(instruction);
			}
		}
	}
}

void DataflowAnalysis::_computeLocalLiveInsAndOuts(BasicBlockSet& worklist)
{
	BasicBlockSet newList;

	// should be for-all
	for(auto block : worklist)
	{
		bool changed = _recomputeLiveInsAndOutsForBlock(block);

		if(changed)
		{
			// TODO: queue up predecessors
			newList.insert(block);
		}
	}

	// gather blocks to form the new worklist
	worklist = std::move(newList);
}	

bool DataflowAnalysis::_recomputeLiveInsAndOutsForBlock(BasicBlock* block)
{
	// live outs is the union of live-ins of all successors
	VirtualRegisterSet liveout;

	auto cfg = static_cast<ControlFlowGraph*>(getAnalysis("ControlFlowGraph"));	

	auto successors = cfg->getSuccessors(*block);

	for(auto successor : successors)
	{
		auto livein = getLiveIns(*successor);

		liveout.insert(livein.begin(), livein.end());
	}

	_liveouts[block->id()] = liveout;

	VirtualRegisterSet livein = std::move(liveout);

	// apply def/use rules in reverse order
	for(auto instruction = block->rbegin(); instruction != block->rend();
		++instruction)
	{
		// spawn on uses
		for(auto read : (*instruction)->reads)
		{
			if(!read->isRegister()) continue;
		
			auto reg = static_cast<ir::RegisterOperand*>(read);

			livein.insert(reg->virtualRegister);
		}

		// kill on defs
		for(auto write : (*instruction)->writes)
		{
			if(!write->isRegister()) continue;
		
			auto reg = static_cast<ir::RegisterOperand*>(write);

			livein.erase(reg->virtualRegister);
		}
	}

	bool changed = _liveins[block->id()] != livein;

	_liveins[block->id()] = std::move(livein);
	
	return changed;
}

}

}


