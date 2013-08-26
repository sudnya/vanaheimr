/*! \file   ConvertFromSSAPass.cpp
	\date   Tuesday November 20, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ConvertFromSSAPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/ConvertFromSSAPass.h>

#include <vanaheimr/analysis/interface/DataflowAnalysis.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/Instruction.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

ConvertFromSSAPass::ConvertFromSSAPass()
: FunctionPass(StringVector({"DataflowAnalysis"}),
  "ConvertFromSSAPass")
{
	
}

void ConvertFromSSAPass::runOnFunction(Function& f)
{
	_removePhis(f);
	_removePsis(f);
}

Pass* ConvertFromSSAPass::clone() const
{
	return new ConvertFromSSAPass;
}

void ConvertFromSSAPass::_removePhis(Function& f)
{
	// TODO split critical edges

	// for all
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		_removePhis(*block);
	}
}

void ConvertFromSSAPass::_removePsis(Function& f)
{
	// for all
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		// for all?
		for(auto instruction : *block)
		{
			if(!instruction->isPsi()) continue;
			
			_removePsi(static_cast<ir::Psi&>(*instruction));
		}
	}
}

static ir::BasicBlock::iterator getFirstNonPhiInstruction(ir::BasicBlock& block)
{
	for(auto instruction = block.begin();
		instruction != block.end(); ++instruction)
	{
		if(!(*instruction)->isPhi()) return instruction;
	}
	
	return block.end();
}

typedef util::SmallSet<ir::Phi*> PhiSet;

static ir::VirtualRegister* lookupOrCreatePhiSourceRegister(
	ir::Phi* phi, PhiSet& phis)
{
	// TODO actually look for a duplicate phi
	phis.insert(phi);
	
	auto newRegister = phi->block->function()->newVirtualRegister(
		phi->d()->type());

	return &*newRegister;
}

typedef analysis::DataflowAnalysis DataflowAnalysis;

static ir::BasicBlock::iterator determineInsertionPosition(
	ir::BasicBlock* block, ir::VirtualRegister* source, DataflowAnalysis* dfg)
{
	if(block->empty()) return block->begin();

	// Discover defs/uses in the block
	auto defs = dfg->getReachingDefinitions(*source);
	auto uses = dfg->getReachedUses(*source);
	
	DataflowAnalysis::InstructionSet defUsesInThisBlock;
	
	for(auto def : defs)
	{
		if(def->block == block) defUsesInThisBlock.insert(def);
	}
	
	for(auto use : uses)
	{
		if(use->block == block) defUsesInThisBlock.insert(use);
	}
	
	// start at the beginning
	auto position = block->begin();
	
	if(!defUsesInThisBlock.empty())
	{
		// rewind from the end
		position = block->end();
		
		while(defUsesInThisBlock.count(*--position) == 0);
		
		++position;
	}
	
	// skip phis
	if((*position)->isPhi())
	{
		position = getFirstNonPhiInstruction(*block);
	}
	
	return position;	
}

static void removeFirstPhi(ir::BasicBlock& block,
	const ir::BasicBlock::iterator& phiEnd, PhiSet& phis, DataflowAnalysis* dfg)
{
	auto phi = static_cast<ir::Phi*>(block.front());

	//  Try to reuse an existing phi (updates phi set)
	auto sourceRegister = lookupOrCreatePhiSourceRegister(phi, phis);

	// Create a register-to-register copy of a source into the PHI destination
	auto copy = new ir::Bitcast;
	
	copy->setGuard(new ir::PredicateOperand(
		ir::PredicateOperand::PredicateTrue, copy));
	copy->setD(new ir::RegisterOperand(phi->d()->virtualRegister, copy));
	copy->setA(new ir::RegisterOperand(sourceRegister, copy));

	// Insert the copy before the non-phi first instruction
	block.insert(phiEnd, copy);

	// Add move instructions of the PHI source values into the copied
	//  register in the predecessors
	auto sources = phi->sources();
	auto blocks  = phi->blocks();
	
	assert(sources.size() == blocks.size());
	
	auto predecessor = blocks.begin();
	for(auto source = sources.begin(); source != sources.end();
		++source, ++predecessor)
	{
		auto insertPosition = determineInsertionPosition(*predecessor,
			(*source)->virtualRegister, dfg);
	
		auto copy = new ir::Bitcast;
	
		copy->setGuard(new ir::PredicateOperand(
			ir::PredicateOperand::PredicateTrue, copy));
		copy->setD(new ir::RegisterOperand(sourceRegister, copy));
		copy->setA(new ir::RegisterOperand((*source)->virtualRegister, copy));

		(*predecessor)->insert(insertPosition, copy);
	}
	
	// Delete the phi from the block
	block.pop_front();	
}

void ConvertFromSSAPass::_removePhis(ir::BasicBlock& block)
{
	// get the first instruction after the phi
	auto phiEnd = getFirstNonPhiInstruction(block);
	
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
		
	// possibly do this in parallel, but have to be careful about dependencies
	while(!block.empty() && block.front()->isPhi())
	{
		PhiSet phis;
		
		removeFirstPhi(block, phiEnd, phis, dfg);
		
		//  1) It will be necessary to destroy these in a post-pass after
		//      removing the phis from blocks first to avoid read-after-deletes
		for(auto phi : phis)
		{
			delete phi;
		}
	}
}

void ConvertFromSSAPass::_removePsi(ir::Psi& psi)
{
	assertM(false, "Not implemented.");
}


}

}


