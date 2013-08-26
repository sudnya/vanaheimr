/*! \file   DataflowAnalysis.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the dataflow analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>
#include <vanaheimr/util/interface/LargeSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir       { class VirtualRegister;  } }
namespace vanaheimr { namespace ir       { class Instruction;      } }
namespace vanaheimr { namespace ir       { class BasicBlock;       } }
namespace vanaheimr { namespace analysis { class ControlFlowGraph; } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing dataflow analysis */	
class DataflowAnalysis : public FunctionAnalysis
{
public:
	typedef             ir::BasicBlock BasicBlock;
	typedef            ir::Instruction Instruction;
	typedef        ir::VirtualRegister VirtualRegister;
	typedef analysis::ControlFlowGraph ControlFlowGraph;

	typedef util::SmallSet<VirtualRegister*> VirtualRegisterSet;
	typedef util::SmallSet<Instruction*>     InstructionSet;


public:
	DataflowAnalysis();
	
public:
	VirtualRegisterSet  getLiveIns(const BasicBlock&);
	VirtualRegisterSet getLiveOuts(const BasicBlock&);

public:
	InstructionSet getReachingDefinitions(const Instruction&);
	InstructionSet getReachedUses(const Instruction&);

public:
	void setLiveOuts(const BasicBlock&, const VirtualRegisterSet&);

public:
	void addReachingDefinition(VirtualRegister&, Instruction&);
	
public:
	InstructionSet getReachingDefinitions(const VirtualRegister&);
	InstructionSet getReachedUses(const VirtualRegister&);
	
public:
	virtual void analyze(Function& function);
	
private:
	typedef std::vector<VirtualRegisterSet> VirtualRegisterSetVector;
	typedef std::vector<InstructionSet>     InstructionSetVector;
	typedef util::LargeSet<BasicBlock*>     BasicBlockSet;
		
private:
	void _analyzeLiveInsAndOuts(Function& function);
	void _analyzeReachingDefinitions(Function& function);

private:
	void _computeLocalLiveInsAndOuts(BasicBlockSet& worklist);
	bool _recomputeLiveInsAndOutsForBlock(BasicBlock* block);

private:
	VirtualRegisterSetVector _liveins;
	VirtualRegisterSetVector _liveouts;
	
	InstructionSetVector _reachingDefinitions;
	InstructionSetVector _reachedUses;
};

}

}


