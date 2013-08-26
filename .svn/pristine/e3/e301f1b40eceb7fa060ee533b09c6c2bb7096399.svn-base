/*! \file   ConvertToSSAPass.h
	\date   Tuesday September 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the ConvertToSSAPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

#include <vanaheimr/util/interface/LargeSet.h>
#include <vanaheimr/util/interface/SmallMap.h>
#include <vanaheimr/util/interface/SmallSet.h>

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace ir { class VirtualRegister;  } }
namespace vanaheimr { namespace ir { class Instruction;      } }
namespace vanaheimr { namespace ir { class BasicBlock;       } }

namespace vanaheimr { namespace analysis { class ControlFlowGraph;  } }
namespace vanaheimr { namespace analysis { class DataflowAnalysis;  } }
namespace vanaheimr { namespace analysis { class DominatorAnalysis; } }

namespace vanaheimr
{

namespace transforms
{

/*! \brief Convert a program IR not in SSA form to SSA */
class ConvertToSSAPass : public FunctionPass
{
public:
	ConvertToSSAPass();

public:
	virtual void runOnFunction(Function& f);

public:
	virtual Pass* clone() const;

private:
	typedef ir::VirtualRegister VirtualRegister;
	typedef ir::BasicBlock      BasicBlock;
	typedef ir::Instruction     Instruction;

private:
	typedef analysis::ControlFlowGraph  ControlFlowGraph;
	typedef analysis::DataflowAnalysis  DataflowAnalysis;
	typedef analysis::DominatorAnalysis DominatorAnalysis;

private:
	typedef util::LargeSet<VirtualRegister*> VirtualRegisterSet;
	typedef util::LargeSet<BasicBlock*> BasicBlockSet;
	typedef util::SmallSet<BasicBlock*> SmallBlockSet;
	
	typedef util::SmallMap<VirtualRegister*, VirtualRegister*>
		VirtualRegisterMap;

private:
	void _insertPhis(Function& f);
	void _insertPsis(Function& f);
	
	void _insertPhi(VirtualRegister& vr, BasicBlock& block);

private:
	void _rename(Function& f);

	void _renameAllDefs(VirtualRegister& vr, BasicBlockSet& worklist);
	
	void _updateDefinition(Instruction& definingInstruction,
		VirtualRegister& value, VirtualRegister& newValue);
	bool _updateUsesInThisBlock(Instruction& definingInstruction,
		VirtualRegister& value, VirtualRegister& newValue);
	
	void _renameLocalBlocks(BasicBlockSet& worklist);
	void _renameValuesInBlock(BasicBlockSet& worklist, BasicBlock* block);

	void _renamePhiInputs(BasicBlock* block, VirtualRegisterMap& renamedValues);
	
private:
	SmallBlockSet _getBlocksThatDefineThisValue(const ir::VirtualRegister&);

private:
	
	typedef std::vector<VirtualRegisterMap> VirtualRegisterMapVector;
	
private:
	/*! \brief Registers that are used by a PHI or PSI */
	VirtualRegisterSet _registersNeedingRenaming;
	
	VirtualRegisterMapVector _renamedLiveIns;
	VirtualRegisterMapVector _renamedLiveOuts;
	

};

}

}

