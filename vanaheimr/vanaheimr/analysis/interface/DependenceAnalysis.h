/*! \file   DependenceAnalysis.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the dependence analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>
#include <vanaheimr/util/interface/LargeMap.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class Instruction; } }
namespace vanaheimr { namespace ir { class BasicBlock;  } }


namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing dependence analysis */	
class DependenceAnalysis : public FunctionAnalysis
{
public:
	typedef  ir::BasicBlock BasicBlock;
	typedef ir::Instruction Instruction;

	typedef util::SmallSet<Instruction*> InstructionSet;

public:
	DependenceAnalysis();
	
public:
	bool hasLocalDependence(const Instruction& predecessor,
		const Instruction& successor) const;
	bool hasDependence(const Instruction& predecessor,
		const Instruction& successor) const;

public:
	InstructionSet getLocalPredecessors(const Instruction& successor) const;
	InstructionSet getLocalSuccessors(const Instruction& predecessor) const;
	
public:
	virtual void analyze(Function& function);

private:
	typedef std::vector<InstructionSet>  InstructionSetVector;
	typedef util::LargeMap<unsigned int, InstructionSetVector>
		BlockToInstructionSetMap;

private:
	void _setLocalDependences(BasicBlock& block);

private:
	BlockToInstructionSetMap _localPredecessors;
	BlockToInstructionSetMap _localSuccessors;
};

}

}


