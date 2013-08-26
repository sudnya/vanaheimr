/*! \file   LiveRangeAnalysis.h
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the live-range analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class VirtualRegister;  } }
namespace vanaheimr { namespace ir { class Instruction;      } }
namespace vanaheimr { namespace ir { class BasicBlock;       } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing live range analysis */	
class LiveRangeAnalysis : public FunctionAnalysis
{
public:
	typedef      ir::BasicBlock BasicBlock;
	typedef     ir::Instruction Instruction;
	typedef ir::VirtualRegister VirtualRegister;

	typedef util::SmallSet<Instruction*> InstructionSet;
	typedef util::SmallSet<BasicBlock*>  BasicBlockSet;

	class LiveRange
	{
	public:
		LiveRange(LiveRangeAnalysis*, VirtualRegister*);
	
	public:
		LiveRangeAnalysis* liveRangeAnalysis() const;
		  VirtualRegister*   virtualRegister() const;

	public:
		BasicBlockSet allBlocksWithLiveValue() const;

	public:
		/* \brief Do live ranges interfere? */
		bool interferesWith(const LiveRange& range) const;

	public:
		BasicBlockSet fullyCoveredBlocks;

	public:
		InstructionSet definingInstructions;
		InstructionSet usingInstructions;	

	private:
		LiveRangeAnalysis* _analysis;
		VirtualRegister*   _virtualRegister;
	};

	typedef std::vector<LiveRange> LiveRangeVector;

	typedef LiveRangeVector::iterator       iterator;
	typedef LiveRangeVector::const_iterator const_iterator;

public:
	LiveRangeAnalysis();
	
public:
	const LiveRange* getLiveRange(const VirtualRegister&) const;
	      LiveRange* getLiveRange(const VirtualRegister&);
	
public:
	virtual void analyze(Function& function);

public:
	LiveRangeAnalysis(const LiveRangeAnalysis& ) = delete;
	LiveRangeAnalysis& operator=(const LiveRangeAnalysis& ) = delete;
	
public:
	      iterator begin();
	const_iterator begin() const;

	      iterator end();
	const_iterator end() const;

public:
	bool   empty() const;
	size_t  size() const;

private:
	void _initializeLiveRanges(ir::Function& );

private:
	LiveRangeVector _liveRanges;

};

typedef LiveRangeAnalysis::LiveRange LiveRange;

}

}



