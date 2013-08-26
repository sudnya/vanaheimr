/*! \file   InterferenceAnalysis.h
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InterferenceAnalysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class VirtualRegister;  } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing interference analysis */	
class InterferenceAnalysis : public FunctionAnalysis
{
public:
	typedef ir::VirtualRegister VirtualRegister;

	typedef util::SmallSet<VirtualRegister*> VirtualRegisterSet;

public:
	InterferenceAnalysis();
	
public:
	bool doLiveRangesInterfere(const VirtualRegister&,
		const VirtualRegister&) const;

public:
	VirtualRegisterSet&       getInterferences(const VirtualRegister&);
	const VirtualRegisterSet& getInterferences(const VirtualRegister&) const;	

public:
	virtual void analyze(Function& function);

public:
	InterferenceAnalysis(const InterferenceAnalysis& ) = delete;
	InterferenceAnalysis& operator=(const InterferenceAnalysis& ) = delete;
	
private:
	typedef std::vector<VirtualRegisterSet> VirtualRegisterSetVector;

private:
	VirtualRegisterSetVector _interferences;

};

}

}




