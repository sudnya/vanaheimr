/*! \file   AnalysisFactory.cpp
	\date   Wednesday October 3, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the AnalysisFactory class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/AnalysisFactory.h>

#include <vanaheimr/analysis/interface/ControlFlowGraph.h>
#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/DominatorAnalysis.h>
#include <vanaheimr/analysis/interface/ReversePostOrderTraversal.h>
#include <vanaheimr/analysis/interface/DependenceAnalysis.h>
#include <vanaheimr/analysis/interface/LiveRangeAnalysis.h>
#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

namespace vanaheimr
{

namespace analysis 
{

Analysis* AnalysisFactory::createAnalysis(const std::string& name,
	const StringVector& options)
{
	Analysis* analysis = nullptr;

	if(name == "ControlFlowGraph")
	{
		analysis = new ControlFlowGraph;
	}
	else if (name == "DataflowAnalysis")
	{
		analysis = new DataflowAnalysis;
	}
	else if (name == "DominatorAnalysis")
	{
		analysis = new DominatorAnalysis;
	}
	else if (name == "ReversePostOrderTraversal")
	{
		analysis = new ReversePostOrderTraversal;
	}
	else if (name == "DependenceAnalysis")
	{
		analysis = new DependenceAnalysis;
	}
	else if (name == "LiveRangeAnalysis")
	{
		analysis = new LiveRangeAnalysis;
	}
	else if (name == "InterferenceAnalysis")
	{
		analysis = new InterferenceAnalysis;
	}

	if(analysis != nullptr)
	{
		analysis->configure(options);
	}
	
	return analysis;
}

}

}

