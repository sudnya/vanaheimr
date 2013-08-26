/*! \file   Analysis.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Saturday May 7, 2011
	\brief  The source file for the Analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/transforms/interface/PassManager.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace analysis
{

Analysis::Analysis(Type t, const std::string& n, const StringVector& r)
: type(t), name(n), required(r), _manager(0)
{

}

Analysis::~Analysis()
{

}

FunctionAnalysis::FunctionAnalysis(const std::string& n, const StringVector& r)
: Analysis(Analysis::FunctionAnalysis, n, r)
{

}

ModuleAnalysis::ModuleAnalysis(const std::string& n, const StringVector& r)
: Analysis(Analysis::ModuleAnalysis, n, r)
{

}

void Analysis::setPassManager(transforms::PassManager* m)
{
	_manager = m;
}

Analysis* Analysis::getAnalysis(const std::string& name)
{
	assert(_manager != 0);
	return _manager->getAnalysis(name);
}

const Analysis* Analysis::getAnalysis(const std::string& name) const
{
	assert(_manager != 0);
	return _manager->getAnalysis(name);
}

void Analysis::invalidateAnalysis(const std::string& name)
{
	assert(_manager != 0);
	_manager->invalidateAnalysis(name);
}

void Analysis::configure(const StringVector&)
{

}

}

}

