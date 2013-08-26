/*! \file   Pass.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Tuesday September 15, 2009
	\brief  The source file for the Pass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassManager.h>
#include <vanaheimr/transforms/interface/Pass.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

Pass::Pass(Type t, const StringVector& a, const std::string& n,
	const StringVector& c)
	: type(t), analyses(a), name(n), classes(c), _manager(0)
{

}

Pass::~Pass()
{

}

void Pass::setPassManager(PassManager* m)
{
	_manager = m;
}

Pass::Analysis* Pass::getAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

const Pass::Analysis* Pass::getAnalysis(const std::string& type) const
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

Pass* Pass::getPass(const std::string& name)
{
	assert(_manager != 0);

	return _manager->getPass(name);
}

const Pass* Pass::getPass(const std::string& name) const
{
	assert(_manager != 0);

	return _manager->getPass(name);
}

void Pass::invalidateAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->invalidateAnalysis(type);
}

Pass::StringVector Pass::getDependentPasses() const
{
	return StringVector();
}

void Pass::configure(const StringVector& options)
{

}

std::string Pass::toString() const
{
	return name;
}

ImmutablePass::ImmutablePass(const StringVector& a, const std::string& n,
	const StringVector& c) 
 : Pass(Pass::ImmutablePass, a, n, c)
{

}

ImmutablePass::~ImmutablePass()
{

}

ModulePass::ModulePass(const StringVector& a, const std::string& n,
	const StringVector& c) 
 : Pass( Pass::ModulePass, a, n, c)
{

}

ModulePass::~ModulePass()
{

}

FunctionPass::FunctionPass(const StringVector& a, const std::string& n,
	const StringVector& c)
 : Pass(Pass::FunctionPass, a, n, c)
{

}

FunctionPass::~FunctionPass()
{

}

void FunctionPass::initialize(const Module& m)
{

}

void FunctionPass::finalize()
{

}

ImmutableFunctionPass::ImmutableFunctionPass(
	const StringVector& a, const std::string& n, const StringVector& c)
 : Pass(Pass::ImmutableFunctionPass, a, n, c)
{

}

ImmutableFunctionPass::~ImmutableFunctionPass()
{

}

void ImmutableFunctionPass::initialize(const Module& m)
{

}

void ImmutableFunctionPass::finalize()
{

}

BasicBlockPass::BasicBlockPass(const StringVector& a, const std::string& n,
	const StringVector& c)
 : Pass(Pass::BasicBlockPass, a, n, c)
{

}

BasicBlockPass::~BasicBlockPass()
{

}

void BasicBlockPass::initialize(const Module& m)
{

}

void BasicBlockPass::initialize(const Function& m)
{

}

void BasicBlockPass::finalizeFunction()
{

}

void BasicBlockPass::finalize()
{

}

}

}

