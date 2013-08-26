/*! \file   PassFactory.cpp
	\date   Wednesday May 2, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the PassFactory class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassFactory.h>

#include <vanaheimr/transforms/interface/ConvertToSSAPass.h>
#include <vanaheimr/transforms/interface/ConvertFromSSAPass.h>

#include <vanaheimr/codegen/interface/EnforceArchaeopteryxABIPass.h>
#include <vanaheimr/codegen/interface/ListInstructionSchedulerPass.h>
#include <vanaheimr/codegen/interface/ChaitinBriggsRegisterAllocatorPass.h>
#include <vanaheimr/codegen/interface/GenericSpillCodePass.h>
#include <vanaheimr/codegen/interface/TranslationTableInstructionSelectionPass.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace transforms
{

class PassDatabase
{
public:
	typedef std::map<std::string, Pass*> PassMap;

public:
	PassMap passes;

public:
	~PassDatabase();
};

PassDatabase::~PassDatabase()
{
	for(auto pass : passes)
	{
		delete pass.second;
	}
}

static PassDatabase passDatabase;

Pass* PassFactory::createPass(const std::string& name,
	const StringVector& options)
{
	Pass* pass = nullptr;

	if(name == "ConvertToSSA" || name == "ConvertToSSAPass")
	{
		pass = new ConvertToSSAPass();
	}
	
	if(name == "ConvertFromSSA" || name == "ConvertFromSSAPass")
	{
		pass = new ConvertFromSSAPass();
	}
	
	if(name == "EnforceArchaeopteryxABIPass")
	{
		pass = new codegen::EnforceArchaeopteryxABIPass();
	}
	
	if(name == "ListInstructionSchedulerPass" || name == "list")
	{
		pass = new codegen::ListInstructionSchedulerPass();
	}
	
	if(name == "chaitin-briggs" || name == "ChaitinBriggsRegisterAllocatorPass")
	{
		pass = new codegen::ChaitinBriggsRegisterAllocatorPass();
	}
	
	if(name == "generic-spiller" || name == "GenericSpillCodePass")
	{
		pass = new codegen::GenericSpillCodePass();
	}
	
	if(name == "translation-table" ||
		name == "TranslationTableInstructionSelectionPass")
	{
		pass = new codegen::TranslationTableInstructionSelectionPass();
	}

	auto databaseEntry = passDatabase.passes.find(name);

	if(databaseEntry != passDatabase.passes.end())
	{
		pass = databaseEntry->second->clone();
	}
	
	if(pass != nullptr)
	{
		pass->configure(options);
	}
	
	return pass;
}

void PassFactory::registerPass(const Pass* newPass)
{
	assert(passDatabase.passes.count(newPass->name) == 0);

	passDatabase.passes.insert(std::make_pair(newPass->name, newPass->clone()));
}

}

}

