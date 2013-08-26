/*! \file   ArchaeopteryxTarget.cpp
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ArchaeopteryxTarget class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ArchaeopteryxTarget.h>

#include <vanaheimr/transforms/interface/PassManager.h>
#include <vanaheimr/transforms/interface/PassFactory.h>

#include <vanaheimr/ir/interface/Module.h>

// Standard Library Includes
#include <stdexcept>

namespace vanaheimr
{

namespace codegen
{

ArchaeopteryxTarget::ArchaeopteryxTarget()
: Target("ArchaeopteryxSimulatorTarget"),
	instructionSelectorName("translation-table"),
	registerAllocatorName("chaitin-briggs"), instructionSchedulerName("list")
{

}

void ArchaeopteryxTarget::lower()
{
	transforms::PassManager manager(_module);

	// Instruction Selection
	auto selector = transforms::PassFactory::createPass(
		instructionSelectorName);
	
	if(selector == nullptr)
	{
		throw std::runtime_error("Failed to create archaeopteryx"
			" instruction selection pass '" + instructionSelectorName + "'.");
	}

	manager.addPass(selector);

	// ABI Lowering	
	auto abiLowering = transforms::PassFactory::createPass(
		"EnforceArchaeopteryxABIPass");

	if(abiLowering == nullptr)
	{
		throw std::runtime_error("Failed to create archaeopteryx"
			" ABI lowering pass.");
	}

	manager.addPass(abiLowering);

	// Instruction Scheduler
	auto scheduler = transforms::PassFactory::createPass(
		instructionSchedulerName);
	
	if(scheduler == nullptr)
	{
		throw std::runtime_error("Failed to get instruction scheduler '" +
			instructionSchedulerName +"'");
	}

	manager.addPass(scheduler);
	
	// Register Allocator
	auto allocator = transforms::PassFactory::createPass(registerAllocatorName);	
	
	if(allocator == nullptr)
	{
		throw std::runtime_error("Failed to get register allocator '" +
			registerAllocatorName +"'");
	}

	manager.addPass(allocator);
	
	// Register Spiller
	auto spiller = transforms::PassFactory::createPass("GenericSpillCodePass");

	if(spiller == nullptr)
	{
		throw std::runtime_error("Failed to create spill code pass.");
	}
	
	manager.addPass(spiller);

	manager.addDependence(abiLowering->name, selector->name);
	manager.addDependence(scheduler->name,   abiLowering->name);
	manager.addDependence(allocator->name,   scheduler->name);
	manager.addDependence(spiller->name,     allocator->name);
	
	manager.runOnModule();
}

ir::ModuleBase* ArchaeopteryxTarget::getLoweredModule()
{
	return _module;
}

Target* ArchaeopteryxTarget::clone() const
{
	return new ArchaeopteryxTarget;
}

}

}


