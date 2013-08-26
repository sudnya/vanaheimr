/*! \file   EnforceArchaeopteryxABIPass.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Friday December 21, 2021
	\brief  The source file for the EnforceArchaeopteryxABIPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/EnforceArchaeopteryxABIPass.h>

#include <vanaheimr/abi/interface/ApplicationBinaryInterface.h>

#include <vanaheimr/ir/interface/Module.h>
#include <vanaheimr/ir/interface/Type.h>

#include <vanaheimr/util/interface/LargeMap.h>
#include <vanaheimr/util/interface/SmallMap.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cstring>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace codegen
{

EnforceArchaeopteryxABIPass::EnforceArchaeopteryxABIPass()
: ModulePass({}, "EnforceArchaeopteryxABIPass")
{

}

typedef util::LargeMap<std::string, uint64_t> GlobalToAddressMap;
typedef util::SmallMap<std::string, uint64_t>  LocalToAddressMap;

static void layoutGlobals(ir::Module& module, GlobalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi);
static void layoutLocals(ir::Function& function, LocalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi);
static void layoutArguments(ir::Function& function, LocalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi);
static void lowerFunction(ir::Function& function,
	const abi::ApplicationBinaryInterface& abi,
	const GlobalToAddressMap& globals, const LocalToAddressMap& locals);

static const abi::ApplicationBinaryInterface* getABI();

void EnforceArchaeopteryxABIPass::runOnModule(Module& m)
{
	report("Lowering " << m.name << " to target the archaeopteryx ABI.");

	const abi::ApplicationBinaryInterface* abi = getABI();

	GlobalToAddressMap globals;

	layoutGlobals(m, globals, *abi);
	
	// barrier
	report(" Lowering functions...");
	
	// For-all
	for(auto function = m.begin(); function != m.end(); ++function)
	{
		LocalToAddressMap locals;
	
		layoutLocals(*function, locals, *abi);
		layoutArguments(*function, locals, *abi);
		
		// barrier

		lowerFunction(*function, *abi, globals, locals);
	}
}

transforms::Pass* EnforceArchaeopteryxABIPass::clone() const
{
	return new EnforceArchaeopteryxABIPass;
}

static unsigned int align(unsigned int address, unsigned int alignment)
{
	unsigned int remainder = address % alignment;
	unsigned int offset = remainder == 0 ? 0 : alignment - remainder;
	
	return address + offset;	
}

static void layoutGlobals(ir::Module& module, GlobalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi)
{
	unsigned int offset = 0;

	report(" Lowering globals...");

	for(auto global = module.global_begin();
		global != module.global_end(); ++global)
	{
		offset = align(offset, global->type().alignment());
		
		report("  Laying out '" << global->name() << "' at " << offset);
		
		globals.insert(std::make_pair(global->name(), offset));

		offset += global->bytes();
	}
}

static void layoutLocals(ir::Function& function, LocalToAddressMap& locals,
	const abi::ApplicationBinaryInterface& abi)
{
	assertM(function.local_empty(), "Lowering locals not implemented.");
}

static void layoutArguments(ir::Function& function, LocalToAddressMap& locals,
	const abi::ApplicationBinaryInterface& abi)
{
	// functions put their parameters on the stack
	if(!function.hasAttribute("kernel")) return;

	report(" Lowering parameters...");

	unsigned int offset = 0;

	for(auto argument = function.argument_begin();
		argument != function.argument_end(); ++argument)
	{
		offset = align(offset, argument->type().alignment());
	
		report("  Laying out '" << argument->name() << "' at " << offset);
		
		locals.insert(std::make_pair(argument->name(), offset));
		
		offset += argument->type().bytes();
	}
	
	// Kernels shouldn't return anything
	assert(function.returned_empty());
}

static void lowerCall(ir::Instruction* i,
	const abi::ApplicationBinaryInterface& abi)
{
	assertM(false, "call lowering not implemented.");
}

static void getVariable(const abi::BoundVariable& variable,
	ir::Operand* destination)
{
	auto    block = destination->instruction->block;
	auto function = block->function();

	switch(variable.binding())
	{
	case abi::BoundVariable::Register:
	{
		const abi::RegisterBoundVariable& registerBinding =
			static_cast<const abi::RegisterBoundVariable&>(variable);
	
		auto move = new ir::Bitcast(block);

		auto vr = function->findVirtualRegister(registerBinding.registerName);
		
		if(vr == function->register_end())
		{
			vr = function->newVirtualRegister(registerBinding.type,
				registerBinding.registerName);
		}

		move->setGuard(new ir::PredicateOperand(
			ir::PredicateOperand::PredicateTrue, move));
		move->setD(destination);
		move->setA(new ir::RegisterOperand(&*vr, move));

		report("    to " << move->toString());

		block->insert(destination->instruction, move);

		break;
	}
	case abi::BoundVariable::Memory:
	{
		assertM(false, "Memory bound variables not implemented.");
		break;
	}
	}
}

static bool tryLoweringSpecialRegisterAccess(ir::Instruction* i,
	const abi::ApplicationBinaryInterface& abi)
{
	typedef std::map<std::string, std::string> StringMap;

	auto call = static_cast<ir::Call*>(i);

	assert(call->target()->isAddress());

	auto targetOperand = static_cast<ir::AddressOperand*>(call->target());

	auto specifier = "_Zintrinsic_getspecial_";

	if(targetOperand->globalValue->name().find(specifier) != 0)
	{
		return false;
	}

	report("   Trying to lower special register access " << i->toString());

	auto special = targetOperand->globalValue->name().substr(
		std::strlen(specifier));

	auto variable = abi.findVariable(special);

	if(variable == nullptr)
	{
		report("    could not find special variable '" << special
			<< "' in the machine model." );
	
		return false;
	}
	
	assert(call->returned().size() == 1);

	auto returned = call->returned();

	getVariable(*variable, returned.back()->clone());

	call->eraseFromBlock();
	
	return true;
}

static bool isSupportedIntrinsic(ir::Instruction* i,
	const abi::ApplicationBinaryInterface& abi)
{
	// TODO: query the machine model to match the intrinsic name

	return true;
}

static void lowerIntrinsic(ir::Instruction* i,
	const abi::ApplicationBinaryInterface& abi)
{
	if(tryLoweringSpecialRegisterAccess(i, abi)) return;
	if(isSupportedIntrinsic(i, abi))             return;

	// TODO add other intrinsics
	assertM(false, "Lowering not implemented for intrisic - " << i->toString());
}

static void lowerReturn(ir::Instruction* i,
	const abi::ApplicationBinaryInterface& abi)
{
	if(i->block->function()->hasAttribute("kernel")) return;

	assertM(false, "function return lowering not implemented.");
}

static void lowerAddress(ir::Operand*& read, const GlobalToAddressMap& globals,
	const LocalToAddressMap& locals)
{
	report("   Lowering address read '" << read->toString()
		<< "' in instruction '"
		<< read->instruction->toString() << "'");

	auto variableRead = static_cast<ir::AddressOperand*>(read);

	auto local = locals.find(variableRead->globalValue->name());
	
	if(local != locals.end())
	{
		auto immediate = new ir::ImmediateOperand(local->second,
			read->instruction, read->type());

		read = immediate;

		report("    to '" << read->toString() << "'");

		delete variableRead;
		return;
	}

	auto global = globals.find(variableRead->globalValue->name());

	assertM(global != globals.end(), "'" << variableRead->globalValue->name()
		<< "' not lowered correctly.");

	auto immediate = new ir::ImmediateOperand(global->second,
		read->instruction, read->type());

	read = immediate;

	report("    to '" << read->toString() << "'");

	delete variableRead;

}

typedef abi::ApplicationBinaryInterface::FixedAddressRegion FixedAddressRegion;

static void lowerArgument(ir::Operand*& read, const LocalToAddressMap& locals,
	const abi::ApplicationBinaryInterface& abi)
{

	report("   Lowering argument read '" << read->toString()
		<< "' in instruction '"
		<< read->instruction->toString() << "'");

	auto argumentRead = static_cast<ir::ArgumentOperand*>(read);

	auto local = locals.find(argumentRead->argument->name());
	assert(local != locals.end());
	
	auto region = abi.findRegion("parameter");
	assert(region != nullptr);
	
	if(region->isFixed())
	{
		auto fixedRegion = static_cast<const FixedAddressRegion*>(region);

		auto immediate = new ir::ImmediateOperand(
			local->second + fixedRegion->address,
			read->instruction, read->type());

		read = immediate;

		report("    to '" << read->toString() << "'");

		delete argumentRead;
	}
	else
	{
		assertM(false, "Not implemented.");
	}
}

static void lowerEntryPoint(ir::Function& function, 
	const abi::ApplicationBinaryInterface& abi)
{
	// kernels don't need explicit entry point code
	if(function.hasAttribute("kernel")) return;

	assertM(false, "Entry point handling for called "
		"functions is not implemented yet");
}

static void lowerFunction(ir::Function& function,
	const abi::ApplicationBinaryInterface& abi,
	const GlobalToAddressMap& globals, const LocalToAddressMap& locals)
{
	if(function.isIntrinsic()) return;

	report("  Lowering function '" << function.name() << "'");
	
	// add an entry point
	lowerEntryPoint(function, abi);

	// for all 
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		for(auto instruction = block->begin(); instruction != block->end(); )
		{
			auto current = *instruction; ++instruction;
		
			// lower calls
			if(current->isCall())
			{
				if(current->isIntrinsic())
				{
					lowerIntrinsic(current, abi);
					continue;
				}
				else
				{
					lowerCall(current, abi);
					continue;
				}
			}
			
			// lower returns
			if(current->isReturn())
			{
				lowerReturn(current, abi);
				continue;
			}

			// lower variable accesses
			for(auto read = current->reads.begin();
				read != current->reads.end(); ++read)
			{
				if((*read)->isAddress())
				{
					if((*read)->isBasicBlock()) continue;
					
					lowerAddress(*read, globals, locals);
				}
				else if((*read)->isArgument())
				{
					lowerArgument(*read, locals, abi);
				}
			}
		}
	}
}

static const abi::ApplicationBinaryInterface* getABI()
{
	auto archaeopteryxABI =
		abi::ApplicationBinaryInterface::getABI("archaeopteryx");

	return archaeopteryxABI;
}

}

}

