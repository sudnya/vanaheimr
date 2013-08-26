/*! \file   PTXToVIRTranslator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday Fubruary 25, 2012
	\brief  The source file for the PTXToVIRTranslator class.
*/

// Vanaheimr Includes
#include <vanaheimr/translation/interface/PTXToVIRTranslator.h>

#include <vanaheimr/ir/interface/Type.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <configure.h>

// Ocelot Includes
#if HAVE_OCELOT

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>

#include <ocelot/transforms/interface/ReadableLayoutPass.h>
#include <ocelot/transforms/interface/PassManager.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace translation
{

PTXToVIRTranslator::PTXToVIRTranslator(compiler::Compiler* compiler)
: _compiler(compiler)
{

}

void PTXToVIRTranslator::translate(const PTXModule& m)
{
	report("Translating PTX module '"  << m.path() << "'");

	_ptx    = &m;
	_module = &*_compiler->newModule(m.path());
	
	// Translate globals
	for(PTXModule::GlobalMap::const_iterator global = m.globals().begin();
		global != m.globals().end(); ++global)
	{
		_translateGlobal(global->second);
	}
	
	// Translate kernel functions
	for(PTXModule::KernelMap::const_iterator kernel = m.kernels().begin();
		kernel != m.kernels().end(); ++kernel)
	{
		_translateKernel(*kernel->second);
	}
}

void PTXToVIRTranslator::_translateParameter(const PTXParameter& parameter)
{
	if(parameter.returnArgument)
	{
		_function->newReturnValue(_getType(parameter.type),
			parameter.name);
	}
	else
	{
		_function->newArgument(_getType(parameter.type),
			parameter.name);
	}
}

void PTXToVIRTranslator::_translateGlobal(const PTXGlobal& global)
{
	report(" Translating PTX global " << global.statement.toString());
	
	ir::Module::global_iterator virGlobal = _module->newGlobal(
		global.statement.name, _getType(global.statement.type),
		_translateLinkage(global.statement.attribute),
		(ir::Global::Level)_translateAddressSpace(global.space()));
		
	if(global.statement.initializedBytes() != 0)
	{
		virGlobal->setInitializer(_translateInitializer(global));
	}
}

static ::ir::ControlFlowGraph::BlockPointerVector
	getBlockSequence(const ::ir::IRKernel& constPTX)
{
	typedef ::ir::IRKernel IRKernel;
	typedef ::ir::Module   PTXModule;
	
	::transforms::ReadableLayoutPass pass;
	
	IRKernel& ptx = const_cast<IRKernel&>(constPTX);
	
	transforms::PassManager manager(const_cast<PTXModule*>(ptx.module));
	manager.addPass(&pass);
	
	manager.runOnKernel(ptx);
	manager.releasePasses();
	
	return pass.blocks;
}

void PTXToVIRTranslator::_translateKernel(const PTXKernel& kernel)
{
	report(" Translating PTX kernel '" << kernel.getPrototype().toString());

	ir::Module::iterator function = _module->getFunction(kernel.name);
	
	if(function != _module->end())
	{
		assert(function->isPrototype());
		
		function->removeAttribute("prototype");
	}
	else
	{
		function = _module->newFunction(kernel.name,
			_translateLinkingDirective(kernel.getPrototype().linkingDirective),
			_translateVisibility(kernel.getPrototype().linkingDirective));
	}
	
	_function = &*function;

	if(!kernel.function())
	{
		_function->addAttribute("kernel");
	}

	// Translate Arguments
	for(PTXKernel::ParameterVector::const_iterator
		argument = kernel.arguments.begin();
		argument != kernel.arguments.end(); ++argument)
	{
		_translateParameter(*argument);
	}
	
	_function->interpretType();

	// Translate Values
	PTXKernel::RegisterVector registers = kernel.getReferencedRegisters();
	
	for(PTXKernel::RegisterVector::iterator reg = registers.begin();
		reg != registers.end(); ++reg)
	{
		_translateRegisterValue(reg->id, reg->type);
	}
	
	// Translate locals
	for(auto local = kernel.locals.begin();
		local != kernel.locals.end(); ++local)
	{
		_translateLocal(local->second);
	}
	
	::ir::ControlFlowGraph::BlockPointerVector sequence =
		getBlockSequence(kernel);

	// Keep a record of blocks
	for(::ir::ControlFlowGraph::BlockPointerVector::iterator
		block = sequence.begin(); block != sequence.end(); ++block)
	{
		if(*block == kernel.cfg()->get_entry_block()) continue;
		if(*block == kernel.cfg()->get_exit_block())  continue;
		_recordBasicBlock(**block);
	}
	
	// Translate blocks
	for(::ir::ControlFlowGraph::BlockPointerVector::iterator
		block = sequence.begin(); block != sequence.end(); ++block)
	{
		if(*block == kernel.cfg()->get_entry_block()) continue;
		if(*block == kernel.cfg()->get_exit_block())  continue;

		_translateBasicBlock(**block);
	}
	
	_registers.clear();
	_blocks.clear();
}

void PTXToVIRTranslator::_translateLocal(const PTXLocal& local)
{
	report("  Translating PTX local " << local.toString());
	
	_function->newLocalValue(local.name, _getType(local.type),
		_translateLinkage(local.attribute),
		(ir::Global::Level)_translateAddressSpace(local.space));

	// No initializer for now (ever?)
}

void PTXToVIRTranslator::_translateRegisterValue(PTXRegisterId reg,
	PTXDataType type)
{
	report("  Translating PTX register "
		<< PTXOperand::toString((PTXOperand::DataType) type)
		<< " r" << reg);

	std::stringstream name;
	
	name << "r"  << reg;
	
	if(_registers.count(reg) != 0)
	{
		throw std::runtime_error("Added duplicate virtual register '"
			+ name.str() + "'");
	}
	
	ir::Function::register_iterator newRegister = _function->newVirtualRegister(
		_getType(type), name.str());

	report("    to " << newRegister->type->name << " r" << newRegister->id);

	_registers.insert(std::make_pair(reg, newRegister));
}

void PTXToVIRTranslator::_recordBasicBlock(const PTXBasicBlock& basicBlock)
{
	report("  Record PTX basic block " << basicBlock.label());
	
	ir::Function::iterator block = _function->newBasicBlock(
		_function->exit_block(), basicBlock.label());	

	if(_blocks.count(basicBlock.label()) != 0)
	{
		throw std::runtime_error("Added duplicate basic block '"
			+ basicBlock.label() + "'");
	}
	
	_blocks.insert(std::make_pair(basicBlock.label(), block));
}

void PTXToVIRTranslator::_translateBasicBlock(const PTXBasicBlock& basicBlock)
{
	report("  Translating PTX basic block " << basicBlock.label());
	
	BasicBlockMap::iterator block = _blocks.find(basicBlock.label());
	
	if(block == _blocks.end())
	{
		throw std::runtime_error("Basic block " + basicBlock.label()
			+ " was not declared in this function.");
	}

	_block = &*block->second;
	
	for(PTXBasicBlock::const_instruction_iterator
		instruction = basicBlock.instructions.begin();
		instruction != basicBlock.instructions.end(); ++instruction)
	{
		const PTXInstruction& ptx = static_cast<const PTXInstruction&>(
			**instruction);
	
		_translateInstruction(ptx);
	}
}

void PTXToVIRTranslator::_translateInstruction(const PTXInstruction& ptx)
{
	report("   Translating PTX instruction " << ptx.toString());

	_ptxInstruction = &ptx;

	// Translate complex instructions
	if(_translateComplexInstruction(ptx)) return;
	
	// Translate simple instructions
	if(_translateSimpleBinaryInstruction(ptx)) return;
	if( _translateSimpleUnaryInstruction(ptx)) return;
	
	assertM(false, "No translation implemented for instruction "
		<< ptx.toString());
}

bool PTXToVIRTranslator::_translateComplexInstruction(const PTXInstruction& ptx)
{
	switch(ptx.opcode)
	{
		case PTXInstruction::St:
		{
			_translateSt(ptx);
			return true; 
		}
		case PTXInstruction::SetP:
		{
			_translateSetp(ptx);
			return true;
		}
		case PTXInstruction::Bra:
		{
			_translateBra(ptx);
			return true;
		}
		case PTXInstruction::Ret:  // fall through
		case PTXInstruction::Exit:
		{
			_translateExit(ptx);
			return true;
		}
		case PTXInstruction::Neg:
		{
			_translateNeg(ptx);
			return true;
		}
		case PTXInstruction::Not:
		{
			_translateNot(ptx);
			return true;
		}
		case PTXInstruction::Ld:
		{
			if(ptx.d.isVector())
			{
				_translateSimpleIntrinsic(ptx);
				return true;
			}
			
			return false;
		}
		case PTXInstruction::Bar:        // fall through
		case PTXInstruction::SelP:       // fall through
		case PTXInstruction::Fma:        // fall through
		case PTXInstruction::Atom:       // fall through
		case PTXInstruction::Reconverge: // fall through
		case PTXInstruction::Membar:     // fall through
		case PTXInstruction::Popc:       // fall through
		case PTXInstruction::Min:        // fall through
		case PTXInstruction::Max:        // fall through
		case PTXInstruction::Mad:
		{
			_translateSimpleIntrinsic(ptx);
			return true;
		}
		case PTXInstruction::Call:
		{
			_translateCall(ptx);
			return true;
		}
		default: break;	
	}

	return false;
}

static ir::UnaryInstruction* newUnaryInstruction(
	const ::ir::PTXInstruction& ptx)
{
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;
	
	switch(ptx.opcode)
	{
	case PTXInstruction::Ld: // fall through
	case PTXInstruction::Ldu:
	{
		return new ir::Ld;
	}
	case PTXInstruction::Cvta:
	case PTXInstruction::Mov:
	{
		return new ir::Bitcast;
	}
	case PTXInstruction::Cvt:
	{
		if(PTXOperand::isFloat(ptx.d.type))
		{
			if(PTXOperand::isFloat(ptx.a.type))
			{
				if(ptx.d.type == PTXOperand::f32)
				{
					if(ptx.a.type == PTXOperand::f32)
					{
						return new ir::Bitcast;
					}
					else
					{
						return new ir::Fptrunc;
					}
				}
				else
				{
					if(ptx.a.type == PTXOperand::f32)
					{
						return new ir::Fpext;
					}
					else
					{
						return new ir::Bitcast;
					}
				}
			}
			else if(PTXOperand::isSigned(ptx.a.type))
			{
				return new ir::Sitofp;
			}
			else
			{
				return new ir::Uitofp;
			}
		}
		else if(PTXOperand::isSigned(ptx.d.type))
		{
			if(PTXOperand::isFloat(ptx.a.type))
			{
				return new ir::Fptosi;
			}
			else
			{
				if(PTXOperand::bytes(ptx.a.type) >
					PTXOperand::bytes(ptx.d.type))
				{
					return new ir::Trunc;
				}
				else if(PTXOperand::bytes(ptx.d.type) ==
					PTXOperand::bytes(ptx.a.type))
				{
					return new ir::Bitcast;
				}
				else if(PTXOperand::isSigned(ptx.a.type))
				{
					return new ir::Sext;
				}
				else
				{
					return new ir::Zext;
				}
			}
		}
		else
		{
			if(PTXOperand::isFloat(ptx.a.type))
			{
				return new ir::Fptoui;
			}
			else
			{
				if(PTXOperand::bytes(ptx.a.type) >
					PTXOperand::bytes(ptx.d.type))
				{
					return new ir::Trunc;
				}
				else if(PTXOperand::bytes(ptx.d.type) ==
					PTXOperand::bytes(ptx.a.type))
				{
					return new ir::Bitcast;
				}
				else
				{
					return new ir::Zext;
				}
			}
		}
		break;
	}
	default:
	{
		break;
	}
	}
	
	assertM(false, "Invalid simple unary translation");	
	return 0;
}

static bool isSimpleUnaryInstruction(const ::ir::PTXInstruction& ptx)
{
	typedef ::ir::PTXInstruction PTXInstruction;

	switch(ptx.opcode)
	{
	case PTXInstruction::Ldu:  // fall through
	case PTXInstruction::Ld:   // fall through
	case PTXInstruction::Mov:  // fall through
	case PTXInstruction::Cvta:
	{
		return true;
		break;
	}
	case PTXInstruction::Cvt:
	{
		if(ptx.modifier == PTXInstruction::Modifier_invalid)
		{
			return true;
		}
		break;
	}
	default:
	{
		break;
	}
	}
	
	return false;
}

bool PTXToVIRTranslator::_translateSimpleUnaryInstruction(
	const PTXInstruction& ptx)
{
	if(!isSimpleUnaryInstruction(ptx)) return false;

	ir::UnaryInstruction* vir = newUnaryInstruction(ptx);
	_instruction = vir;
	
	vir->setGuard(_translatePredicateOperand(ptx.pg));
	vir->setD(_newTranslatedOperand(ptx.d));
	vir->setA(_newTranslatedOperand(ptx.a));

	report("    to " << vir->toString());
	
	_block->push_back(vir);
	
	return true;
}

static bool isSimpleBinaryInstruction(const ::ir::PTXInstruction& ptx)
{
	typedef ::ir::PTXInstruction PTXInstruction;

	switch(ptx.opcode)
	{
	case PTXInstruction::Add: // fall through
	case PTXInstruction::And: // fall through
	case PTXInstruction::Div: // fall through
	case PTXInstruction::Mul: // fall through
	case PTXInstruction::Or:  // fall through
	case PTXInstruction::Rem: // fall through
	case PTXInstruction::Shl: // fall through
	case PTXInstruction::Shr: // fall through
	case PTXInstruction::Sub: // fall through
	case PTXInstruction::Xor:
	{
		return true;
	}
	default:
	{
		break;
	}
	}
	
	return false;
}

static ir::BinaryInstruction* newBinaryInstruction(
	const ::ir::PTXInstruction& ptx)
{
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;
	
	switch(ptx.opcode)
	{
	case PTXInstruction::Add:
	{
		return new ir::Add;
	}
	case PTXInstruction::And:
	{
		return new ir::And;		
	}
	case PTXInstruction::Div:
	{
		if(PTXOperand::isFloat(ptx.type))
		{
			return new ir::Fdiv;
		}
		else
		{
			if(PTXOperand::isSigned(ptx.type))
			{
				return new ir::Sdiv;
			}
			else
			{
				return new ir::Udiv;
			}
		}
	}
	case PTXInstruction::Mul:
	{
		if(PTXOperand::isFloat(ptx.type))
		{
			return new ir::Fmul;
		}
		else
		{
			return new ir::Mul;
		}
	}
	case PTXInstruction::Or:
	{
		return new ir::Or;		
	}
	case PTXInstruction::Rem:
	{
		if(PTXOperand::isFloat(ptx.type))
		{
			return new ir::Frem;
		}
		else
		{
			if(PTXOperand::isSigned(ptx.type))
			{
				return new ir::Srem;
			}
			else
			{
				return new ir::Urem;
			}
		}
	}
	case PTXInstruction::Shl:
	{
		return new ir::Shl;		
	}
	case PTXInstruction::Shr:
	{
		if(PTXOperand::isSigned(ptx.type))
		{
			return new ir::Ashr;
		}
		else
		{
			return new ir::Lshr;
		}
	}
	case PTXInstruction::Sub:
	{
		return new ir::Sub;
	}
	case PTXInstruction::Xor:
	{
		return new ir::Xor;		
	}
	default:
	{
		break;
	}
	}

	assertM(false, "Invalid simple binary instruction.");	
	return 0;
}

bool PTXToVIRTranslator::_translateSimpleBinaryInstruction(
	const PTXInstruction& ptx)
{
	if(!isSimpleBinaryInstruction(ptx)) return false;
	
	ir::BinaryInstruction* vir = newBinaryInstruction(ptx);
	
	vir->setGuard(_translatePredicateOperand(ptx.pg));
	vir->setD(_newTranslatedOperand(ptx.d));
	vir->setA(_newTranslatedOperand(ptx.a));
	vir->setB(_newTranslatedOperand(ptx.b));
	
	report("    to " << vir->toString());

	_block->push_back(vir);
	
	return true;
}

void PTXToVIRTranslator::_translateSt(const PTXInstruction& ptx)
{
	if(ptx.a.isVector())
	{
		_translateSimpleIntrinsic(ptx);
		return;
	}
	
	ir::St* st = new ir::St(_block);
	
	st->setGuard(_translatePredicateOperand(ptx.pg));
	st->setD(_newTranslatedOperand(ptx.d));
	st->setA(_newTranslatedOperand(ptx.a));

	report("    to " << st->toString());

	_block->push_back(st);
}

static ir::ComparisonInstruction::Comparison translateComparison(
	::ir::PTXOperand::DataType d, ::ir::PTXInstruction::CmpOp cmp)
{
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;
	
	switch(cmp)
	{
	case PTXInstruction::Eq:  return ir::ComparisonInstruction::OrderedEqual;
	case PTXInstruction::Ne:  return ir::ComparisonInstruction::OrderedNotEqual;
	case PTXInstruction::Lo:  /* fall through */
	case PTXInstruction::Lt:  return ir::ComparisonInstruction::OrderedLessThan;
	case PTXInstruction::Ls:  /* fall through */
	case PTXInstruction::Le:  return ir::ComparisonInstruction::OrderedLessOrEqual;
	case PTXInstruction::Hi:  /* fall through */
	case PTXInstruction::Gt:  return ir::ComparisonInstruction::OrderedGreaterThan;
	case PTXInstruction::Hs:  /* fall through */ 
	case PTXInstruction::Ge:  return ir::ComparisonInstruction::OrderedGreaterOrEqual;
	case PTXInstruction::Equ: return ir::ComparisonInstruction::UnorderedEqual;
	case PTXInstruction::Neu: return ir::ComparisonInstruction::UnorderedNotEqual;
	case PTXInstruction::Ltu: return ir::ComparisonInstruction::UnorderedLessThan;
	case PTXInstruction::Leu: return ir::ComparisonInstruction::UnorderedLessOrEqual;
	case PTXInstruction::Gtu: return ir::ComparisonInstruction::UnorderedGreaterThan;
	case PTXInstruction::Geu: return ir::ComparisonInstruction::UnorderedGreaterOrEqual;
	case PTXInstruction::Num: return ir::ComparisonInstruction::IsANumber;
	case PTXInstruction::Nan: return ir::ComparisonInstruction::NotANumber;
	case PTXInstruction::CmpOp_Invalid: break;
	}

	return ir::ComparisonInstruction::InvalidComparison;
}

void PTXToVIRTranslator::_translateSetp(const PTXInstruction& ptx)
{	
	ir::Setp* setp = new ir::Setp(translateComparison(
		ptx.type, ptx.comparisonOperator), _block);
	
	if(ptx.c.addressMode == PTXOperand::Invalid)
	{
		setp->setGuard(_translatePredicateOperand(ptx.pg));
		setp->setD(_newTranslatedOperand(ptx.d));
		setp->setA(_newTranslatedOperand(ptx.a));
		setp->setB(_newTranslatedOperand(ptx.b));
	}
	else
	{
		assertM(false, "not implemented.");
	}

	report("    to " << setp->toString());
	
	_block->push_back(setp);
}

void PTXToVIRTranslator::_translateBra(const PTXInstruction& ptx)
{
	ir::Bra::BranchModifier modifier =
		ptx.uni ? ir::Bra::UniformBranch : ir::Bra::MultitargetBranch;

	ir::Bra* bra = new ir::Bra(modifier, _block);
	
	bra->setGuard(_translatePredicateOperand(ptx.pg));
	bra->setTarget(_newTranslatedOperand(ptx.d));

	report("    to " << bra->toString());
	
	_block->push_back(bra);
}

void PTXToVIRTranslator::_translateExit(const PTXInstruction& ptx)
{
	ir::Ret* exit = new ir::Ret(_block);
	
	exit->setGuard(_translatePredicateOperand(ptx.pg));

	report("    to " << exit->toString());
	
	_block->push_back(exit);
}

void PTXToVIRTranslator::_translateNeg(const PTXInstruction& ptx)
{
	ir::Sub* sub = new ir::Sub(_block);
	
	sub->setGuard(_translatePredicateOperand(ptx.pg));
	
	sub->setD(_newTranslatedOperand(ptx.d));
	sub->setA(new ir::ImmediateOperand((uint64_t)0, sub, _getType(ptx.type)));
	sub->setB(_newTranslatedOperand(ptx.a));

	report("    to " << sub->toString());
	
	_block->push_back(sub);
}

void PTXToVIRTranslator::_translateNot(const PTXInstruction& ptx)
{
	ir::Xor* xxor = new ir::Xor(_block);
	
	xxor->setGuard(_translatePredicateOperand(ptx.pg));
	
	xxor->setD(_newTranslatedOperand(ptx.d));
	xxor->setA(new ir::ImmediateOperand((uint64_t)0xffffffffffffffffULL,
		xxor, _getType(ptx.type)));
	xxor->setB(_newTranslatedOperand(ptx.a));

	report("    to " << xxor->toString());
	
	_block->push_back(xxor);
}

static std::string translateTypeName(::ir::PTXOperand::DataType type)
{
	switch(type)
	{
	case ::ir::PTXOperand::b8:  /* fall through */
	case ::ir::PTXOperand::s8:  /* fall through */
	case ::ir::PTXOperand::u8:
	{
		return "i8";
	}
	case ::ir::PTXOperand::s16: /* fall through */
	case ::ir::PTXOperand::u16: /* fall through */
	case ::ir::PTXOperand::b16:
	{
		return "i16";
	}
	case ::ir::PTXOperand::s32: /* fall through */
	case ::ir::PTXOperand::b32: /* fall through */
	case ::ir::PTXOperand::u32:
	{
		return "i32";
	}
	case ::ir::PTXOperand::s64: /* fall through */
	case ::ir::PTXOperand::b64: /* fall through */
	case ::ir::PTXOperand::u64:
	{
		return "i64";
	}
	case ::ir::PTXOperand::f32:
	{
		return "f32";
	}
	case ::ir::PTXOperand::f64:
	{
		return "f64";
	}
	case ::ir::PTXOperand::pred:
	{
		return "i1";
	}
	default: break;
	}
	
	return "";
}

static std::string modifierString(const ::ir::PTXInstruction& ptx)
{
	typedef ::ir::PTXInstruction PTXInstruction;
	
	std::string result;
	
	if(ptx.modifier & PTXInstruction::approx)
	{
		result += "_approx";
	}
	else if(ptx.modifier & PTXInstruction::wide)
	{
		result += "_wide";
	}
	else if(ptx.modifier & PTXInstruction::hi)
	{
		result += "_hi";
	}
	else if(ptx.modifier & PTXInstruction::lo)
	{
		result += "_lo";
	}
	else if(ptx.modifier & PTXInstruction::rn)
	{
		result += "_rn";
	}
	else if(ptx.modifier & PTXInstruction::rz)
	{
		result += "_rz";
	}
	else if(ptx.modifier & PTXInstruction::rm)
	{
		result += "_rm";
	}
	else if(ptx.modifier & PTXInstruction::rp)
	{
		result += "_rp";
	}
	else if(ptx.modifier & PTXInstruction::rni)
	{
		result += "_rni";
	}
	else if(ptx.modifier & PTXInstruction::rzi)
	{
		result += "_rzi";
	}
	else if(ptx.modifier & PTXInstruction::rmi)
	{
		result += "_rmi";
	}
	else if(ptx.modifier & PTXInstruction::rpi)
	{
		result += "_rpi";
	}

	if(ptx.modifier & PTXInstruction::ftz)
	{
		result += "_ftz";
	}
	if(ptx.modifier & PTXInstruction::sat)
	{
		result += "_sat";
	}
	if(ptx.carry == PTXInstruction::CC)
	{
		result += "_cc";
	}

	if(ptx.opcode == PTXInstruction::Bar)
	{
		if(ptx.barrierOperation == PTXInstruction::BarSync)
		{
			result += "_sync";
		}
		else if(ptx.barrierOperation == PTXInstruction::BarArrive)
		{
			result += "_arrive";
		}
		else if(ptx.barrierOperation == PTXInstruction::BarReduction)
		{
			result += "_reduce";
		}
	}

	if(ptx.opcode == PTXInstruction::Membar)
	{
		if(ptx.level == PTXInstruction::CtaLevel)
		{
			result += "_cta";
		}
		else if(ptx.level == PTXInstruction::GlobalLevel)
		{
			result += "_global";
		}
	}
	
	if(ptx.opcode == PTXInstruction::Atom)
	{
		if(ptx.atomicOperation == PTXInstruction::AtomicAnd)
		{
			result += "_and";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicOr)
		{
			result += "_or";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicXor)
		{
			result += "_xor";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicCas)
		{
			result += "_cas";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicExch)
		{
			result += "_exch";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicAdd)
		{
			result += "_add";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicInc)
		{
			result += "_inc";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicDec)
		{
			result += "_dec";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicMin)
		{
			result += "_min";
		}
		else if(ptx.atomicOperation == PTXInstruction::AtomicMax)
		{
			result += "_max";
		}
	}

	return result;
}

static bool hasDestination(const ::ir::PTXInstruction& ptx)
{
	return ptx.opcode != ::ir::PTXInstruction::Bar &&
		ptx.opcode != ::ir::PTXInstruction::Membar &&
		ptx.opcode != ::ir::PTXInstruction::Reconverge;
}

static bool destinationIsSource(const ::ir::PTXInstruction& ptx)
{
	return ptx.opcode == ::ir::PTXInstruction::Bar;
}

void PTXToVIRTranslator::_translateSimpleIntrinsic(const PTXInstruction& ptx)
{	
	ir::Call* call = new ir::Call(_block);
	
	call->setGuard(static_cast<ir::PredicateOperand*>(
		_translatePredicateOperand(ptx.pg)));
	
	if(hasDestination(ptx))
	{
		if(ptx.d.isVector())
		{
			for(auto operand = ptx.d.array.begin();
				operand != ptx.d.array.end(); ++operand)
			{
				assert(operand->isRegister());

				call->addReturn(_newTranslatedOperand(*operand));
			}
		}
		else
		{
			call->addReturn(_newTranslatedOperand(ptx.d));
		}
	}
	else if(destinationIsSource(ptx))
	{
		call->addArgument(_newTranslatedOperand(ptx.d));
	}
	
	auto operands = {&ptx.a, &ptx.b, &ptx.c};

	for(auto operand : operands)
	{
		if(operand->addressMode == PTXOperand::Invalid) continue;
		
		if(operand->isVector())
		{
			for(auto argument = operand->array.begin();
				argument != operand->array.end(); ++argument)
			{
				assert(argument->isRegister());

				call->addArgument(_newTranslatedOperand(*argument));
			}
		}
		else
		{
			call->addArgument(_newTranslatedOperand(*operand));
		}
	}
	
	std::stringstream stream;
	
	stream << "_Zintrinsic_" << PTXInstruction::toString(ptx.opcode);
	
	std::string modifiers = modifierString(ptx);
	
	if(!modifiers.empty())
	{
		stream << modifiers;
	}
	
	if(hasDestination(ptx))
	{
		stream << "_" << translateTypeName(ptx.type);
	}
	
	_addPrototype(stream.str(), *call);
	
	call->setTarget(new ir::AddressOperand(_getGlobal(stream.str()),
		_instruction));

	_block->push_back(call);
	
	report("    to " << call->toString());
}

void PTXToVIRTranslator::_translateCall(const PTXInstruction& ptx)
{	
	ir::Call* call = new ir::Call(_block);
	
	call->setGuard(static_cast<ir::PredicateOperand*>(
		_translatePredicateOperand(ptx.pg)));
	
	if(ptx.a.addressMode == PTXOperand::FunctionName)
	{
		// direct call
		_addPrototype(ptx.a.identifier, *call);
		
		auto function = _getGlobal(ptx.a.identifier);
			
		assert(function != nullptr);
		
		call->setTarget(new ir::AddressOperand(function, _instruction));
	}
	else
	{
		// indirect call
		call->setTarget(_newTranslatedOperand(ptx.a));
	}

	for(auto operand = ptx.d.array.begin();
		operand != ptx.d.array.end(); ++operand)
	{
		assert(operand->isRegister());

		call->addReturn(_newTranslatedOperand(*operand));
	}

	for(auto operand = ptx.b.array.begin();
		operand != ptx.b.array.end(); ++operand)
	{
		assert(operand->isRegister());

		call->addArgument(_newTranslatedOperand(*operand));
	}
	
	_block->push_back(call);
	
	report("    to " << call->toString());
}

ir::Operand* PTXToVIRTranslator::_newTranslatedOperand(const PTXOperand& ptx)
{
	switch(ptx.addressMode)
	{
	case PTXOperand::Register:
	{
		return new ir::RegisterOperand(_getRegister(ptx.reg), _instruction);
	}
	case PTXOperand::Indirect:
	{
		return new ir::IndirectOperand(_getRegister(ptx.reg),
			ptx.offset, _instruction);
	}
	case PTXOperand::Immediate:
	{
		return new ir::ImmediateOperand((uint64_t)ptx.imm_uint, _instruction,
			_getType(ptx.type));
	}
	case PTXOperand::Address:
	{
		if(_ptxInstruction->addressSpace == PTXInstruction::Param &&
			_isArgument(ptx.identifier))
		{
			return new ir::ArgumentOperand(_getArgument(ptx.identifier),
				_instruction);
		}
		else
		{
			return new ir::AddressOperand(
				_getGlobal(ptx.identifier), _instruction);
		}
	}
	case PTXOperand::Label:
	{
		return new ir::AddressOperand(_getBasicBlock(ptx.identifier),
			_instruction);
	}
	case PTXOperand::FunctionName:
	{
		return new ir::AddressOperand(_getGlobal(ptx.identifier), _instruction);
	}
	case PTXOperand::Special:
	{
		return _getSpecialValueOperand(ptx.special, ptx.vIndex);
	}
	case PTXOperand::BitBucket:
	{
		return new ir::RegisterOperand(_newTemporaryRegister("i64"),
			_instruction);
	}
	default: break;
	}
	
	throw std::runtime_error("No translation implemented for "
		+ ptx.toString());
}

static ir::PredicateOperand::PredicateModifier translatePredicateCondition(
	::ir::PTXOperand::PredicateCondition c)
{
	switch(c)
	{
	case ::ir::PTXOperand::PT:
	{
		return ir::PredicateOperand::PredicateTrue;
	}
	case ::ir::PTXOperand::nPT:
	{
		return ir::PredicateOperand::PredicateFalse;
	}
	case ::ir::PTXOperand::Pred:
	{
		return ir::PredicateOperand::StraightPredicate;
	}
	case ::ir::PTXOperand::InvPred:
	{
		return ir::PredicateOperand::InversePredicate;
	}
	}

	return ir::PredicateOperand::StraightPredicate;
}

ir::PredicateOperand* PTXToVIRTranslator::_translatePredicateOperand(
	const PTXOperand& ptx)
{
	ir::VirtualRegister* predicateRegister = 0;

	if(ptx.condition != PTXOperand::PT && ptx.condition != PTXOperand::nPT)
	{
		predicateRegister = _getRegister(ptx.reg);
	}
	
	return new ir::PredicateOperand(predicateRegister,
		translatePredicateCondition(ptx.condition), _instruction);
}

ir::PredicateOperand* PTXToVIRTranslator::_translatePredicateOperand(
	unsigned int condition)
{
	return new ir::PredicateOperand(nullptr,
		translatePredicateCondition((PTXOperand::PredicateCondition)condition),
		_instruction);
}


ir::VirtualRegister* PTXToVIRTranslator::_getSpecialVirtualRegister(
	unsigned int id, unsigned int vectorIndex)
{
	std::stringstream stream;
	
	stream << "_Zintrinsic_getspecial_";
	
	bool isScalar = true;
	
	switch (id) 
	{
	case PTXOperand::tid:     // fall through
	case PTXOperand::ntid:    // fall through
	case PTXOperand::ctaId:   // fall through
	case PTXOperand::nctaId:  // fall through
	case PTXOperand::smId:    // fall through
	case PTXOperand::nsmId:   // fall through
	case PTXOperand::gridId:  // fall through
		isScalar = false;
		break;
	default:
		isScalar = true;
	}
	
	std::string specialName =
		PTXOperand::toString((PTXOperand::SpecialRegister)id).substr(1);
	
	if(vectorIndex != PTXOperand::v1 || isScalar) 
	{
		stream << specialName;
	}
	else
	{
		stream << specialName + "_" +
			PTXOperand::toString((PTXOperand::VectorIndex)vectorIndex);
	}
	
	_addSpecialPrototype(stream.str());

	ir::Call* call = new ir::Call(_block);
	
	call->setGuard(_translatePredicateOperand(PTXOperand::PT));

	ir::RegisterOperand* specialValue = new ir::RegisterOperand(
		_newTemporaryRegister("i32"), call);

	call->addReturn(specialValue);
	
	call->setTarget(new ir::AddressOperand(
		_getGlobal(stream.str()), _instruction));

	_block->push_back(call);
	
	report("    to " << call->toString());
	
	return specialValue->virtualRegister;
}

ir::VirtualRegister* PTXToVIRTranslator::_getRegister(PTXRegisterId id)
{
	RegisterMap::iterator reg = _registers.find(id);
	
	if(reg == _registers.end())
	{
		std::stringstream name;
		
		name << "r" << id;

		throw std::runtime_error("PTX register " + name.str()
			+ " used without declaration.");
	}
	
	return &*reg->second;
}

ir::Variable* PTXToVIRTranslator::_getGlobal(const std::string& name)
{
	ir::Function::local_iterator local = _function->findLocalValue(name);

	if(local != _function->local_end()) return &*local;

	ir::Module::iterator function = _module->getFunction(name);

	if(function != _module->end()) return &*function;

	ir::Module::global_iterator global = _module->getGlobal(name);
	
	if(global == _module->global_end())
	{
		throw std::runtime_error("Global variable " + name
			+ " used without declaration.");
	}
	
	return &*global;
}

ir::Variable* PTXToVIRTranslator::_getBasicBlock(const std::string& name)
{
	BasicBlockMap::iterator block = _blocks.find(name);
	
	if(block == _blocks.end())
	{
		throw std::runtime_error("Basic block " + name
			+ " was not declared in this function.");
	}

	return &*block->second;
}

ir::Argument* PTXToVIRTranslator::_getArgument(const std::string& name)
{
	for(ir::Function::argument_iterator argument = _function->argument_begin();
		argument != _function->argument_end(); ++argument)
	{
		if(argument->name() == name) return &*argument;
	}
	
	for(ir::Function::argument_iterator argument = _function->returned_begin();
		argument != _function->returned_end(); ++argument)
	{
		if(argument->name() == name) return &*argument;
	}
	
	throw std::runtime_error("Argument " + name
		+ " was not declared in this function.");
		
	return nullptr;
}

ir::Operand* PTXToVIRTranslator::_getSpecialValueOperand(
	unsigned int id, unsigned int vIndex)
{
	return new ir::RegisterOperand(
		_getSpecialVirtualRegister(id, vIndex), _instruction);
}

ir::VirtualRegister* PTXToVIRTranslator::_newTemporaryRegister(
	const std::string& typeName)
{
	ir::Function::register_iterator temp = _function->newVirtualRegister(
		_getType(typeName));
		
	return &*temp;
}

const ir::Type* PTXToVIRTranslator::_getType(PTXDataType ptxType)
{
	return _getType(translateTypeName((::ir::PTXOperand::DataType)ptxType));
}

const ir::Type* PTXToVIRTranslator::_getType(const std::string& typeName)
{
	const ir::Type* type = _compiler->getType(typeName);

	if(type == 0)
	{
		throw std::runtime_error("PTX translated type name '"
			+ typeName + "' is not a valid Vanaheimr type.");
	}
	
	return type;
}

ir::Variable::Linkage PTXToVIRTranslator::_translateLinkage(PTXAttribute attr)
{
	if(attr == ::ir::PTXStatement::Extern)
	{
		return ir::Variable::ExternalLinkage;
	}
	else
	{
		return ir::Variable::PrivateLinkage;
	}
}

ir::Variable::Visibility PTXToVIRTranslator::_translateVisibility(
	PTXAttribute attr)
{
	if(attr == ::ir::PTXStatement::Visible)
	{
		return ir::Variable::VisibleVisibility;
	}
	else
	{
		return ir::Variable::HiddenVisibility;
	}
}

ir::Variable::Linkage PTXToVIRTranslator::_translateLinkingDirective(
	PTXLinkingDirective d)
{
	if(d == PTXKernel::Prototype::Extern)
	{
		return ir::Variable::ExternalLinkage;
	}
	else
	{
		return ir::Variable::PrivateLinkage;
	}
}

unsigned int PTXToVIRTranslator::_translateAddressSpace(unsigned int space)
{
	typedef ::ir::PTXInstruction PTXInstruction;
	
	unsigned int level = ir::Global::InvalidLevel;

	switch(space)
	{
	case PTXInstruction::Const:   // fall through
	case PTXInstruction::Global:  // fall through
	case PTXInstruction::Texture: // fall through
	case PTXInstruction::Generic:
	{
		level = ir::Global::Shared;
		break;
	}
	case PTXInstruction::Local:
	{
		level = ir::Global::Thread;
		break;
	}
	case PTXInstruction::Param:
	{
		// Parameters are invalid for globals
		break;
	}
	case PTXInstruction::Shared:
	{
		level = ir::Global::CTA;
		break;
	}
	default: break;
	}
	
	return level;
}

ir::Constant* PTXToVIRTranslator::_translateInitializer(const PTXGlobal& g)
{
	if(g.statement.elements() == 1)
	{
		assert(g.statement.elements() == 1);
	
		if(PTXOperand::isFloat(g.statement.type))
		{
			if(g.statement.type == PTXOperand::f32)
			{
				float value = 0.0f;
				
				g.statement.copy(&value);
			
				return new ir::FloatingPointConstant(value);
			}
			else
			{
				double value = 0.0;
				
				g.statement.copy(&value);
				
				return new ir::FloatingPointConstant(value);
			}
		}
		else
		{
			uint64_t value = 0;
			
			g.statement.copy(&value);
				
			return new ir::IntegerConstant(value,
				8 * PTXOperand::bytes(g.statement.type));
		}
	}
	
	auto array = new ir::ArrayConstant(g.statement.elements(),
		_getType(g.statement.type));	
	
	g.statement.copy(array->storage());
	
	return array;
}

bool PTXToVIRTranslator::_isArgument(const std::string& name)
{
	for(ir::Function::argument_iterator argument = _function->argument_begin();
		argument != _function->argument_end(); ++argument)
	{
		if(argument->name() == name) return true;
	}
	
	for(ir::Function::argument_iterator argument = _function->returned_begin();
		argument != _function->returned_end(); ++argument)
	{
		if(argument->name() == name) return true;
	}
	
	return false;
}

void PTXToVIRTranslator::_addSpecialPrototype(const std::string& name)
{
	auto prototype = _module->getFunction(name);
	
	if(prototype != _module->end()) return;
	
	auto function = _module->newFunction(name, ir::Variable::ExternalLinkage,
		ir::Variable::HiddenVisibility);

	function->newReturnValue(_getType("i32"), "returnedValue");

	function->addAttribute("intrinsic");
	function->addAttribute("prototype");

	function->interpretType();
}

void PTXToVIRTranslator::_addPrototype(const std::string& name,
	const ir::Call& call)
{
	auto prototype = _module->getFunction(name);
	
	if(prototype != _module->end()) return;
	
	auto function = _module->newFunction(name, ir::Variable::ExternalLinkage,
		ir::Variable::HiddenVisibility);

	function->addAttribute("prototype");
	function->addAttribute("intrinsic");
	
	unsigned int index = 0;

	auto returnedArguments = call.returned();

	for(auto returned : returnedArguments)
	{
		std::stringstream stream;
		
		stream << "returnValue_" << index++;
	
		function->newReturnValue(returned->type(), stream.str());
	}

	index = 0;

	auto arguments = call.arguments();
	
	for(auto argument : arguments)
	{
		std::stringstream stream;
		
		stream << "argument_" << index++;

		function->newArgument(argument->type(), stream.str());
	}
	
	function->interpretType();
}

}

}

#endif

