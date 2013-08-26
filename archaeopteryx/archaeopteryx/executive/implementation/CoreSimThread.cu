/*! \file   CoreSimThread.cpp
	\date   Saturday May 8, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The source file for the Core simulator of the thread class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/CoreSimThread.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/executive/interface/Intrinsics.h>
#include <archaeopteryx/executive/interface/OperandAccess.h>

#include <archaeopteryx/util/interface/debug.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Operand.h>
#include <vanaheimr/asm/interface/Instruction.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace executive
{

// Typedefs 
typedef executive::CoreSimThread::Value   Value;
typedef executive::CoreSimThread::SValue  SValue;
typedef executive::CoreSimThread::FValue  FValue;
typedef executive::CoreSimThread::DValue  DValue;
typedef executive::CoreSimThread::Address Address;

typedef vanaheimr::as::Instruction Instruction;

typedef vanaheimr::as::Operand          Operand;
typedef vanaheimr::as::RegisterOperand  RegisterOperand;
typedef vanaheimr::as::PredicateOperand PredicateOperand;
typedef vanaheimr::as::ImmediateOperand ImmediateOperand;
typedef vanaheimr::as::IndirectOperand  IndirectOperand;
typedef vanaheimr::as::OperandContainer OperandContainer;


__device__ CoreSimThread::CoreSimThread(CoreSimBlock* parentBlock,
	unsigned threadId, unsigned p, bool b)
: pc(0), finished(false), instructionPriority(p), barrierBit(b),
	m_parentBlock(parentBlock), m_tId(threadId)
{

}

__device__ void CoreSimThread::setParentBlock(CoreSimBlock* p)
{
	m_parentBlock = p;
}

__device__ void CoreSimThread::setThreadId(unsigned id)
{
	m_tId = id;
}

template<typename T, typename F>
__device__ static T bitcast(const F& from)
{
	union UnionCast
	{
		T to;
		F from;
	};
	
	UnionCast cast;
	
	cast.to   = 0;
	cast.from = from;
	
	return cast.to;

}

template<typename T>
static __device__ CoreSimThread::FValue getOperandAs(
	const OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	CoreSimThread::FValue value = getOperand(operandContainer,
		parentBlock, threadId);

	return bitcast<T>(value);
}

static __device__ ir::Binary::PC executeAdd(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Add Add;

	Add* add = static_cast<Add*>(instruction);

	Value a = getOperand(add->a, parentBlock, threadId);
	Value b = getOperand(add->b, parentBlock, threadId);

	Value d = a + b;

	setRegister(add->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeAnd(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::And And;

	And* andd = static_cast<And*>(instruction);

	Value a = getOperand(andd->a, parentBlock, threadId);
	Value b = getOperand(andd->b, parentBlock, threadId);

	Value d = a & b;

	setRegister(andd->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeAshr(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Ashr Ashr;

	Ashr* ashr = static_cast<Ashr*>(instruction);

	SValue a = getOperand(ashr->a, parentBlock, threadId);
	Value b = getOperand(ashr->b, parentBlock, threadId);

	SValue d = a >> b;

	setRegister(ashr->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeAtom(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Atom Atom;

	Atom* atom = static_cast<Atom*>(instruction);

	Value a = getOperand(atom->a, parentBlock, threadId);
	Value b = getOperand(atom->b, parentBlock, threadId);

	Value physical = parentBlock->translateVirtualToPhysical(a);

	//TO DO
	device_assert_m(false, "Atomic operations not supported yet.");
	Value d = 0; //atomicAdd(bitcast<long long int*>(physical), b);

	setRegister(atom->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeBar(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	parentBlock->barrier(threadId);

	return pc + 1;
}

static __device__ ir::Binary::PC executeBitcast(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Bitcast Bitcast;

	Bitcast* bitcast = static_cast<Bitcast*>(instruction);

	Value a = getOperand(bitcast->a, parentBlock, threadId);

	setRegister(bitcast->d, parentBlock, threadId, a);

	return pc + 1;
}

static __device__ ir::Binary::PC executeBra(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Bra Bra;

	Bra* bra = static_cast<Bra*>(instruction);

	Value a = getOperand(bra->target, parentBlock, threadId);

	//TO DO
	return a;
}

static __device__ ir::Binary::PC executeCall(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Call Call;

	Call* call = static_cast<Call*>(instruction);

	if(Intrinsics::isIntrinsic(call, parentBlock))
	{
		Intrinsics::execute(call, parentBlock, threadId);

		return pc + 1;
	}

	Value a = getOperand(call->target, parentBlock, threadId);

	setRegister(parentBlock->getLinkRegister(), parentBlock, threadId, pc + 1);

	return a;
}

static __device__ ir::Binary::PC executeFdiv(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fdiv Fdiv;

	Fdiv* div = static_cast<Fdiv*>(instruction);

	FValue a = getOperandAs<FValue>(div->a, parentBlock, threadId);
	FValue b = getOperandAs<FValue>(div->b, parentBlock, threadId);

	FValue d = a / b;

	setRegister(div->d, parentBlock, threadId, d);
	
	return pc + 1;
}

static __device__ ir::Binary::PC executeFmul(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fmul Fmul;

	Fmul* mul = static_cast<Fmul*>(instruction);

	FValue a = getOperandAs<FValue>(mul->a, parentBlock, threadId);
	FValue b = getOperandAs<FValue>(mul->b, parentBlock, threadId);

	FValue d = a * b;

	setRegister(mul->d, parentBlock, threadId, d);
	
	return pc + 1;
}

static __device__ ir::Binary::PC executeFpext(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fpext Fpext;

	Fpext* fpext = static_cast<Fpext*>(instruction);

	Value a = getOperand(fpext->a, parentBlock, threadId);

	float temp = bitcast<float>(a); 
	double d = temp;

	setRegister(fpext->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeFptosi(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fptosi Fptosi;

	Fptosi* fptosi = static_cast<Fptosi*>(instruction);

	Value a = getOperand(fptosi->a, parentBlock, threadId);

	float temp = bitcast<float>(a);
	SValue d   = temp;

	setRegister(fptosi->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeFptoui(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fptoui Fptoui;

	Fptoui* fptoui = static_cast<Fptoui*>(instruction);

	Value a = getOperand(fptoui->a, parentBlock, threadId);

	float temp = bitcast<float>(a);
	Value d	= temp;

	setRegister(fptoui->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeFpTrunc(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Fptrunc Fptrunc;

	Fptrunc* fptrunc = static_cast<Fptrunc*>(instruction);

	Value a = getOperand(fptrunc->a, parentBlock, threadId);

	double temp = bitcast<double>(a);
	float d	 = temp;

	setRegister(fptrunc->d, parentBlock, threadId, d);
	return pc + 1;
}
	
static __device__ ir::Binary::PC executeFrem(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Frem Frem;

	Frem* rem = static_cast<Frem*>(instruction);

	FValue a = getOperandAs<FValue>(rem->a, parentBlock, threadId);
	FValue b = getOperandAs<FValue>(rem->b, parentBlock, threadId);

	// TODO: implement this
	device_assert_m(false, "Floating point mod not implemented.");
	FValue d = 0.0f;//a % b;

	setRegister(rem->d, parentBlock, threadId, d);
	
	return pc + 1;
}

static __device__ ir::Binary::PC executeLaunch(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	device_assert_m(false, "Device-side kernel launch not implemented yet.");

	return pc;
}

static __device__ ir::Binary::PC executeLd(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Ld Ld;
	
	Ld* ld = static_cast<Ld*>(instruction);

	Value a = getOperand(ld->a, parentBlock, threadId);

	Value physical = parentBlock->translateVirtualToPhysical(a);

	device_report(" Thread %d, loading from (%p virtual) (%p physical)\n",
		threadId, a, physical);

	Value d = 0;
	
	switch(ld->d.asIndirect.type)
	{
		case vanaheimr::as::i1:
		case vanaheimr::as::i8:
		{
			d = *bitcast<uint8_t*>(physical);
			break;
		}
		case vanaheimr::as::i16:
		{
			d = *bitcast<uint16_t*>(physical);
			break;
		}
		case vanaheimr::as::f32:
		case vanaheimr::as::i32:
		{
			d = *bitcast<uint32_t*>(physical);
			break;
		}
		case vanaheimr::as::f64:
		case vanaheimr::as::i64:
		{
			d = *bitcast<uint64_t*>(physical);
			break;
		}
		default: break;
	}

	setRegister(ld->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeLshr(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Lshr Lshr;
	
	Lshr* lshr = static_cast<Lshr*>(instruction);

	Value a = getOperand(lshr->a, parentBlock, threadId);
	Value b = getOperand(lshr->b, parentBlock, threadId);

	Value d = a >> b;

	setRegister(lshr->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeMembar(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	//__threadfence_block();
	
	return pc + 1;
}

static __device__ ir::Binary::PC executeMul(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Mul Mul;
	
	Mul* mul = static_cast<Mul*>(instruction);

	Value a = getOperand(mul->a, parentBlock, threadId);
	Value b = getOperand(mul->b, parentBlock, threadId);

	Value d = a * b;

	setRegister(mul->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeOr(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Or Or;
	
	Or* orr = static_cast<Or*>(instruction);

	Value a = getOperand(orr->a, parentBlock, threadId);
	Value b = getOperand(orr->b, parentBlock, threadId);

	Value d = a | b;

	setRegister(orr->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeRet(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	return parentBlock->returned(threadId, pc); 
}

static __device__ ir::Binary::PC executeSetp(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Setp Setp;
	
	Setp* setp = static_cast<Setp*>(instruction);

	Value a = getOperand(setp->a, parentBlock, threadId);
	Value b = getOperand(setp->b, parentBlock, threadId);

	//TO DO
	Value d = a > b ? 1 : 0 ;
	setRegister(setp->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeSext(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Sext Sext;
	
	Sext* sext = static_cast<Sext*>(instruction);

	Value a = getOperand(sext->a, parentBlock, threadId);

	int temp = bitcast<int>(a);
	SValue d = temp;

	setRegister(sext->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeSdiv(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Sdiv Sdiv;
	
	Sdiv* sdiv = static_cast<Sdiv*>(instruction);

	Value a = getOperand(sdiv->a, parentBlock, threadId);
	Value b = getOperand(sdiv->b, parentBlock, threadId);

	//TO DO
	SValue d = (SValue) a / (SValue) b;
	setRegister(sdiv->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeShl(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Shl Shl;
	
	Shl* shl = static_cast<Shl*>(instruction);

	Value a = getOperand(shl->a, parentBlock, threadId);
	Value b = getOperand(shl->b, parentBlock, threadId);

	Value d = a << b;

	setRegister(shl->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeSitofp(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Sitofp Sitofp;
	
	Sitofp* sitofp = static_cast<Sitofp*>(instruction);

	Value a = getOperand(sitofp->a, parentBlock, threadId);

	//TO DO
	float d = (SValue)a;

	setRegister(sitofp->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeSrem(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Srem Srem;
	
	Srem* srem = static_cast<Srem*>(instruction);

	SValue a = getOperand(srem->a, parentBlock, threadId);
	SValue b = getOperand(srem->b, parentBlock, threadId);

	SValue d = a % b;

	setRegister(srem->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeSt(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::St St;
	
	St* st = static_cast<St*>(instruction);

	Value d = getOperand(st->d, parentBlock, threadId);
	Value physical = parentBlock->translateVirtualToPhysical(d);

	Value a = getOperand(st->a, parentBlock, threadId);

	switch(st->a.asIndirect.type)
	{
		case vanaheimr::as::i1:
		case vanaheimr::as::i8:
		{
			*bitcast<uint8_t*>(physical) = a;
			break;
		}
		case vanaheimr::as::i16:
		{
			*bitcast<uint16_t*>(physical) = a;
			break;
		}
		case vanaheimr::as::f32:
		case vanaheimr::as::i32:
		{
			*bitcast<uint32_t*>(physical) = a;
			break;
		}
		case vanaheimr::as::f64:
		case vanaheimr::as::i64:
		{
			*bitcast<uint64_t*>(physical) = a;
			break;
		}
		default: break;
	}

	return pc + 1;
}

static __device__ ir::Binary::PC executeSub(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Sub Sub;
	
	Sub* sub = static_cast<Sub*>(instruction);

	Value a = getOperand(sub->a, parentBlock, threadId);
	Value b = getOperand(sub->b, parentBlock, threadId);

	Value d = a - b;

	setRegister(sub->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeTrunc(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Trunc Trunc;
	
	Trunc* trunc = static_cast<Trunc*>(instruction);

	Value a = getOperand(trunc->a, parentBlock, threadId);

	//TO DO
	Value d = unsigned (a & 0x00000000FFFFFFFFULL); 

	setRegister(trunc->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeUdiv(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Udiv Udiv;
	
	Udiv* udiv = static_cast<Udiv*>(instruction);

	Value a = getOperand(udiv->a, parentBlock, threadId);
	Value b = getOperand(udiv->b, parentBlock, threadId);

	//TO DO
	Value d = a / b;

	setRegister(udiv->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeUitofp(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Uitofp Uitofp;
	
	Uitofp* uitofp = static_cast<Uitofp*>(instruction);

	Value a = getOperand(uitofp->a, parentBlock, threadId);

	//TO DO
	float d = a;

	setRegister(uitofp->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeUrem(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Urem Urem;
	
	Urem* urem = static_cast<Urem*>(instruction);

	Value a = getOperand(urem->a, parentBlock, threadId);
	Value b = getOperand(urem->b, parentBlock, threadId);

	//TO DO
	Value d = a % b;
	setRegister(urem->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeXor(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Xor Xor;
	
	Xor* xorr = static_cast<Xor*>(instruction);

	Value a = getOperand(xorr->a, parentBlock, threadId);
	Value b = getOperand(xorr->b, parentBlock, threadId);

	Value d = a ^ b;

	setRegister(xorr->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executeZext(Instruction* instruction,
	ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
	typedef vanaheimr::as::Zext Zext;
	
	Zext* zext = static_cast<Zext*>(instruction);

	Value a = getOperand(zext->a, parentBlock, threadId);

	//TO DO
	Value d = (unsigned int)a;
	setRegister(zext->d, parentBlock, threadId, d);
	return pc + 1;
}

static __device__ ir::Binary::PC executePhi(
	Instruction* instruction, ir::Binary::PC pc,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	device_assert_m(false, "PHI instructions are not valid in machine code.");

	return pc + 1;
}

static __device__ ir::Binary::PC executePsi(
	Instruction* instruction, ir::Binary::PC pc,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	device_assert_m(false, "PSI instructions are not valid in machine code.");

	return pc + 1;
}

static __device__ ir::Binary::PC executeInvalidOpcode(
	Instruction* instruction, ir::Binary::PC pc,
	CoreSimBlock* parentBlock, unsigned threadId)
{
	device_assert_m(false, "Executed instruction with invalid opcode.");

	//TODO: Add exceptions
	return pc + 1;
}

typedef ir::Binary::PC (*JumpTablePointer)(Instruction*,
	ir::Binary::PC, CoreSimBlock*, unsigned);

static __device__ JumpTablePointer decodeTable[] = 
{
	executeAdd,
	executeAnd,
	executeAshr,
	executeAtom,
	executeBar,
	executeBitcast,
	executeBra,
	executeCall,
	executeFdiv,
	executeFmul,
	executeFpext,
	executeFptosi,
	executeFptoui,
	executeFpTrunc,
	executeFrem,
	executeLaunch,
	executeLd,
	executeLshr,
	executeMembar,
	executeMul,
	executeOr,
	executeRet,
	executeSetp,
	executeSext,
	executeSdiv,
	executeShl,
	executeSitofp,
	executeSrem,
	executeSt,
	executeSub,
	executeTrunc,
	executeUdiv,
	executeUitofp,
	executeUrem,
	executeXor,
	executeZext,
	executePhi,
	executePsi,
	executeInvalidOpcode
};

static __device__ const char* toString(Instruction::Opcode opcode)
{
	switch(opcode)
	{
	case Instruction::Add:     return "Add";
	case Instruction::And:     return "And";
	case Instruction::Ashr:    return "Ashr";
	case Instruction::Atom:    return "Atom";
	case Instruction::Bar:     return "Bar";
	case Instruction::Bitcast: return "Bitcast";
	case Instruction::Bra:     return "Bra";
	case Instruction::Call:    return "Call";
	case Instruction::Fdiv:    return "Fdiv";
	case Instruction::Fmul:    return "Fmul";
	case Instruction::Fpext:   return "Fpext";
	case Instruction::Fptosi:  return "Fptosi";
	case Instruction::Fptoui:  return "Fptoui";
	case Instruction::Fptrunc: return "Fptrunc";
	case Instruction::Frem:    return "Frem";
	case Instruction::Launch:  return "Launch";
	case Instruction::Ld:      return "Ld";
	case Instruction::Lshr:    return "Lshr";
	case Instruction::Membar:  return "Membar";
	case Instruction::Mul:     return "Mul";
	case Instruction::Or:      return "Or";
	case Instruction::Ret:     return "Ret";
	case Instruction::Setp:    return "Setp";
	case Instruction::Sext:    return "Sext";
	case Instruction::Sdiv:    return "Sdiv";
	case Instruction::Shl:     return "Shl";
	case Instruction::Sitofp:  return "Sitofp";
	case Instruction::Srem:    return "Srem";
	case Instruction::St:      return "St";
	case Instruction::Sub:     return "Sub";
	case Instruction::Trunc:   return "Trunc";
	case Instruction::Udiv:    return "Udiv";
	case Instruction::Uitofp:  return "Uitofp";
	case Instruction::Urem:    return "Urem";
	case Instruction::Xor:     return "Xor";
	case Instruction::Zext:    return "Zext";
	case Instruction::Phi:     return "Phi";
	case Instruction::Psi:     return "Psi";
	default: break;
	}
	
	return "invalid_instruction";
}

__device__ ir::Binary::PC CoreSimThread::executeInstruction(
	Instruction* instruction, ir::Binary::PC pc)
{
	JumpTablePointer decoderFunction = decodeTable[instruction->opcode];
	
	device_report("Thread %d, executing instruction[%d] '%s'\n", m_tId, (int)pc,
		toString(instruction->opcode));
	
	return decoderFunction(instruction, pc, m_parentBlock, m_tId);
}

}

}

