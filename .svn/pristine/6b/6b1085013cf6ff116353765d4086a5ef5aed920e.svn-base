/*! \file   CoreSimThread.cpp
    \date   Saturday May 8, 2011
    \author Gregory and Sudnya Diamos
        <gregory.diamos@gatech.edu, mailsudnya@gmail.com>
    \brief  The source file for the Core simulator of the thread class.
*/

#include <archaeopteryx/executive/interface/CoreSimThread.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/ir/interface/Operand.h>
#include <archaeopteryx/ir/interface/Instruction.h>

#include <archaeopteryx/util/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

// Typedefs 
typedef executive::CoreSimThread::Value Value;
typedef executive::CoreSimThread::SValue SValue;
typedef executive::CoreSimThread::Address Address;

namespace executive
{

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
__device__ T bitcast(const F& from)
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

static __device__ CoreSimThread::Value getRegisterOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::RegisterOperand& reg = static_cast<const ir::RegisterOperand&>(operand); 

    CoreSimThread::Value value = block->getRegister(threadId, reg.reg);

    return value;
}

static __device__ CoreSimThread::Value getImmediateOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::ImmediateOperand& imm = static_cast<const ir::ImmediateOperand&>(operand); 

    return imm.uint;
}

static __device__ CoreSimThread::Value getPredicateOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::PredicateOperand& reg = static_cast<const ir::PredicateOperand&>(operand); 
    //FIX ME    
    
    Value value = block->getRegister(threadId, reg.reg);

    switch(reg.modifier)
    {
    case ir::PredicateOperand::StraightPredicate:
    {
        value = value;
        break;
    }
    // TODO
    }

    return value;
}

static __device__ CoreSimThread::Value getIndirectOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::IndirectOperand& indirect = static_cast<const ir::IndirectOperand&>(operand); 
    
    Value address = block->getRegister(threadId, indirect.reg) + indirect.offset;

    //FIXMe    
    return address;
}


typedef Value (*GetOperandValuePointer)(const ir::Operand&, CoreSimBlock*, unsigned);

static __device__ GetOperandValuePointer getOperandFunctionTable[] = {
    getRegisterOperand,
    getImmediateOperand,
    getPredicateOperand,
    getIndirectOperand
};

static __device__ CoreSimThread::Value getOperand(const ir::Operand& operand, CoreSimBlock* parentBlock, unsigned threadId)
{
    GetOperandValuePointer function = getOperandFunctionTable[operand.mode];

    return function(operand, parentBlock, threadId);
}

static __device__ CoreSimThread::Value getOperand(const ir::OperandContainer& operandContainer, CoreSimBlock* parentBlock, unsigned threadId)
{
    return getOperand(operandContainer.asOperand, parentBlock, threadId);
}

static void __device__ setRegister(ir::OperandContainer& operandContainer, CoreSimBlock* parentBlock, unsigned threadId, const Value& result)
{
    const ir::RegisterOperand& reg = operandContainer.asRegister;

    parentBlock->setRegister(threadId, reg.reg, result);
}

static __device__ ir::Binary::PC executeAdd(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Add* add = static_cast<ir::Add*>(instruction);

    Value a = getOperand(add->a, parentBlock, threadId);
    Value b = getOperand(add->b, parentBlock, threadId);

    Value d = a + b;

    setRegister(add->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeAnd(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::And* andd = static_cast<ir::And*>(instruction);

    Value a = getOperand(andd->a, parentBlock, threadId);
    Value b = getOperand(andd->b, parentBlock, threadId);

    Value d = a & b;

    setRegister(andd->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeAshr(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Ashr* ashr = static_cast<ir::Ashr*>(instruction);

    SValue a = getOperand(ashr->a, parentBlock, threadId);
    Value b = getOperand(ashr->b, parentBlock, threadId);

    SValue d = a >> b;

    setRegister(ashr->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeAtom(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Atom* atom = static_cast<ir::Atom*>(instruction);

    Value a = getOperand(atom->a, parentBlock, threadId);
    Value b = getOperand(atom->b, parentBlock, threadId);

    Value physical = parentBlock->translateVirtualToPhysical(a);

    //TO DO
    Value d = atomicAdd(bitcast<Value*>(physical), b);

    setRegister(atom->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeBar(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    parentBlock->barrier(threadId);

    return pc+1;
}

static __device__ ir::Binary::PC executeBitcast(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Bitcast* bitcast = static_cast<ir::Bitcast*>(instruction);

    Value a = getOperand(bitcast->a, parentBlock, threadId);

    setRegister(bitcast->d, parentBlock, threadId, a);

    return pc+1;
}

static __device__ ir::Binary::PC executeBra(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Bra* bra = static_cast<ir::Bra*>(instruction);

    Value a = getOperand(bra->target, parentBlock, threadId);

    //TO DO
    return a;
}

static __device__ ir::Binary::PC executeFpext(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Fpext* fpext = static_cast<ir::Fpext*>(instruction);

    Value a = getOperand(fpext->a, parentBlock, threadId);

    float temp = bitcast<float>(a); 
    double d = temp;

    setRegister(fpext->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeFptosi(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Fptosi* fptosi = static_cast<ir::Fptosi*>(instruction);

    Value a = getOperand(fptosi->a, parentBlock, threadId);

    float temp = bitcast<float>(a);
    SValue d   = temp;

    setRegister(fptosi->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeFptoui(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Fptoui* fptoui = static_cast<ir::Fptoui*>(instruction);

    Value a = getOperand(fptoui->a, parentBlock, threadId);

    float temp = bitcast<float>(a);
    Value d    = temp;

    setRegister(fptoui->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeFpTrunc(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Fptrunc* fptrunc = static_cast<ir::Fptrunc*>(instruction);

    Value a = getOperand(fptrunc->a, parentBlock, threadId);

    double temp = bitcast<double>(a);
    float d     = temp;

    setRegister(fptrunc->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeLd(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Ld* ld = static_cast<ir::Ld*>(instruction);

    Value a = getOperand(ld->a, parentBlock, threadId);

    Value physical = parentBlock->translateVirtualToPhysical(a);

    Value d = 0;
    
    switch(ld->d.asIndirect.type)
    {
        case ir::i1:
        case ir::i8:
        {
            d = *bitcast<uint8_t*>(physical);
            break;
        }
        case ir::i16:
        {
            d = *bitcast<uint16_t*>(physical);
            break;
        }
        case ir::f32:
        case ir::i32:
        {
            d = *bitcast<uint32_t*>(physical);
            break;
        }
        case ir::f64:
        case ir::i64:
        {
            d = *bitcast<uint64_t*>(physical);
            break;
        }
        default: break;
    }

    setRegister(ld->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeLshr(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Lshr* lshr = static_cast<ir::Lshr*>(instruction);

    Value a = getOperand(lshr->a, parentBlock, threadId);
    Value b = getOperand(lshr->b, parentBlock, threadId);

    Value d = a >> b;

    setRegister(lshr->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeMembar(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    //__threadfence_block();
    return pc+1;
}

static __device__ ir::Binary::PC executeMul(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Mul* mul = static_cast<ir::Mul*>(instruction);

    Value a = getOperand(mul->a, parentBlock, threadId);
    Value b = getOperand(mul->b, parentBlock, threadId);

    Value d = a * b;

    setRegister(mul->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeOr(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Or* orr = static_cast<ir::Or*>(instruction);

    Value a = getOperand(orr->a, parentBlock, threadId);
    Value b = getOperand(orr->b, parentBlock, threadId);

    Value d = a | b;

    setRegister(orr->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeRet(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    return parentBlock->returned(threadId, pc); 
}

static __device__ ir::Binary::PC executeSetP(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::SetP* setp = static_cast<ir::SetP*>(instruction);

    Value a = getOperand(setp->a, parentBlock, threadId);
    Value b = getOperand(setp->b, parentBlock, threadId);

    //TO DO
    Value d = a > b ? 1 : 0 ;
    setRegister(setp->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeSext(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Sext* sext = static_cast<ir::Sext*>(instruction);

    Value a = getOperand(sext->a, parentBlock, threadId);

    int temp = bitcast<int>(a);
    SValue d = temp;

    setRegister(sext->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeSdiv(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Sdiv* sdiv = static_cast<ir::Sdiv*>(instruction);

    Value a = getOperand(sdiv->a, parentBlock, threadId);
    Value b = getOperand(sdiv->b, parentBlock, threadId);

    //TO DO
    SValue d = (SValue) a / (SValue) b;
    setRegister(sdiv->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeShl(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Shl* shl = static_cast<ir::Shl*>(instruction);

    Value a = getOperand(shl->a, parentBlock, threadId);
    Value b = getOperand(shl->b, parentBlock, threadId);

    Value d = a << b;

    setRegister(shl->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeSitofp(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Sitofp* sitofp = static_cast<ir::Sitofp*>(instruction);

    Value a = getOperand(sitofp->a, parentBlock, threadId);

    //TO DO
    float d = (SValue)a;

    setRegister(sitofp->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeSrem(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Srem* srem = static_cast<ir::Srem*>(instruction);

    SValue a = getOperand(srem->a, parentBlock, threadId);
    SValue b = getOperand(srem->b, parentBlock, threadId);

    SValue d = a % b;

    setRegister(srem->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeSt(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::St* st = static_cast<ir::St*>(instruction);

    Value d = getOperand(st->d, parentBlock, threadId);
    Value physical = parentBlock->translateVirtualToPhysical(d);

    Value a = getOperand(st->a, parentBlock, threadId);

    switch(st->a.asIndirect.type)
    {
        case ir::i1:
        case ir::i8:
        {
            *bitcast<uint8_t*>(physical) = a;
            break;
        }
        case ir::i16:
        {
            *bitcast<uint16_t*>(physical) = a;
            break;
        }
        case ir::f32:
        case ir::i32:
        {
            *bitcast<uint32_t*>(physical) = a;
            break;
        }
        case ir::f64:
        case ir::i64:
        {
            *bitcast<uint64_t*>(physical) = a;
            break;
        }
        default: break;
    }

    return pc+1;
}

static __device__ ir::Binary::PC executeSub(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Sub* sub = static_cast<ir::Sub*>(instruction);

    Value a = getOperand(sub->a, parentBlock, threadId);
    Value b = getOperand(sub->b, parentBlock, threadId);

    Value d = a - b;

    setRegister(sub->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeTrunc(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Trunc* trunc = static_cast<ir::Trunc*>(instruction);

    Value a = getOperand(trunc->a, parentBlock, threadId);

    //TO DO
    Value d = unsigned (a & 0x00000000FFFFFFFFULL); 

    setRegister(trunc->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeUdiv(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Udiv* udiv = static_cast<ir::Udiv*>(instruction);

    Value a = getOperand(udiv->a, parentBlock, threadId);
    Value b = getOperand(udiv->b, parentBlock, threadId);

    //TO DO
    Value d = a / b;

    setRegister(udiv->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeUitofp(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Uitofp* uitofp = static_cast<ir::Uitofp*>(instruction);

    Value a = getOperand(uitofp->a, parentBlock, threadId);

    //TO DO
    float d = a;

    setRegister(uitofp->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeUrem(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Urem* urem = static_cast<ir::Urem*>(instruction);

    Value a = getOperand(urem->a, parentBlock, threadId);
    Value b = getOperand(urem->b, parentBlock, threadId);

    //TO DO
    Value d = a % b;
    setRegister(urem->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeXor(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Xor* xorr = static_cast<ir::Xor*>(instruction);

    Value a = getOperand(xorr->a, parentBlock, threadId);
    Value b = getOperand(xorr->b, parentBlock, threadId);

    Value d = a ^ b;

    setRegister(xorr->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeZext(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    ir::Zext* zext = static_cast<ir::Zext*>(instruction);

    Value a = getOperand(zext->a, parentBlock, threadId);

    //TO DO
    Value d = (unsigned int)a;
    setRegister(zext->d, parentBlock, threadId, d);
    return pc+1;
}

static __device__ ir::Binary::PC executeInvalidOpcode(ir::Instruction* instruction, ir::Binary::PC pc, CoreSimBlock* parentBlock, unsigned threadId)
{
    // TODO add this

    //TODO: Add exceptions
    return pc+1;
}

typedef ir::Binary::PC (*JumpTablePointer)(ir::Instruction*, ir::Binary::PC, CoreSimBlock*, unsigned);

static __device__ JumpTablePointer decodeTable[] = 
{
    executeAdd,
    executeAnd,
    executeAshr,
    executeAtom,
    executeBar,
    executeBitcast,
    executeBra,
    executeFpext,
    executeFptosi,
    executeFptoui,
    executeFpTrunc,
    executeLd,
    executeLshr,
    executeMembar,
    executeMul,
    executeOr,
    executeRet,
    executeSetP,
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
    executeInvalidOpcode
};

static __device__ const char* toString(ir::Instruction::Opcode opcode)
{
    switch(opcode)
    {
    case ir::Instruction::Add: return "Add";
    case ir::Instruction::And: return "And";
    case ir::Instruction::Ashr: return "Ashr";
    case ir::Instruction::Atom: return "Atom";
    case ir::Instruction::Bar: return "Bar";
    case ir::Instruction::Bitcast: return "Bitcast";
    case ir::Instruction::Bra: return "Bra";
    case ir::Instruction::Fpext: return "Fpext";
    case ir::Instruction::Fptosi: return "Fptosi";
    case ir::Instruction::Fptoui: return "Fptoui";
    case ir::Instruction::Fptrunc: return "Fptrunc";
    case ir::Instruction::Ld: return "Ld";
    case ir::Instruction::Lshr: return "Lshr";
    case ir::Instruction::Membar: return "Membar";
    case ir::Instruction::Mul: return "Mul";
    case ir::Instruction::Or: return "Or";
    case ir::Instruction::Ret: return "Ret";
    case ir::Instruction::SetP: return "SetP";
    case ir::Instruction::Sext: return "Sext";
    case ir::Instruction::Sdiv: return "Sdiv";
    case ir::Instruction::Shl: return "Shl";
    case ir::Instruction::Sitofp: return "Sitofp";
    case ir::Instruction::Srem: return "Srem";
    case ir::Instruction::St: return "St";
    case ir::Instruction::Sub: return "Sub";
    case ir::Instruction::Trunc: return "Trunc";
    case ir::Instruction::Udiv: return "Udiv";
    case ir::Instruction::Uitofp: return "Uitofp";
    case ir::Instruction::Urem: return "Urem";
    case ir::Instruction::Xor: return "Xor";
    case ir::Instruction::Zext: return "Zext";
    default: break;
    }
    
    return "invalid_instruction";
}

__device__ ir::Binary::PC CoreSimThread::executeInstruction(
    ir::Instruction* instruction, ir::Binary::PC pc)
{
    JumpTablePointer decoderFunction = decodeTable[instruction->opcode];
    
    device_report("Thread %d, executing instruction[%d] '%s'\n", m_tId, (int)pc,
        toString(instruction->opcode));
    
    return decoderFunction(instruction, pc, m_parentBlock, m_tId);
}

}

