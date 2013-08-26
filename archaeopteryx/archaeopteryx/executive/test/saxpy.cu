/*! \file   saxpy.cu
    \date   Saturday Feburary 26, 2011
    \author Gregory Diamos and Sudnya Padalikar 
        <gregory.diamos@gatech.edu, mailsudnya@gmail.com>
    \brief  A test for VIR and the simulator core.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Parameters.h>
#include <archaeopteryx/ir/interface/Instruction.h>
#include <archaeopteryx/util/interface/CudaUtilities.h>

// Standard Library Includes
#include <cstdlib>
#include <cstdio>

#define ARRAY_LENGTH 256
#define GLOBAL_MEMORY_WINDOW_SIZE 0x2000
#define REGISTERS_PER_THREAD 64
#define SIMULATED_THREADS ARRAY_LENGTH
#define TARGET_BLOCK_SIZE 2

/*
saxpy(int* y, int* x, int a)


    Begin:
        bitcast r11, "parameter_base";   // get address
        ld      r0, [r11]; // r0 is base of y
        ld      r1, [r11+8]; // r1 is base of x
        ld      r2, [r11+16]; // r2 is alpha

        bitcast r3,  "global_thread_id";
        zext    r12, r3;
        mul,    r4,  r12, 4;
        
        add     r5, r4, r0; // r5 is y[i]
        add     r6, r4, r1; // r6 is x[i]
        
        ld      r7, [r5];
        ld      r8, [r6];
        
        mul     r9,  r8, r2;
        add     r10, r7, r9;
        
        st      [r5], r10;
        
*/

__device__ void createSaxpy(void* parameters)
{
    ir::InstructionContainer* vir = 
        util::getParameter<ir::InstructionContainer*>(parameters);

    {
        ir::Bitcast& bitcast = vir[0].asBitcast; 
        
        bitcast.opcode = ir::Instruction::Bitcast;
        
        bitcast.d.asRegister.mode = ir::Operand::Register;
        bitcast.d.asRegister.type = ir::i64;
        bitcast.d.asRegister.reg  = 11;
        
        bitcast.a.asRegister.mode = ir::Operand::Register;
        bitcast.a.asRegister.type = ir::i64;
        bitcast.a.asRegister.reg  = 32;
    }
    
    {
        ir::Ld& load = vir[1].asLd; 

        load.opcode = ir::Instruction::Ld;
    
        load.d.asRegister.mode = ir::Operand::Register;
        load.d.asRegister.type = ir::i64;
        load.d.asRegister.reg  = 0;
        
        load.a.asIndirect.mode   = ir::Operand::Indirect;
        load.a.asIndirect.type   = ir::i64;
        load.a.asIndirect.reg    = 11;
        load.a.asIndirect.offset = 0;
    }
    
    {
        ir::Ld& load = vir[2].asLd; 

        load.opcode = ir::Instruction::Ld;
    
        load.d.asRegister.mode   = ir::Operand::Register;
        load.d.asRegister.type   = ir::i64;
        load.d.asRegister.reg    = 1;
        
        load.a.asIndirect.mode   = ir::Operand::Indirect;
        load.a.asIndirect.type   = ir::i64;
        load.a.asIndirect.reg    = 11;
        load.a.asIndirect.offset = 8;
    }
    
    {
        ir::Ld& load = vir[3].asLd; 

        load.opcode = ir::Instruction::Ld;
    
        load.d.asRegister.mode   = ir::Operand::Register;
        load.d.asRegister.type   = ir::i32;
        load.d.asRegister.reg    = 2;
        
        load.a.asIndirect.mode   = ir::Operand::Indirect;
        load.a.asIndirect.type   = ir::i64;
        load.a.asIndirect.reg    = 11;
        load.a.asIndirect.offset = 16;
    }
    
    {
        ir::Bitcast& bitcast = vir[4].asBitcast; 

        bitcast.opcode = ir::Instruction::Bitcast;
        
        bitcast.d.asRegister.mode = ir::Operand::Register;
        bitcast.d.asRegister.type = ir::i32;
        bitcast.d.asRegister.reg  = 3;
        
        bitcast.a.asRegister.mode = ir::Operand::Register;
        bitcast.a.asRegister.type = ir::i32;
        bitcast.a.asRegister.reg  = 33;
    }
    
    {
        ir::Zext& zext = vir[5].asZext; 

        zext.opcode = ir::Instruction::Zext;
        
        zext.d.asRegister.mode = ir::Operand::Register;
        zext.d.asRegister.type = ir::i64;
        zext.d.asRegister.reg  = 12;
        
        zext.a.asRegister.mode = ir::Operand::Register;
        zext.a.asRegister.type = ir::i32;
        zext.a.asRegister.reg  = 3;
    }

    {
        ir::Mul& multiply = vir[6].asMul; 

        multiply.opcode = ir::Instruction::Mul;
        
        multiply.d.asRegister.mode = ir::Operand::Register;
        multiply.d.asRegister.type = ir::i64;
        multiply.d.asRegister.reg  = 4;
        
        multiply.a.asRegister.mode = ir::Operand::Register;
        multiply.a.asRegister.type = ir::i64;
        multiply.a.asRegister.reg  = 12;

        multiply.b.asImmediate.mode = ir::Operand::Immediate;
        multiply.b.asImmediate.type = ir::i64;
        multiply.b.asImmediate.uint = 4;
    }

    {
        ir::Add& add = vir[7].asAdd; 

        add.opcode = ir::Instruction::Add;
        
        add.d.asRegister.mode = ir::Operand::Register;
        add.d.asRegister.type = ir::i64;
        add.d.asRegister.reg  = 5;
        
        add.a.asRegister.mode = ir::Operand::Register;
        add.a.asRegister.type = ir::i64;
        add.a.asRegister.reg  = 4;

        add.b.asRegister.mode = ir::Operand::Register;
        add.b.asRegister.type = ir::i64;
        add.b.asRegister.reg  = 0;
    }

    {
        ir::Add& add = vir[8].asAdd; 
        
        add.opcode = ir::Instruction::Add;
        
        add.d.asRegister.mode = ir::Operand::Register;
        add.d.asRegister.type = ir::i64;
        add.d.asRegister.reg  = 6;
        
        add.a.asRegister.mode = ir::Operand::Register;
        add.a.asRegister.type = ir::i64;
        add.a.asRegister.reg  = 4;

        add.b.asRegister.mode = ir::Operand::Register;
        add.b.asRegister.type = ir::i64;
        add.b.asRegister.reg  = 1;
    }
    
    {
        ir::Ld& load = vir[9].asLd; 
    
        load.opcode = ir::Instruction::Ld;
        
        load.d.asRegister.mode   = ir::Operand::Register;
        load.d.asRegister.type   = ir::i32;
        load.d.asRegister.reg    = 7;
        
        load.a.asIndirect.mode   = ir::Operand::Indirect;
        load.a.asIndirect.type   = ir::i64;
        load.a.asIndirect.reg    = 5;
        load.a.asIndirect.offset = 0;
    }
    
    {
        ir::Ld& load = vir[10].asLd; 

        load.opcode = ir::Instruction::Ld;        
    
        load.d.asRegister.mode   = ir::Operand::Register;
        load.d.asRegister.type   = ir::i32;
        load.d.asRegister.reg    = 8;
        
        load.a.asIndirect.mode   = ir::Operand::Indirect;
        load.a.asIndirect.type   = ir::i64;
        load.a.asIndirect.reg    = 6;
        load.a.asIndirect.offset = 0;
    }
    
    {
        ir::Mul& multiply = vir[11].asMul; 

        multiply.opcode = ir::Instruction::Mul;
                
        multiply.d.asRegister.mode = ir::Operand::Register;
        multiply.d.asRegister.type = ir::i32;
        multiply.d.asRegister.reg  = 9;
        
        multiply.a.asRegister.mode = ir::Operand::Register;
        multiply.a.asRegister.type = ir::i32;
        multiply.a.asRegister.reg  = 8;

        multiply.b.asRegister.mode = ir::Operand::Register;
        multiply.b.asRegister.type = ir::i32;
        multiply.b.asRegister.reg  = 2;
    }

    {
        ir::Add& add = vir[12].asAdd; 
        
        add.opcode = ir::Instruction::Add;

        add.d.asRegister.mode = ir::Operand::Register;
        add.d.asRegister.type = ir::i32;
        add.d.asRegister.reg  = 10;
        
        add.a.asRegister.mode = ir::Operand::Register;
        add.a.asRegister.type = ir::i32;
        add.a.asRegister.reg  = 7;

        add.b.asRegister.mode = ir::Operand::Register;
        add.b.asRegister.type = ir::i32;
        add.b.asRegister.reg  = 9;
    }

    {
        ir::St& store = vir[13].asSt; 
    
        store.opcode = ir::Instruction::St;

        store.d.asIndirect.mode   = ir::Operand::Indirect;
        store.d.asIndirect.type   = ir::i64;
        store.d.asIndirect.reg    = 5;
        store.d.asIndirect.offset = 0;
        
        store.a.asRegister.mode   = ir::Operand::Register;
        store.a.asRegister.type   = ir::i32;
        store.a.asRegister.reg    = 10;
    }
    
    {
        ir::Ret& ret = vir[14].asRet; 
    
        ret.opcode = ir::Instruction::Ret;
    }

}

typedef long long unsigned int uint64;
typedef long long unsigned int Register;
typedef Register* RegisterFile;

class SimulatorState 
{
    public:
        /* Parameters seen by entire kernel */
        uint64 globalMemoryWindowHi;
        uint64 globalMemoryWindowLow;
        void*  globalMemoryWindow;
        uint64 baseProgramCounter;
        ir::InstructionContainer* instructionMemory;
        ir::Parameters* parameters;
        /* Parameters seen by entire block */
        /* Parameters seen by each thread */
        RegisterFile registerFile;
    public:
        __device__ SimulatorState(uint64 gh, 
            uint64 gl, void* g,
            uint64 b, RegisterFile r,
            ir::InstructionContainer* i, ir::Parameters* p)
            : globalMemoryWindowHi(gh),
              globalMemoryWindowLow(gl),
              globalMemoryWindow(g),
              baseProgramCounter(b),
              instructionMemory(i),
              parameters(p),
              registerFile(r)
            {};
		__device__ SimulatorState() {};
};

////////////////////////////////////////////////////////////////////////////////
// createParameters
////////////////////////////////////////////////////////////////////////////////
__device__ void createParameters(void* parameterList)
{
    ir::Parameters* parameters = 
        util::getParameter<ir::Parameters*>(parameterList);
    {
        parameters->ctaSize = 1;
        parameters->numberOfThreads = 1;
        parameters->localMemoryWindowBase = 0x00;
        parameters->localMemoryWindowSize = 0;
        parameters->sharedMemoryWindowBase = 0x00;
        parameters->sharedMemoryWindowSize = 0;
        parameters->parameterMemoryWindowBase = 0x1000;
        parameters->parameterMemoryWindowSize = 8+8+4;
    }
}

////////////////////////////////////////////////////////////////////////////////
// setupSimulatorState
////////////////////////////////////////////////////////////////////////////////
__device__ void setupSimulatorState(void* parameters)
{
    SimulatorState* state = util::getParameter<SimulatorState*>(parameters, 0);
    ir::InstructionContainer* instructionMemory = 
        util::getParameter<ir::InstructionContainer*>(
        parameters, sizeof(SimulatorState*));
    ir::Parameters* parameterDescription = 
        util::getParameter<ir::Parameters*>(parameters,
        sizeof(SimulatorState*) + sizeof(ir::InstructionContainer*));

    RegisterFile registerFile = (RegisterFile)std::malloc(
        sizeof(Register)*REGISTERS_PER_THREAD*SIMULATED_THREADS);
    memset(registerFile, 0, sizeof(Register)*REGISTERS_PER_THREAD*SIMULATED_THREADS);
    void* globalMemoryWindow = std::malloc(GLOBAL_MEMORY_WINDOW_SIZE);

    *state = SimulatorState(GLOBAL_MEMORY_WINDOW_SIZE, 0x0, globalMemoryWindow,
        0, registerFile, instructionMemory, parameterDescription);

	long long unsigned xAddress = ARRAY_LENGTH*sizeof(int);
	long long unsigned yAddress = 0;
	int a = 2;
	
	std::memcpy((char*)globalMemoryWindow
		+ parameterDescription->parameterMemoryWindowBase,
		&yAddress, sizeof(long long unsigned));
	std::memcpy((char*)globalMemoryWindow
		+ parameterDescription->parameterMemoryWindowBase + 8,
		&xAddress, sizeof(long long unsigned));
	std::memcpy((char*)globalMemoryWindow
		+ parameterDescription->parameterMemoryWindowBase + 16,
		&a, sizeof(int));

	int* yData = (int*)((char*)globalMemoryWindow + yAddress);
	int* xData = (int*)((char*)globalMemoryWindow + xAddress);

	for(int i = 0; i < ARRAY_LENGTH; ++i)
	{
		yData[i] = i;
		xData[i] = i;
	}
}

////////////////////////////////////////////////////////////////////////////////
// float_cast
////////////////////////////////////////////////////////////////////////////////
union FloatUintUnionCast
{
	float f;
	long long unsigned int i;
};

__device__ float float_cast(long long unsigned int reg)
{
	FloatUintUnionCast cast;
	
	cast.i = reg;
	
	return cast.f;
}

////////////////////////////////////////////////////////////////////////////////
// double_cast
////////////////////////////////////////////////////////////////////////////////
union DoubleUintUnionCast
{
	double f;
	long long unsigned int i;
};

__device__ double double_cast(long long unsigned int reg)
{
	DoubleUintUnionCast cast;
	
	cast.i = reg;
	
	return cast.f;
}

////////////////////////////////////////////////////////////////////////////////
// runSimulation
////////////////////////////////////////////////////////////////////////////////
__device__ void runSimulation(void* parameters)
{
    SimulatorState* state = util::getParameter<SimulatorState*>(parameters, 0);
    unsigned int threadId = util::getGlobalThreadId();

    if(threadId >= SIMULATED_THREADS) return;

    RegisterFile registerFile = &state->registerFile[threadId];
    uint64 pc = state->baseProgramCounter;
    bool running = true;

    registerFile[33 * SIMULATED_THREADS] = threadId;
    registerFile[32 * SIMULATED_THREADS]
        = state->parameters->parameterMemoryWindowBase;

    printf("Running simulation for thread %d...\n", threadId);
    printf(" global-hi:  %x\n", state->globalMemoryWindowHi);
    printf(" global-lo:  %x\n", state->globalMemoryWindowLow);
    printf(" physical:   %x\n", state->globalMemoryWindow);
    printf(" pc-base:    %x\n", state->baseProgramCounter);
    printf(" param-base: %x\n", state->parameters->parameterMemoryWindowBase);

    while(running)
    {
        ir::InstructionContainer instruction = state->instructionMemory[pc];
        
        ir::Instruction& baseInstruction = instruction.asInstruction;
        
        switch(baseInstruction.opcode)
        {
            case ir::Instruction::Add:
                {
                    ir::Add& add = instruction.asAdd;
                    
                    ir::RegisterType aId = add.a.asRegister.reg;
                    ir::RegisterType bId = add.b.asRegister.reg;
                    ir::RegisterType dId = add.d.asRegister.reg;

                    Register a = registerFile[aId*SIMULATED_THREADS];
                    Register b = registerFile[bId*SIMULATED_THREADS];
                    Register d = 0;
                    
                    switch(add.a.asRegister.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (char)a + (char)b;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (short)a + (short)b;
                                break;
                            }
                        case ir::i32:
                            {
                                d = (int)a + (int)b;
                                break;
                            }
                        case ir::f32:
                            {
                                d = float_cast(a) + float_cast(b);
                                break;
                            }
                        case ir::i64:
                            {
                                d = (long long int)a + (long long int)b;
                                break;
                            }
                        case ir::f64:
                            {
                                d = double_cast(a) + double_cast(b);
                                break;
                            }
                        default: break;
                    }
                    
                    registerFile[dId*SIMULATED_THREADS] = d;
                    printf("[Thread:%d] %llu Add %llu to give %llu \n",threadId, a, b, d);
                    ++pc;
                    break;
                }
            case ir::Instruction::Bitcast:
                {
                    ir::Bitcast& bitcast = instruction.asBitcast;
                    
                    ir::RegisterType aId = bitcast.a.asRegister.reg;
                    ir::RegisterType dId = bitcast.d.asRegister.reg;
                    
                    registerFile[dId*SIMULATED_THREADS] = registerFile[aId*SIMULATED_THREADS];
                    ++pc;
                    printf("[Thread:%d] %llu Bitcast %llu \n", threadId, registerFile[aId*SIMULATED_THREADS], registerFile[dId*SIMULATED_THREADS]);
                    break;
                }
            case ir::Instruction::Ld:
                {
                    ir::Ld& load = instruction.asLd;
                    
                    ir::RegisterType dId = load.d.asRegister.reg;
                    ir::RegisterType aId = load.a.asIndirect.reg;
                    int offset = load.a.asIndirect.offset;
                    uint64 vaddress = registerFile[aId*SIMULATED_THREADS];
                    vaddress += offset;
                    uint64 base = (uint64)(size_t)state->globalMemoryWindow;
                    uint64 address = vaddress - state->globalMemoryWindowLow
                        + base;
                    
                    Register value = 0;
                    
                    switch(load.d.asRegister.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                value = *((char*)(size_t)address);
                                break;
                            }
                        case ir::i16:
                            {
                                value = *((short*)(size_t)address);
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                value = *((int*)(size_t)address);
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                value = *((long long int*)(size_t)address);
                                break;
                            }
                        default: break;
                    }
                    
                    registerFile[dId*SIMULATED_THREADS] = value;
                    ++pc;
                    printf("[Thread:%d] LD r[%d], r[%d] [value %lld, address %llx] \n", threadId, aId, dId, value, vaddress );
                    break;
                }
            case ir::Instruction::Mul:
                {
                    ir::Mul& mul = instruction.asMul;
                    
                    ir::RegisterType dId = mul.d.asRegister.reg;

                    Register a = 0;
                    Register b = 0;
                    Register d = 0;

                    ir::DataType type = ir::InvalidDataType;

                    if(mul.a.asRegister.mode == ir::Operand::Register)
                    {
                        ir::RegisterType aId = mul.a.asRegister.reg;

                        type = mul.a.asRegister.type;
                        
                        a = registerFile[aId*SIMULATED_THREADS];
                        printf("[Thread:%d] mul is as register mode, r[%d] = %llu\n", threadId, aId, a);
                    }
                    else
                    {
                        type = mul.a.asImmediate.type;

                        switch(mul.a.asImmediate.type)
                        {
                            case ir::i1:  /* fall through */
                            case ir::i8:  /* fall through */
                            case ir::i16: /* fall through */
                            case ir::i32: /* fall through */
                            case ir::i64:
                                {
                                    a = mul.a.asImmediate.uint;
                                    break;
                                }
                            case ir::f32:
                                {
                                    a = float_cast(
                                        (float)mul.a.asImmediate.fp);
                                    break;
                                }
                            case ir::f64:
                                {
                                    a = mul.a.asImmediate.fp;
                                    break;
                                }
                            default: break;                        
                        }
                    }

                    if(mul.b.asRegister.mode == ir::Operand::Register)
                    {
                        ir::RegisterType bId = mul.b.asRegister.reg;
                        
                        b = registerFile[bId*SIMULATED_THREADS];
                        printf("[Thread:%d] mul is as register mode, r[%d] = %llu\n",threadId, bId, b);
                    }
                    else
                    {
                        switch(mul.b.asImmediate.type)
                        {
                            case ir::i1:  /* fall through */
                            case ir::i8:  /* fall through */
                            case ir::i16: /* fall through */
                            case ir::i32: /* fall through */
                            case ir::i64:
                                {
                                    b = mul.b.asImmediate.uint;
                                    break;
                                }
                            case ir::f32:
                                {
                                    b = float_cast(
                                        (float)mul.b.asImmediate.fp);
                                    break;
                                }
                            case ir::f64:
                                {
                                    b = mul.b.asImmediate.fp;
                                    break;
                                }
                            default: break;                        
                        }
                    }
                    
                    switch(type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (char)a * (char)b;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (short)a * (short)b;
                                break;
                            }
                        case ir::i32:
                            {
                                d = (int)a * (int)b;
                                break;
                            }
                        case ir::f32:
                            {
                                d = float_cast(a) * float_cast(b);
                                break;
                            }
                        case ir::i64:
                            {
                                d = (long long int)a * (long long int)b;
                                break;
                            }
                        case ir::f64:
                            {
                                d = double_cast(a) * double_cast(b);
                                break;
                            }
                        default: break;
                    }

                    registerFile[dId*SIMULATED_THREADS] = d;
                    printf("[Thread:%d] %llu = %llu  Mul  %llu \n", threadId, d, a, b);

                    ++pc;
                    break;
                }
            case ir::Instruction::Ret:
                {
                    running = false;
                    break;
                }
            case ir::Instruction::St:
                {
                    ir::St& store = instruction.asSt;
                    
                    ir::RegisterType aId = store.a.asRegister.reg;
                    ir::RegisterType dId = store.d.asIndirect.reg;
                    int offset = store.d.asIndirect.offset;
                    uint64 vaddress = registerFile[dId*SIMULATED_THREADS];
                    vaddress += offset;
                    uint64 base = (uint64)(size_t)state->globalMemoryWindow;
                    uint64 address = vaddress - state->globalMemoryWindowLow
                        + base;
                    
                    Register value = registerFile[aId*SIMULATED_THREADS];

                    switch(store.a.asRegister.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                *((char*)(size_t)address) = value;
                                break;
                            }
                        case ir::i16:
                            {
                                *((short*)(size_t)address) = value;
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                *((int*)(size_t)address) = value;
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                *((long long int*)(size_t)address) = value;
                                break;
                            }
                        default: break;
                    }
                    ++pc;
                    printf("[Thread:%d] ST %llu address %llx \n", threadId, value, vaddress);
                    break;
                }
            case ir::Instruction::Zext:
                {
                    ir::Zext& zext = instruction.asZext;
                    
                    ir::RegisterType dId = zext.d.asRegister.reg;
                    ir::RegisterType aId = zext.a.asRegister.reg;
                    
                    Register a = registerFile[aId*SIMULATED_THREADS];
                    Register d = 0;
                    
                    switch(zext.a.asRegister.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (unsigned char)a;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (unsigned short)a;
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                d = (unsigned int)a;
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                d = a;
                                break;
                            }
                        default: break;
                    }
                    
                    registerFile[dId*SIMULATED_THREADS] = d;
                    printf("[Thread:%d] %llu Zext %llu \n", threadId, a, d);
                    ++pc;
                    break;
                }
            default:
                {
                    printf("[Thread:%d] Running unknown instruction at PC %d\n", threadId, pc);
                    ++pc;
                    break;
                }
        }
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// memoryCompare
////////////////////////////////////////////////////////////////////////////////
__device__ void memoryCompare(void* parameters)
{
    SimulatorState* state  = util::getParameter<SimulatorState*>(parameters, 0);
    ir::Parameters* params = util::getParameter<ir::Parameters*>(parameters,
    	sizeof(SimulatorState*));
	
	int* y = (int*)((char*)state->globalMemoryWindow);
	
	bool passed = true;
	
	for(int i = 0; i < ARRAY_LENGTH; ++i)
	{
		if(y[i] != i * 2 + i)
		{
			printf("At y[%d], computed %d != reference %d\n",
				i, y[i], i * 2 + i);
			passed = false;
		}
	}
	
	if(passed)
	{
		printf("Memory image check succeeded!\n");
	}
	else
	{
		printf("Memory image check failed!\n");
	}
}

////////////////////////////////////////////////////////////////////////////////
// deleteInstructionMemory
////////////////////////////////////////////////////////////////////////////////
__device__ void deleteInstructionMemory(void* parameters)
{
    ir::InstructionContainer* instructionMemory = 
        util::getParameter<ir::InstructionContainer*>(parameters, 0);

    delete[] instructionMemory;
}

////////////////////////////////////////////////////////////////////////////////
// deleteSimulatorState
////////////////////////////////////////////////////////////////////////////////
__device__ void deleteSimulatorState(void* parameters)
{
    SimulatorState* state = util::getParameter<SimulatorState*>(parameters, 0);
	
    delete state;
}

////////////////////////////////////////////////////////////////////////////////
// deleteParameters
////////////////////////////////////////////////////////////////////////////////
__device__ void deleteParameters(void* parameters)
{
    ir::Parameters* params = util::getParameter<ir::Parameters*>(parameters, 0);
	
    delete params;
}

__global__ void system()
{
    ir::InstructionContainer* instructionMemory =
   		new ir::InstructionContainer[14];
    SimulatorState* state = new SimulatorState;
	ir::Parameters* parameters = new ir::Parameters;

    // 1) call createSaxypy()
    util::async_system_call(1, 1, "createSaxpy", instructionMemory);
    //    __bar()
	// 2) call setupParameters
	util::async_system_call(1, 1, "createParameters", parameters);
	//    __bar()
    // 3) call setupSimulatorState()
    util::async_system_call(1, 1, "setupSimulatorState", state,
    	instructionMemory, parameters);
    //    __bar()
    // 4) call runSimulation()
    util::async_system_call(
    	(SIMULATED_THREADS + TARGET_BLOCK_SIZE - 1) / TARGET_BLOCK_SIZE,
    	TARGET_BLOCK_SIZE, "runSimulation", state);
    //    __bar()
    // 5) call memoryCompare()
    util::async_system_call(1,1,"memoryCompare", state, parameters);
    //    __bar()
    util::async_system_call(1,1,"deleteInstructionMemory", instructionMemory);
    util::async_system_call(1,1,"deleteSimulatorState", state);
    util::async_system_call(1,1,"deleteParameterMemory", parameters);
}

__global__ void setupFunctionTable()
{
	util::functionTable[0].name     = "createSaxpy";
	util::functionTable[0].function =  createSaxpy;

	util::functionTable[1].name     = "setupSimulatorState";
	util::functionTable[1].function =  setupSimulatorState;

	util::functionTable[2].name     = "createParameters";
	util::functionTable[2].function =  createParameters;

	util::functionTable[3].name     = "runSimulation";
	util::functionTable[3].function =  runSimulation;

	util::functionTable[4].name     = "memoryCompare";
	util::functionTable[4].function =  memoryCompare;

	util::functionTable[5].name     = "deleteInstructionMemory";
	util::functionTable[5].function =  deleteInstructionMemory;

	util::functionTable[6].name     = "deleteSimulatorState";
	util::functionTable[6].function =  deleteSimulatorState;

	util::functionTable[7].name     = "deleteParameters";
	util::functionTable[7].function =  deleteParameters;
}

int main(int argc, char** argv)
{
	util::setupHostReflection();

	setupFunctionTable<<<1, 1, 1>>>();
	system<<<1, 1, 1>>>();
	
	util::teardownHostReflection();

	return 0;
}

