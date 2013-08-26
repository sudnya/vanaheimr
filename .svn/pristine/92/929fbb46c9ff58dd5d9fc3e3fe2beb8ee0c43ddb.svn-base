/*! \file FirstBinary.cpp 
    \date   Saturday October 23, 2011
    \author Sudnya Padalikar
            <mailsudnya@gmail.com>
    \brief A test program to write a binary file for saxpy.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

// Standard Library Includes
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cstring>

#define ARRAY_LENGTH 256
#define GLOBAL_MEMORY_WINDOW_SIZE 0x2000
#define REGISTERS_PER_THREAD 64
#define SIMULATED_THREADS ARRAY_LENGTH
#define TARGET_BLOCK_SIZE 2


namespace util { class File;                 }
namespace ir   { union InstructionContainer; }


#include <archaeopteryx/runtime/test/FirstBinary.h>

int main()
{
    Header binHeader;
    binHeader.dataPages = 0;
    binHeader.codePages = 1;
    binHeader.symbols   = 1;
    binHeader.strings   = 1;

    typedef unsigned int PageDataType[1 << 13];

    PageDataType page;
    std::memset(page, 0, sizeof(PageDataType));
    ir::InstructionContainer* vir = (ir::InstructionContainer*)page;

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
 
    std::ofstream ofs("BinarySaxpy.exe", std::ofstream::binary);
    
    SymbolTableEntry symbolInfo;
    symbolInfo.type              = 2; //FunctionSymbolType from the enum. Hardcoded for now.
    symbolInfo.stringTableOffset = 0;
    symbolInfo.pageId            = 0;
    symbolInfo.pageOffset        = 0;
    symbolInfo.attributes        = 0;
    
    std::string entryPoint("saxpy");
    if (!ofs.is_open())
    {
        std::cout << "Could not open the binary file\n";
        exit(1);
    }
    ofs.write((char*)&binHeader, sizeof(Header));
    ofs.write((char*)vir, sizeof(PageDataType));
    ofs.write((char*)&symbolInfo, sizeof(SymbolTableEntry));
    ofs.write(entryPoint.c_str(), entryPoint.size() + 1);
    ofs.close();
}
