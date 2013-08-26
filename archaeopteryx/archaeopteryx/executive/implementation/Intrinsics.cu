/* 	\file Intrinsics.cpp
	\date Tuesday February 4, 2013
	\author Gregory Diamos
	\brief The source file for the archeopteryx intrinsic functions.

*/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/Intrinsics.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/executive/interface/OperandAccess.h>

#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/debug.h>
#include <archaeopteryx/util/interface/string.h>

#include <archaeopteryx/util/interface/map.h>

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

__device__ bool Intrinsics::isIntrinsic(const vanaheimr::as::Call* call,
	CoreSimBlock* block)
{
	cta_report("Checking if call is an intrinsic.\n");

	if(call->target.asOperand.mode != vanaheimr::as::Operand::Symbol)
	{
		return false;
	}

	const vanaheimr::as::SymbolOperand* symbol = &call->target.asSymbol;

	cta_report(" checking if symbol '%d' is "
		"an intrinsisc...\n", symbol->symbolTableOffset);

	util::string name = block->binary()->getSymbolName(
		symbol->symbolTableOffset);

	bool isIntrinsic = name.find("_Zintrinsic") == 0;

	if(isIntrinsic)
	{
		cta_report("  it is\n");
	}
	else
	{
		cta_report("  it isn't\n");
	}

	return isIntrinsic;
}

class Intrinsic
{
public:
	__device__ virtual ~Intrinsic() {}

public:
	__device__ virtual void execute(const vanaheimr::as::Call* call,
		CoreSimBlock* block, unsigned int threadId) = 0;
};

class IntrinsicDatabase
{
public:
	typedef util::map<util::string, Intrinsic*> IntrinsicMap;
	
public:
	__device__ ~IntrinsicDatabase();
	
public:
	__device__ Intrinsic* getIntrinsic(const util::string& name);
	__device__ void addIntrinsic(const util::string& name,
		Intrinsic* intrinsic);
	
private:
	IntrinsicMap _database;

};

__device__ IntrinsicDatabase* _intrinsics = 0;

__device__ void Intrinsics::execute(const vanaheimr::as::Call* call,
	CoreSimBlock* block, unsigned int threadId)
{
	device_assert(call->target.asOperand.mode ==
		vanaheimr::as::Operand::Symbol);

	const vanaheimr::as::SymbolOperand* symbol = &call->target.asSymbol;

	util::string name = block->binary()->getSymbolName(
		symbol->symbolTableOffset);

	device_report("thread %d, executing intrinsic '%s'\n", threadId,
		name.c_str());
	
	Intrinsic* intrinsic = _intrinsics->getIntrinsic(name);
	
	device_assert(intrinsic != 0);
	
	intrinsic->execute(call, block, threadId);
}

__device__ IntrinsicDatabase::~IntrinsicDatabase()
{
	// TODO in parallel
	for(IntrinsicDatabase::IntrinsicMap::iterator
		intrinsic = _database.begin();
		intrinsic != _database.end(); ++intrinsic)
	{
		delete intrinsic->second;
	}
}

__device__ Intrinsic* IntrinsicDatabase::getIntrinsic(const util::string& name)
{
	IntrinsicMap::iterator intrinsic = _database.find(name);
	
	if(intrinsic == _database.end()) return 0;
	
	return intrinsic->second;
}

__device__ void IntrinsicDatabase::addIntrinsic(const util::string& name,
	Intrinsic* intrinsic)
{
	device_assert(_database.count(name) == 0);

	_database[name] = intrinsic;
}

class GetNumberOfCtasInX : public Intrinsic
{
public:
	__device__ virtual void execute(const vanaheimr::as::Call* call,
		CoreSimBlock* block, unsigned int threadId)
	{
		uint64_t d = block->getSimulatedBlockCount();

		setRegister(getReturnRegister(call, block), block, threadId, d);
	}
};

class MadLoI32 : public Intrinsic
{
public:
	__device__ virtual void execute(const vanaheimr::as::Call* call,
		CoreSimBlock* block, unsigned int threadId)
	{
		uint32_t a = getOperand(call, block, threadId, 0);
		uint32_t b = getOperand(call, block, threadId, 1);
		uint32_t c = getOperand(call, block, threadId, 2);
		
		uint32_t d = a * b + c;

		setRegister(getReturnRegister(call, block), block, threadId, d);
	}
};

__device__ void Intrinsics::loadIntrinsics()
{
	_intrinsics = new IntrinsicDatabase;
	
	// TODO add intrinsics
	_intrinsics->addIntrinsic("_Zintrinsic_getspecial_nctaid_x",
		new GetNumberOfCtasInX);
	_intrinsics->addIntrinsic("_Zintrinsic_mad_lo_i32",
		new MadLoI32);
}

__device__ void Intrinsics::unloadIntrinsics()
{
	delete _intrinsics;
}

}

}

