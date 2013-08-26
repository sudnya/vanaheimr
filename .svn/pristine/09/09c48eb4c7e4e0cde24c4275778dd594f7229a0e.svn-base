/*! \file   ChaitinBriggsRegisterAllocatorPass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ChaitinBriggsRegisterAllocatorPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ChaitinBriggsRegisterAllocatorPass.h>

#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

#include <vanaheimr/machine/interface/MachineModel.h>

#include <vanaheimr/machine/interface/PhysicalRegisterOperand.h>
#include <vanaheimr/machine/interface/PhysicalIndirectOperand.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace codegen
{

ChaitinBriggsRegisterAllocatorPass::ChaitinBriggsRegisterAllocatorPass()
: RegisterAllocator({"InterferenceAnalysis"},
	"ChaitinBriggsRegisterAllocatorPass")
{

}

typedef analysis::InterferenceAnalysis InterferenceAnalysis;
typedef util::LargeMap<unsigned int, unsigned int> RegisterMap;

static void color(RegisterAllocator::VirtualRegisterSet& spilled,
	RegisterMap& allocated, const ir::Function& function,
	const InterferenceAnalysis& interferences, unsigned int colors);
static void assignRegisters(ir::Function& f,
	const ChaitinBriggsRegisterAllocatorPass& allocator);

void ChaitinBriggsRegisterAllocatorPass::runOnFunction(Function& f)
{
	report("Running chaitin-briggs graph coloring register allocator on "
		<< f.name());
	
	auto interferenceAnalysis = static_cast<InterferenceAnalysis*>(
		getAnalysis("InterferenceAnalysis"));
	assert(interferenceAnalysis != nullptr);
	
	_machine = compiler::Compiler::getSingleton()->getMachineModel();
	
	// attempt to color the interferences
	color(_spilled, _allocated, f, *interferenceAnalysis,
		_machine->totalRegisterCount());
	
	// TODO: spill if allocation fails
	assertM(_spilled.empty(), "No support for spills yet.");
	
	// TODO: Map colors to registers
	
	// Assign registers
	assignRegisters(f, *this);
}

transforms::Pass* ChaitinBriggsRegisterAllocatorPass::clone() const
{
	return new ChaitinBriggsRegisterAllocatorPass;
}

RegisterAllocator::VirtualRegisterSet
	ChaitinBriggsRegisterAllocatorPass::getSpilledRegisters()
{
	return _spilled;
}

const machine::PhysicalRegister*
	ChaitinBriggsRegisterAllocatorPass::getPhysicalRegister(
	const ir::VirtualRegister& vr) const
{
	auto allocatedRegister = _allocated.find(vr.id);
	
	if(allocatedRegister == _allocated.end()) return nullptr;
	
	return _machine->getPhysicalRegister(allocatedRegister->second);
}

class RegisterInfo
{
public:
	RegisterInfo(const ir::VirtualRegister* r, unsigned int d = 0,
		unsigned int c = 0, unsigned int s = 0, bool f = false)
	: virtualRegister(r), nodeDegree(d), color(c), schedulingOrder(s),
		finished(f)
	{
	
	}

public:
	const ir::VirtualRegister* virtualRegister;
	unsigned int               nodeDegree;
	unsigned int               color;
	unsigned int               schedulingOrder;
	bool                       finished;

};

typedef std::vector<RegisterInfo> RegisterInfoVector;
	
typedef util::SmallSet<unsigned int> ColorSet;
typedef std::vector<unsigned int> ColorVector;

static unsigned int randomColorThatDoesntCollide(ColorSet& usedColors,
	unsigned int maxColor)
{
	ColorSet unusedColors;
	
	for(unsigned int i = 0; i < maxColor; ++i)
	{
		if(usedColors.count(i) != 0) continue;
		
		unusedColors.insert(i);
	}
	
	ColorVector availableColors(unusedColors.begin(), unusedColors.end());
	
	return availableColors[std::rand() % availableColors.size()];
}

static unsigned int computeColor(bool& finished, const RegisterInfo& reg,
	const RegisterInfoVector& registerInfo,
	const InterferenceAnalysis& interferences)
{
	ColorSet usedColors;

	// Fix the color after the scheduling order window has passed
	if(reg.finished) return reg.color;
	
	auto regInterferences =
		interferences.getInterferences(*reg.virtualRegister);
	
	finished = true;
	
	unsigned int predecessorCount = 0;
	
	for(auto interference : regInterferences)
	{
		assert(interference->id < registerInfo.size());
	
		const RegisterInfo& info = registerInfo[interference->id];

		if(info.schedulingOrder > reg.schedulingOrder) continue;
	
		++predecessorCount;
	
		finished &= info.finished;
	
		usedColors.insert(info.color);
	}
	
	unsigned int spread = usedColors.size() + 1;
	
	// Define the range of possible colors [0 to usedRegisterCount]
	if(finished && (usedColors.count(reg.color) != 0 || reg.color >= spread))
	{
		return randomColorThatDoesntCollide(usedColors, spread);
	}
	
	// keep the original color if it doesn't collide
	if(usedColors.count(reg.color) == 0)
	{
		return reg.color;
	}
	
	// Otherwise, assign a new register randomly in the possible window
	unsigned int maxColor = predecessorCount + 1;
	
	return randomColorThatDoesntCollide(usedColors, maxColor);
}

static bool propagateColorsInParallel(RegisterInfoVector& registers,
	unsigned int iteration, const InterferenceAnalysis& interferences)
{
	report("  -------------------- Iteration "
		<< iteration << " ------------------");

	RegisterInfoVector newRegisters;
	
	newRegisters.reserve(registers.size());
	bool changed = false;
	
	for(auto reg = registers.begin(); reg != registers.end(); ++reg)
	{
		bool predecessorsFinished = true;
		unsigned int newColor = computeColor(predecessorsFinished, *reg,
			registers, interferences);

		newRegisters.push_back(RegisterInfo(reg->virtualRegister,
			reg->nodeDegree, newColor, reg->schedulingOrder,
			predecessorsFinished));

		changed |= reg->color != newColor;

		reportE(reg->color != newColor,
			"   vr" << reg->virtualRegister->id
			<< " (degree " << reg->nodeDegree
			<< ") | (color " << reg->color << ") -> (color " << newColor
			<< ")");
	}
	
	registers = std::move(newRegisters);
	
	return changed;
}

static void initializeColors(RegisterInfoVector& registers,
	const InterferenceAnalysis& interferences)
{
	// initialize the register randomly in the possible range
	for(auto& reg : registers)
	{
		auto regInterferences =
			interferences.getInterferences(*reg.virtualRegister);
	
		unsigned int predecessorCount = 0;
	
		for(auto interference : regInterferences)
		{
			assert(interference->id < registers.size());
	
			const RegisterInfo& info = registers[interference->id];

			if(info.schedulingOrder > reg.schedulingOrder) continue;
			
			++predecessorCount;
		}
		
		unsigned int maxColor = predecessorCount + 1;
	
		reg.color = std::rand() % maxColor;
	}
}

static void initializeSchedulingOrder(RegisterInfoVector& registerInfo)
{
	typedef std::pair<unsigned int, RegisterInfo*> DegreeAndInfoPair;
	typedef std::vector<DegreeAndInfoPair>         DegreeAndInfoVector;
	
	report(" Ranking registers by interference graph node degree");
	
	DegreeAndInfoVector degrees;
	
	degrees.reserve(registerInfo.size());
	
	for(auto info = registerInfo.begin(); info != registerInfo.end(); ++info)
	{
		degrees.push_back(std::make_pair(info->nodeDegree, &*info));
	}
	
	// Sort by node degree (parallel)
	std::sort(degrees.begin(), degrees.end(),
		std::greater<DegreeAndInfoPair>());

	for(auto degree = degrees.begin(); degree != degrees.end(); ++degree)
	{
		report("  vr" << degree->second->virtualRegister->id
			<< " (" << degree->first << ")");
		degree->second->schedulingOrder =
			std::distance(degrees.begin(), degree);
	}
}

static void color(RegisterAllocator::VirtualRegisterSet& spilled,
	RegisterMap& allocated, const ir::Function& function,
	const InterferenceAnalysis& interferences, unsigned int colors)
{
	//std::srand(std::time(0));
	std::srand(0);
	
	// Create a map from node degree to virtual register
	RegisterInfoVector registers;
	
	registers.reserve(function.register_size());
	
	for(auto reg = function.register_begin();
		reg != function.register_end(); ++reg)
	{
		registers.push_back(RegisterInfo(&*reg,
			interferences.getInterferences(*reg).size()));
	}
	
	// Initialize scheduling order for each node
	initializeSchedulingOrder(registers);
	
	// Initialize the colors
	initializeColors(registers, interferences);
	
	// Propagate colors until converged
	report(" Propating colors until converged.");
	
	unsigned int iteration = 0;
	bool changed = true;
	
	while(changed)
	{
		changed = propagateColorsInParallel(registers,
			iteration++, interferences);
		
		// Check iteration count
		assertM(iteration <= colors, "Too many iterations: " << iteration);
	}
	
	report("  -------------------- Iteration Count "
		<< iteration << " ------------------");
	
	// finish
	report("  Final report");
	unsigned int score = 0;
	
	for(auto& reg : registers)
	{
		allocated.insert(std::make_pair(reg.virtualRegister->id, reg.color));
		
		score += reg.color;
		
		report("   vr" << reg.virtualRegister->id
			<< " (degree " << reg.nodeDegree
			<< ") | (color " << reg.color << ")");
	}

	report("  allocation score " << score);
	
}

static void replaceVirtualRegisterWithPhysical(ir::Operand*& operand,
	const ChaitinBriggsRegisterAllocatorPass& allocator)
{
	if(!operand->isRegister()) return;

	auto newOperand = operand;
	
	if(operand->isIndirect())
	{
		auto indirectOperand = static_cast<ir::IndirectOperand*>(operand);
		
		newOperand = new machine::PhysicalIndirectOperand(
			allocator.getPhysicalRegister(*indirectOperand->virtualRegister),
			indirectOperand->virtualRegister, indirectOperand->offset,
			indirectOperand->instruction);
	}
	else
	{
		auto registerOperand = static_cast<ir::RegisterOperand*>(operand);
	
		newOperand = new machine::PhysicalRegisterOperand(
			allocator.getPhysicalRegister(*registerOperand->virtualRegister),
			registerOperand->virtualRegister, registerOperand->instruction);
	}

	delete operand;
	
	operand = newOperand;
}

static void assignRegisters(ir::Function& f,
	const ChaitinBriggsRegisterAllocatorPass& allocator)
{
	for(auto& block : f)
	{
		for(auto& instruction : block)
		{
			for(auto& read : instruction->reads)
			{
				replaceVirtualRegisterWithPhysical(read, allocator);
			}

			for(auto& write : instruction->writes)
			{
				replaceVirtualRegisterWithPhysical(write, allocator);
			}
		}
	}
}


}

}


