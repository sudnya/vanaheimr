/*! \file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream, const std::string& name)
{
	      _readHeader(stream);
	 _readDataSection(stream);
	 _readStringTable(stream);
	 _readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(name,
		compiler::Compiler::getSingleton());

	_loadTypes();
	_initializeModule(*module);
	
	report("Finished loading binary...");
	
	return module;
}

void BinaryReader::_readHeader(std::istream& stream)
{
	report("Reading header...");
	stream.read((char*)&_header, sizeof(BinaryHeader));

	if(stream.gcount() != sizeof(BinaryHeader))
	{
		throw std::runtime_error("Failed to read binary "
			"header, hit EOF.");
	}

	if(_header.magic != BinaryHeader::MagicNumber)
	{
		throw std::runtime_error("Failed to read binary "
			"header, invalid magic number.");
	}

	report(" data pages:    " << _header.dataPages);
	report(" code pages:    " << _header.codePages);
	report(" symbols:       " << _header.symbols);
	report(" string pages:  " << _header.stringPages);
	report(" data offset:   " << _header.dataOffset);
	report(" code offset:   " << _header.codeOffset);
	report(" symbol offset: " << _header.symbolOffset);
	report(" string offset: " << _header.stringsOffset);
	report(" name offset:   " << _header.nameOffset);
}

void BinaryReader::_readDataSection(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.dataPages;

	stream.seekg(_header.dataOffset, std::ios::beg);

	_dataSection.resize(dataSize);

	stream.read((char*) _dataSection.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read binary data section, hit"
			" EOF."); 
	}
}

void BinaryReader::_readStringTable(std::istream& stream)
{
	size_t stringTableSize = BinaryHeader::PageSize * _header.stringPages;

	stream.seekg(_header.stringsOffset, std::ios::beg);

	_stringTable.resize(stringTableSize);

	stream.read((char*) _stringTable.data(), stringTableSize);

	if((size_t)stream.gcount() != stringTableSize)
	{
		throw std::runtime_error("Failed to read string table, hit EOF");
	}
}

void BinaryReader::_readSymbolTable(std::istream& stream)
{
	size_t symbolTableSize = sizeof(SymbolTableEntry) * _header.symbols;

	stream.seekg(_header.symbolOffset, std::ios::beg);

	_symbolTable.resize(_header.symbols);

	stream.read((char*) _symbolTable.data(), symbolTableSize);

	if((size_t)stream.gcount() != symbolTableSize)
	{
		throw std::runtime_error("Failed to read symbol table, hit EOF");
	}
}

void BinaryReader::_readInstructions(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.codePages;
	size_t sizeInInstructions = (dataSize + sizeof(InstructionContainer) - 1) /
		sizeof(InstructionContainer);

	_instructions.resize(sizeInInstructions);

	// TODO obey page alignment
	stream.seekg(_header.codeOffset, std::ios::beg);

	stream.read((char*) _instructions.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read code section, hit EOF.");
	}
}

void BinaryReader::_loadTypes()
{
	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		compiler::Compiler::getSingleton()->getOrInsertType(
			_getSymbolTypeName(*symbol));
	}
}

void BinaryReader::_initializeModule(ir::Module& m)
{
	_loadGlobals(m);
	_loadFunctions(m);

	_variables.clear();
	_locals.clear();
}

void BinaryReader::_loadGlobals(ir::Module& m)
{
	report(" Loading global variables from symbol table...");
	
	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		if(symbol->type != SymbolTableEntry::VariableType &&
			symbol->type != SymbolTableEntry::FunctionType) continue;

		uint64_t symbolTableOffset = _header.symbolOffset +
			sizeof(SymbolTableEntry) *
			std::distance(_symbolTable.begin(), symbol);
		
		report("  loaded " << _getSymbolName(*symbol)
			<< " at offset " << symbol->offset
			<< ", symbol offset is " << symbolTableOffset);

		auto type = _getSymbolType(*symbol);

		if(type == nullptr)
		{
			throw std::runtime_error("Could not find type with name '" +
				_getSymbolTypeName(*symbol) + "' for symbol '" +
				_getSymbolName(*symbol) + "'");
		}

		ir::Variable* variable = nullptr;

		if(symbol->type == SymbolTableEntry::VariableType)
		{
			
			if(symbol->attributes.level == ir::Global::Shared)
			{
				auto global = m.newGlobal(_getSymbolName(*symbol), type,
					_getSymbolLinkage(*symbol), _getSymbolLevel(*symbol));

				if(_hasInitializer(*symbol))
				{
					global->setInitializer(_getInitializer(*symbol));
				}
			
				variable = &*global;
			}
			else if(symbol->attributes.level == ir::Global::Thread)
			{
				bool success = _locals.insert(
					std::make_pair(symbolTableOffset, nullptr)).second;
				
				if(!success)
				{
					throw std::runtime_error(
						"Thread-local variable with name '" +
						_getSymbolName(*symbol) + "' defined multiple times.");
				}
				
				// Don't update the variable set
				return;
			}
			else
			{
				assertM(false, "Variable scoping level not implemented.");
			}
		}
		else
		{
			ir::Module::iterator function = m.newFunction(
				_getSymbolName(*symbol), _getSymbolLinkage(*symbol),
				_getSymbolVisibility(*symbol), type);

			// Add attributes
			auto attributeList = _getSymbolAttributes(*symbol);
			
			auto attributes = hydrazine::split(attributeList, ", ");
			
			for(auto attribute : attributes)
			{
				report("   added attribute '" << attribute << "'");
				function->addAttribute(attribute);
			}

			variable = &*function;
		}
		
		_variables.insert(std::make_pair(symbolTableOffset, variable));
	}
}

void BinaryReader::_loadFunctions(ir::Module& m)
{
	typedef std::unordered_map<uint64_t, ir::BasicBlock*> PCToBasicBlockMap;

	report(" Loading functions from symbol table...");

	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		if(symbol->type != SymbolTableEntry::FunctionType) continue;

		report("  loaded function " << _getSymbolName(*symbol));

		uint64_t symbolTableOffset = _header.symbolOffset +
			sizeof(SymbolTableEntry) *
			std::distance(_symbolTable.begin(), symbol);
	
		ir::Variable* variable = _getVariableAtSymbolOffset(symbolTableOffset);
		ir::Function* function = static_cast<ir::Function*>(variable);
		
		_function = function;
		
		report("   loading arguments...");

		for(auto argumentSymbol = _symbolTable.begin();
			argumentSymbol != _symbolTable.end(); ++argumentSymbol)
		{
			if(argumentSymbol->type != SymbolTableEntry::ArgumentType)
			{
				continue;
			}
			
			std::string functionName =
				_getSymbolName(*argumentSymbol).substr(2,
				function->name().size());
		
			if(functionName != function->name()) continue;

			uint64_t symbolTableOffset = _header.symbolOffset +
				sizeof(SymbolTableEntry) *
				std::distance(_symbolTable.begin(), argumentSymbol);

			std::string name = _getSymbolName(*argumentSymbol).substr(
				2 + function->name().size());

			report("    loaded argument " << name
				<< " at offset " << argumentSymbol->offset
				<< ", symbol offset is " << symbolTableOffset);

			auto type = _getSymbolType(*argumentSymbol);

			if(type == nullptr)
			{
				throw std::runtime_error("Could not find type with name '" +
					_getSymbolTypeName(*argumentSymbol) + "' for symbol '" +
					name + "'");
			}
	
			auto argument = function->newArgument(type, name);

			_arguments.insert(std::make_pair(symbolTableOffset,
				&*argument));
		}

		BasicBlockDescriptorVector blocks = _getBasicBlocksInFunction(*symbol);
	
		PCToBasicBlockMap blockPCs;

		for(auto blockOffset : blocks)
		{
			ir::Function::iterator block = function->newBasicBlock(
				function->end(), blockOffset.name);

			blockPCs.insert(std::make_pair(blockOffset.begin, &*block));

			report("   adding basic block " << blockOffset.name
				<< " using instructions [" 
				<< blockOffset.begin << ", " << blockOffset.end << "]");
		
			for(unsigned int i = blockOffset.begin; i != blockOffset.end; ++i)
			{
				assert(i < _instructions.size());
				_addInstruction(block, _instructions[i]);
				report("    added instruction '" 
					<< block->back()->toString() << "'");
			}
		}

		report("  resolving branch targets...");

		for(auto unresolved : _unresolvedTargets)
		{
			// find the symbol with the specified offset
			const SymbolTableEntry& targetSymbol =
				_getSymbolEntryAtOffset(unresolved.first);

			uint64_t pc = (targetSymbol.offset - _header.codeOffset) /
				sizeof(InstructionContainer);
		
			report("   for branch to pc " << pc);

			auto block = blockPCs.find(pc);

			if(block == blockPCs.end())
			{
				std::stringstream message;

				message << "Could not find basic block starting at pc " << pc;

				throw std::runtime_error(message.str());
			}
			
			report("    setting target to " << block->second->name());

			static_cast<ir::AddressOperand*>(unresolved.second)->globalValue =
				block->second;
		}

		_unresolvedTargets.clear();
		_virtualRegisters.clear();
		_arguments.clear();
		
		for(auto local = _locals.begin(); local != _locals.end(); ++local)
		{
			local->second = nullptr;
		}
	}
}

std::string BinaryReader::_getSymbolName(const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.stringOffset);
}

std::string BinaryReader::_getSymbolTypeName(
	const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.typeOffset);
}

std::string BinaryReader::_getSymbolAttributes(
	const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.attributeOffset);
}

ir::Type* BinaryReader::_getSymbolType(const SymbolTableEntry& symbol) const
{
	return compiler::Compiler::getSingleton()->getType(
		_getSymbolTypeName(symbol));
}

ir::Variable::Linkage BinaryReader::_getSymbolLinkage(
	const SymbolTableEntry& symbol) const
{
	return (ir::Variable::Linkage)(symbol.attributes.linkage);
}

ir::Variable::Visibility BinaryReader::_getSymbolVisibility(
	const SymbolTableEntry& symbol) const
{
	return (ir::Variable::Visibility)(symbol.attributes.visibility);
}

ir::Global::Level BinaryReader::_getSymbolLevel(
	const SymbolTableEntry& symbol) const
{
	return (ir::Global::Level)symbol.attributes.level;
}

bool BinaryReader::_hasInitializer(const SymbolTableEntry& symbol) const
{
	// currently binaries never have initializers
	return false;
}

ir::Constant* BinaryReader::_getInitializer(
	const SymbolTableEntry& symbol) const
{
	assertM(false, "Not imeplemented.");
}

BinaryReader::BasicBlockDescriptorVector
	BinaryReader::_getBasicBlocksInFunction(
	const SymbolTableEntry& symbol) const
{
	typedef std::unordered_set<uint64_t> TargetSet;

	BasicBlockDescriptorVector blocks;
	
	// Get the first and last instruction in the function
	uint64_t begin = (symbol.offset - _header.codeOffset) /
		sizeof(InstructionContainer);
	
	uint64_t end = begin + symbol.size / sizeof(InstructionContainer);

	report("   getting basic block for symbol '" << _getSymbolName(symbol)
		<< "' (offset " << symbol.offset
		<< ", size " << symbol.size << ", range ["
		<< begin << ", " << end <<"])");

	TargetSet targets;

	for(uint64_t i = begin; i != end; ++i)
	{
		const InstructionContainer& instruction = _instructions[i];

		if(instruction.asInstruction.opcode == Instruction::Bra)
		{
			report("   found branch at pc " << i);
			auto operand = instruction.asBra.target;

			if(operand.asOperand.mode == Operand::Immediate)
			{
				report("    to target " << operand.asImmediate.uint);

				targets.insert(operand.asImmediate.uint);
			}
			else if(operand.asOperand.mode == Operand::Symbol)
			{
				const SymbolTableEntry& targetSymbol = 
					_getSymbolEntryAtOffset(operand.asSymbol.symbolTableOffset);
				
				uint64_t targetPC = (targetSymbol.offset - _header.codeOffset) /
					sizeof(InstructionContainer);
				
				report("    to target " << targetPC);

				targets.insert(targetPC);
			}
			else
			{
				assertM(false, "branch mode "
					<< operand.asOperand.mode << " not implemented");
			}
		}
	}

	BasicBlockDescriptor block("BB_0", begin);

	for(uint64_t i = begin; i != end; ++i)
	{
		bool isTerminator = false;
		uint64_t blockEnd = i;

		const InstructionContainer&
			instruction = _instructions[i];

		if(targets.count(i) != 0)
		{
			isTerminator = true;
		}
		else if(instruction.asInstruction.opcode ==	Instruction::Bra)
		{
			isTerminator = true;
			blockEnd = i + 1;
		}

		if(isTerminator && blockEnd != block.begin)
		{
			block.end = blockEnd;
			blocks.push_back(block);

			std::stringstream name;

			name << "BB_" << blocks.size();

			block = BasicBlockDescriptor(name.str(), blockEnd);
		}
	}

	if(block.begin != end)
	{
		block.end = end;
		blocks.push_back(block);
	}

	return blocks;
}

void BinaryReader::_addInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	if(_addSimpleBinaryInstruction(block, container)) return;
	if(_addSimpleUnaryInstruction(block, container))  return;
	if(_addComplexInstruction(block, container))      return;

	assertM(false, "Translation for instruction '" <<
		ir::Instruction::toString((ir::Instruction::Opcode)
		container.asInstruction.opcode) << "' not implemented.");
}

bool BinaryReader::_addSimpleBinaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{	
	switch(container.asInstruction.opcode)
	{
	case Instruction::Add:
	case Instruction::And:
	case Instruction::Ashr:
	case Instruction::Fdiv:
	case Instruction::Fmul:
	case Instruction::Frem:
	case Instruction::Lshr:
	case Instruction::Mul:
	case Instruction::Or:
	case Instruction::Sdiv:
	case Instruction::Shl:
	case Instruction::Srem:
	case Instruction::Sub:
	case Instruction::Udiv:
	case Instruction::Urem:
	case Instruction::Xor:
	{
		auto instruction = static_cast<ir::BinaryInstruction*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asBinaryInstruction.guard, instruction));

		instruction->setD(_translateOperand(container.asBinaryInstruction.d,
			instruction));
		instruction->setA(_translateOperand(container.asBinaryInstruction.a,
			instruction));
		instruction->setB(_translateOperand(container.asBinaryInstruction.b,
			instruction));

		block->push_back(instruction);
	
		return true;
	}
	default: break;
	}

	return false;
}

bool BinaryReader::_addSimpleUnaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	switch(container.asInstruction.opcode)
	{
	case Instruction::Bitcast:
	case Instruction::Fpext:
	case Instruction::Fptosi:
	case Instruction::Fptoui:
	case Instruction::Fptrunc:
	case Instruction::Ld:
	case Instruction::Sext:
	case Instruction::Sitofp:
	case Instruction::Trunc:
	case Instruction::Uitofp:
	case Instruction::Zext:
	{
		auto instruction = static_cast<ir::UnaryInstruction*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asUnaryInstruction.guard, instruction));

		instruction->setD(_translateOperand(container.asUnaryInstruction.d,
			instruction));
		instruction->setA(_translateOperand(container.asUnaryInstruction.a,
			instruction));

		block->push_back(instruction);
		
		return true;
	}
	default: break;
	}

	return false;
}

bool BinaryReader::_addComplexInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	if(container.asInstruction.opcode == Instruction::St)
	{
		auto instruction = static_cast<ir::St*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asSt.guard, instruction));

		instruction->setD(_translateOperand(container.asSt.d,
			instruction));
		instruction->setA(_translateOperand(container.asSt.a,
			instruction));

		block->push_back(instruction);
		
		return true;
		
	}
	else if(container.asInstruction.opcode == Instruction::Setp)
	{
		auto instruction = static_cast<ir::Setp*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asSetp.guard, instruction));

		instruction->setD(_translateOperand(container.asSetp.d,
			instruction));
		instruction->setA(_translateOperand(container.asSetp.a,
			instruction));
		instruction->setB(_translateOperand(container.asSetp.b,
			instruction));

		instruction->comparison =
			(ir::ComparisonInstruction::Comparison)
			container.asSetp.comparison;

		block->push_back(instruction);
		
		return true;
		
	}
	else if(container.asInstruction.opcode == Instruction::Bra)
	{
		auto instruction = static_cast<ir::Bra*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asBra.guard, instruction));

		instruction->setTarget(_translateOperand(container.asBra.target,
			instruction));
		
		instruction->modifier = (ir::Bra::BranchModifier)
			container.asBra.modifier;

		block->push_back(instruction);
		
		return true;
	}
	else if(container.asInstruction.opcode == Instruction::Call)
	{
		_addCallInstruction(block, container);

		return true;
	}
	else if(container.asInstruction.opcode == Instruction::Ret)
	{
		auto instruction = static_cast<ir::Ret*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asRet.guard, instruction));

		block->push_back(instruction);
		
		return true;
	}
	else if(container.asInstruction.opcode == Instruction::Phi)
	{
		_addPhiInstruction(block, container);

		return true;
	}
	
	return false;
}

void BinaryReader::_addCallInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	auto instruction = static_cast<ir::Call*>(
		ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

	instruction->setGuard(_translateOperand(
		container.asCall.guard, instruction));

	instruction->setTarget(_translateOperand(container.asCall.target,
		instruction));
	
	for(unsigned int returned = 0;
		returned != container.asCall.returnArguments; ++returned)
	{
		uint64_t offset = returned * sizeof(OperandContainer) +
			container.asCall.returnArgumentOffset;
		const OperandContainer* operand =
			reinterpret_cast<OperandContainer*>(&_dataSection[offset]);
	
		instruction->addReturn(_translateOperand(*operand, instruction));
	}

	for(unsigned int argument = 0;
		argument != container.asCall.arguments; ++argument)
	{
		uint64_t offset = argument * sizeof(OperandContainer) +
			container.asCall.argumentOffset;
		const OperandContainer* operand =
			reinterpret_cast<OperandContainer*>(&_dataSection[offset]);
	
		instruction->addArgument(_translateOperand(*operand, instruction));
	}
	
	block->push_back(instruction);
}

void BinaryReader::_addPhiInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	auto instruction = static_cast<ir::Phi*>(
		ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

	instruction->setGuard(_translateOperand(
		container.asCall.guard, instruction));

	auto destinationOperand = _translateOperand(container.asPhi.destination,
		instruction);

	instruction->setD(static_cast<ir::RegisterOperand*>(destinationOperand));
	
	for(unsigned int source = 0, sourceBlock = container.asPhi.sources;
		source != container.asPhi.sources; ++source, ++sourceBlock)
	{
		uint64_t offset = source * sizeof(OperandContainer) +
			container.asPhi.sourcesOffset;
		uint64_t blockOffset = sourceBlock * sizeof(OperandContainer) +
			container.asPhi.sourcesOffset;
		
		const OperandContainer* operandSource =
			reinterpret_cast<OperandContainer*>(&_dataSection[offset]);
		const OperandContainer* operandBlock =
			reinterpret_cast<OperandContainer*>(&_dataSection[blockOffset]);
			
		auto registerSource = static_cast<ir::RegisterOperand*>(
			_translateOperand(*operandSource, instruction));
		auto addressBlock   = static_cast<ir::AddressOperand*>(
			_translateOperand(*operandBlock, instruction));
			
		instruction->addSource(registerSource, addressBlock);
	}
	
	block->push_back(instruction);
}

ir::Operand* BinaryReader::_translateOperand(const OperandContainer& container,
	ir::Instruction* instruction)
{
	typedef Operand Operand;

	switch(container.asOperand.mode)
	{
	case Operand::Predicate:
	{
		return _translateOperand(container.asPredicate, instruction);
	}
	case Operand::Register:
	{
		ir::RegisterOperand* operand = new ir::RegisterOperand(
			_getVirtualRegister(container.asRegister.reg,
			container.asRegister.type, instruction->block->function()),
			instruction);
		
		return operand;
	}
	case Operand::Immediate:
	{
		ir::ImmediateOperand* operand = new ir::ImmediateOperand(
			container.asImmediate.uint, instruction,
			_getType(container.asImmediate.type));

		return operand;
	}
	case Operand::Indirect:
	{
		ir::IndirectOperand* operand = new ir::IndirectOperand(
			_getVirtualRegister(container.asIndirect.reg,
			container.asIndirect.type, instruction->block->function()),
			container.asIndirect.offset, instruction);
		
		return operand;
	}
	case Operand::Symbol:
	{
		ir::Argument* argument = _getArgumentAtSymbolOffset(
			container.asSymbol.symbolTableOffset);

		if(argument != nullptr)
		{
			ir::ArgumentOperand* operand = new ir::ArgumentOperand(argument,
				instruction);
			
			return operand;
		}

		ir::Variable* variable = _getVariableAtSymbolOffset(
			container.asSymbol.symbolTableOffset);

		ir::AddressOperand* operand = new ir::AddressOperand(
			variable, instruction);
	
		if(variable == nullptr)
		{
			report("  adding unresolved basic block for symbol at offset "
				<< container.asSymbol.symbolTableOffset);
			_unresolvedTargets.insert(std::make_pair(
				container.asSymbol.symbolTableOffset, operand));
		}

		return operand;
	}
	case Operand::InvalidOperand: break;
	}	

	assertM(false, "Invalid operand type.");

	return 0;
}

ir::PredicateOperand* BinaryReader::_translateOperand(
	const PredicateOperand& operand, ir::Instruction* instruction)
{
	ir::VirtualRegister* virtualRegister = nullptr;

	if(operand.modifier != PredicateOperand::PredicateTrue &&
		operand.modifier != PredicateOperand::PredicateFalse)
	{
		virtualRegister = _getVirtualRegister(operand.reg,
			i1, instruction->block->function());
	}

	return new ir::PredicateOperand(virtualRegister, 
		(ir::PredicateOperand::PredicateModifier)operand.modifier,
		instruction);
}

const ir::Type* BinaryReader::_getType(DataType type) const
{
	switch(type)
	{
	case i1:
	{
		return compiler::Compiler::getSingleton()->getType("i1");
	}
	case i8:
	{
		return compiler::Compiler::getSingleton()->getType("i8");
	}
	case i16:
	{
		return compiler::Compiler::getSingleton()->getType("i16");
	}
	case i32:
	{
		return compiler::Compiler::getSingleton()->getType("i32");
	}
	case i64:
	{
		return compiler::Compiler::getSingleton()->getType("i64");
	}
	case f32:
	{
		return compiler::Compiler::getSingleton()->getType("f32");
	}
	case f64:
	{
		return compiler::Compiler::getSingleton()->getType("f64");
	}
	default: break;
	}

	assertM(false, "Invalid data type.");

	return 0;
}
	
ir::VirtualRegister* BinaryReader::_getVirtualRegister(
	RegisterType reg,
	DataType type, ir::Function* function)
{
	auto virtualRegister = _virtualRegisters.find(reg);

	if(_virtualRegisters.end() == virtualRegister)
	{
		std::stringstream name;

		name << "r" << reg;

		auto insertedRegister = function->newVirtualRegister(
			_getType(type), name.str());
	
		virtualRegister = _virtualRegisters.insert(std::make_pair(reg,
			&*insertedRegister)).first;
	}

	return virtualRegister->second;
}

ir::Variable* BinaryReader::_getVariableAtSymbolOffset(uint64_t offset)
{
	auto local = _locals.find(offset);
	
	if(local != _locals.end())
	{
		if(local->second == nullptr)
		{
			auto symbol = _getSymbolEntryAtOffset(local->first);
			
			auto type = _getSymbolType(symbol);

			if(type == nullptr)
			{
				throw std::runtime_error("Could not find type with name '" +
					_getSymbolTypeName(symbol) + "' for symbol '" +
					_getSymbolName(symbol) + "'");
			}
		
			auto newLocal = _function->newLocalValue(_getSymbolName(symbol),
				type, _getSymbolLinkage(symbol), _getSymbolLevel(symbol));
			
			local->second = &*newLocal;
		}
	
		return local->second;
	}
	
	auto variable = _variables.find(offset);

	if(variable == _variables.end())
	{
		return nullptr;
	}

	return variable->second;
}

ir::Argument* BinaryReader::_getArgumentAtSymbolOffset(uint64_t offset) const
{
	auto argument = _arguments.find(offset);

	if(argument == _arguments.end())
	{
		return nullptr;
	}

	return argument->second;
}

const SymbolTableEntry& BinaryReader::_getSymbolEntryAtOffset(
	uint64_t offset) const
{
	uint64_t symbolOffset =
		(offset - _header.symbolOffset) / sizeof(SymbolTableEntry);
	
	assertM(symbolOffset < _symbolTable.size(), "Invalid symbol "
		<< symbolOffset << " out of " << _symbolTable.size());
	
	return _symbolTable[symbolOffset];
}

BinaryReader::BasicBlockDescriptor::BasicBlockDescriptor(
	const std::string& n, uint64_t b, uint64_t e)
: name(n), begin(b), end(e)
{

}

}

}

