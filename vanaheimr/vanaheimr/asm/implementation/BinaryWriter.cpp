/*!	\file   BinaryWriter.cpp
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The implementation file for the helper class that traslates
	        compiler IR to a binary.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryWriter.h>

#include <vanaheimr/ir/interface/Module.h>
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{

/*! \brief A namespace for the internal representation */
namespace as
{

BinaryWriter::BinaryWriter()
: m_module(0)
{

}

void BinaryWriter::write(std::ostream& binary, const ir::Module& m)
{
	m_module = &m;

	report("Serializing module " << m.name << " to binary bytecode...");

	populateData();
	populateInstructions();
	linkSymbols();
	
	populateHeader();

	report(" writing header");
	writePage(binary, (const char*)&m_header, sizeof(BinaryHeader));
	report(" writing symbols");
	writePage(binary, (const char*)m_symbolTable.data(), getSymbolTableSize());
	report(" writing instructions");
	writePage(binary, (const char*)m_instructions.data(),
		getInstructionStreamSize());
	report(" writing data");
	writePage(binary, (const char*)m_data.data(), getDataSize());
	report(" writing string table");
	writePage(binary, (const char*)m_stringTable.data(), getStringTableSize());
}

void BinaryWriter::writePage(std::ostream& binary, const void* data,
	uint64_t size)
{
	uint64_t currentPosition = pageAlign(binary.tellp());
	
	report("  page aligning " << binary.tellp() << " to " << currentPosition 
		<< " before writing.");

	while((uint64_t)binary.tellp() != currentPosition)
	{
		char temp = 0;
		binary.write(&temp, 1);
	}

	binary.write((const char*)data, size);
	
	currentPosition = pageAlign(binary.tellp());
	
	report("  page aligning " << binary.tellp() << " to " << currentPosition 
		<< " after writing.");
	
	while((uint64_t)binary.tellp() != currentPosition)
	{
		char temp = 0;
		binary.write(&temp, 1);
	}
}

void BinaryWriter::populateData()
{
	report(" Adding global variables...");

	for(ir::Module::const_global_iterator i = m_module->global_begin();
		i != m_module->global_end(); ++i)
	{
		addGlobal(*i);
	}
}

static std::string flattenAttributes(const ir::Function& function)
{
	auto attributes = function.attributes();
	
	std::stringstream list;
	
	for(auto attribute = attributes.begin();
		attribute != attributes.end(); ++attribute)
	{
		if(attribute != attributes.begin()) list << ", ";
		
		list << *attribute;
	}
	
	return list.str();
}

void BinaryWriter::populateInstructions()
{
	report(" Adding function symbols.");
	for(ir::Module::const_iterator function = m_module->begin();
		function != m_module->end(); ++function)
	{
		report("  " << function->name());
		addSymbol(SymbolTableEntry::FunctionType, function->linkage(),
			function->visibility(), ir::Global::InvalidLevel, function->name(),
			0, 0, function->type().name, flattenAttributes(*function));
	}
	
	report(" Adding functions.");
	for(ir::Module::const_iterator function = m_module->begin();
		function != m_module->end(); ++function)
	{
		report("  " << function->name());
	
		// Arguments
		for(auto argument = function->argument_begin();
			argument != function->argument_end(); ++argument)
		{
			addSymbol(SymbolTableEntry::ArgumentType, 0x0, 0x0,
				ir::Global::InvalidLevel, argument->mangledName(),
				m_data.size(), 0x0, argument->type().name);
			m_data.resize(m_data.size() + argument->type().bytes());
		}

		// Locals
		for(auto local = function->local_begin();
			local != function->local_end(); ++local)
		{
			addGlobal(*local);
		}
		
		// Instructions
		unsigned int instructionsBegin =
			m_instructions.size() * sizeof(InstructionContainer);
		unsigned int instructionOffset = m_instructions.size();	
		for(auto bb = function->begin(); bb != function->end(); ++bb)
		{
			m_basicBlockOffsets.insert(std::make_pair(bb->name(),
				instructionOffset * sizeof(InstructionContainer)));

			instructionOffset += bb->size();
		}

		unsigned int instructionsSize =
			instructionOffset * sizeof(InstructionContainer)
			- instructionsBegin;
	
		for(auto bb = function->begin(); bb != function->end(); ++bb)
		{
			report("   Basic Block " << bb->name());
			for(auto inst = bb->begin(); inst != bb->end(); ++inst)
			{
				m_instructions.push_back(convertToContainer(**inst));
			}
		}

		patchSymbol(function->name(), instructionsBegin, instructionsSize);

		m_basicBlockOffsets.clear();
		m_basicBlockSymbols.clear();
	}
}

void BinaryWriter::linkSymbols()
{
	for (symbol_iterator symb = m_symbolTable.begin();
		symb != m_symbolTable.end(); ++symb)
	{
		if(symb->type == SymbolTableEntry::FunctionType
			|| symb->type == SymbolTableEntry::BasicBlockType)
		{
			symb->offset += getInstructionOffset();
		}
		else if(symb->type == SymbolTableEntry::VariableType ||
			symb->type == SymbolTableEntry::ArgumentType)
		{
			symb->offset += getDataOffset();
		}
	}
}

void BinaryWriter::populateHeader()
{
	m_header.magic         = BinaryHeader::MagicNumber;
	m_header.dataPages     = (m_data.size() + PageSize - 1) / PageSize; 
	m_header.codePages     =
		((m_instructions.size() * sizeof(InstructionContainer)) + PageSize - 1)
		/ PageSize;
	m_header.symbols       = m_symbolTable.size(); 
	m_header.stringPages   = (m_stringTable.size() + PageSize - 1) / PageSize;
	m_header.dataOffset    = getDataOffset();
	m_header.codeOffset    = getInstructionOffset();
	m_header.symbolOffset  = getSymbolTableOffset();
	m_header.stringsOffset = getStringTableOffset();
}

size_t BinaryWriter::getHeaderOffset() const
{
	return pageAlign(0);
}

size_t BinaryWriter::getInstructionOffset() const
{
	return pageAlign(getSymbolTableSize() + getSymbolTableOffset());
}

size_t BinaryWriter::getDataOffset() const
{
	return pageAlign(getInstructionStreamSize() + getInstructionOffset());
}

size_t BinaryWriter::getSymbolTableOffset() const
{
	return pageAlign(sizeof(m_header));
}

size_t BinaryWriter::getStringTableOffset() const
{
	return pageAlign(getDataSize() + getDataOffset());
}

size_t BinaryWriter::getSymbolTableSize() const
{
	return m_symbolTable.size() * sizeof(SymbolTableEntry);
}

size_t BinaryWriter::getInstructionStreamSize() const
{
	return m_instructions.size() * sizeof(InstructionContainer);
}

size_t BinaryWriter::getDataSize() const
{
	return m_data.size();
}

size_t BinaryWriter::getStringTableSize() const
{
	return m_stringTable.size();
}

static Instruction::Opcode convertOpcode(
	ir::Instruction::Opcode opcode)
{
	typedef Instruction AInstruction;
	
	switch(opcode)
	{
	case ir::Instruction::Add:           return AInstruction::Add;
	case ir::Instruction::And:           return AInstruction::And;
	case ir::Instruction::Ashr:          return AInstruction::Ashr;
	case ir::Instruction::Atom:          return AInstruction::Atom;
	case ir::Instruction::Bar:           return AInstruction::Bar;
	case ir::Instruction::Bitcast:       return AInstruction::Bitcast;
	case ir::Instruction::Bra:           return AInstruction::Bra;
	case ir::Instruction::Call:          return AInstruction::Call;
	case ir::Instruction::Fdiv:          return AInstruction::Fdiv;
	case ir::Instruction::Fmul:          return AInstruction::Fmul;
	case ir::Instruction::Fpext:         return AInstruction::Fpext;
	case ir::Instruction::Fptosi:        return AInstruction::Fptosi;
	case ir::Instruction::Fptoui:        return AInstruction::Fptoui;
	case ir::Instruction::Fptrunc:       return AInstruction::Fptrunc;
	case ir::Instruction::Frem:          return AInstruction::Frem;
	case ir::Instruction::Launch:        return AInstruction::Launch;
	case ir::Instruction::Ld:            return AInstruction::Ld;
	case ir::Instruction::Lshr:          return AInstruction::Lshr;
	case ir::Instruction::Membar:        return AInstruction::Membar;
	case ir::Instruction::Mul:           return AInstruction::Mul;
	case ir::Instruction::Phi:           return AInstruction::Phi;
	case ir::Instruction::Psi:           return AInstruction::Psi;
	case ir::Instruction::Or:            return AInstruction::Or;
	case ir::Instruction::Ret:           return AInstruction::Ret;
	case ir::Instruction::Setp:          return AInstruction::Setp;
	case ir::Instruction::Sext:          return AInstruction::Sext;
	case ir::Instruction::Sdiv:          return AInstruction::Sdiv;
	case ir::Instruction::Shl:           return AInstruction::Shl;
	case ir::Instruction::Sitofp:        return AInstruction::Sitofp;
	case ir::Instruction::Srem:          return AInstruction::Srem;
	case ir::Instruction::St:            return AInstruction::St;
	case ir::Instruction::Sub:           return AInstruction::Sub;
	case ir::Instruction::Trunc:         return AInstruction::Trunc;
	case ir::Instruction::Udiv:          return AInstruction::Udiv;
	case ir::Instruction::Uitofp:        return AInstruction::Uitofp;
	case ir::Instruction::Urem:          return AInstruction::Urem;
	case ir::Instruction::Xor:           return AInstruction::Xor;
	case ir::Instruction::Zext:          return AInstruction::Zext;
	case ir::Instruction::InvalidOpcode: return AInstruction::InvalidOpcode;
	default: assertM(false, "Invalid opcode.");
	}

	return AInstruction::InvalidOpcode;	
}

static DataType convertType(const ir::Type* type)
{
	if(type->isInteger())
	{
		const ir::IntegerType* integer = static_cast<const ir::IntegerType*>(type);

		switch(integer->bits())
		{
		case 1:
		{
			return i1;
		}
		case 8:
		{
			return i8;
		}
		case 16:
		{
			return i16;
		}
		case 32:
		{
			return i32;
		}
		case 64:
		{
			return i64;
		}
		default: assertM(false, "Invalid integer bit width.");
		}
	}
	else if(type->isFloatingPoint())
	{
		if(type->isSinglePrecisionFloat())
		{
			return f32;
		}
		else
		{
			return f64;
		}
	}

	assertM(false, "Data type conversion not implemented in binary writer");

	return InvalidDataType;
}

static PredicateOperand::PredicateModifier convertPredicate(
	ir::PredicateOperand::PredicateModifier modifier)
{
	switch(modifier)
	{
	case ir::PredicateOperand::StraightPredicate:
	{
		return PredicateOperand::StraightPredicate;
	}
	case ir::PredicateOperand::InversePredicate:
	{
		return PredicateOperand::InversePredicate;
	}
	case ir::PredicateOperand::PredicateTrue:
	{
		return PredicateOperand::PredicateTrue;
	}
	case ir::PredicateOperand::PredicateFalse:
	{
		return PredicateOperand::PredicateFalse;
	}
	default: break;
	}

	assertM(false, "Invalid predicate.");

	return PredicateOperand::InvalidPredicate;
}

OperandContainer BinaryWriter::convertOperand(
	const ir::Operand& operand)
{
	OperandContainer result;

	switch(operand.mode())
	{
	case ir::Operand::Register:
	{
		const ir::RegisterOperand& reg =
			static_cast<const ir::RegisterOperand&>(operand);

		report("     converting virtual register " << reg.virtualRegister->id
			<< " (" << reg.virtualRegister->type->name << ")");
		
		result.asRegister.reg  = reg.virtualRegister->id;
		result.asRegister.type = convertType(reg.virtualRegister->type);
		
		result.asOperand.mode = as::Operand::Register;
		break;
	}
	case ir::Operand::Immediate:
	{
		const ir::ImmediateOperand& immediate =
			static_cast<const ir::ImmediateOperand&>(operand);
		
		result.asImmediate.type = convertType(immediate.type());
		result.asImmediate.uint = immediate.uint;
		
		result.asOperand.mode = as::Operand::Immediate;
		break;
	}
	case ir::Operand::Predicate:
	{
		const ir::PredicateOperand& predicate =
			static_cast<const ir::PredicateOperand&>(operand);
		
		result.asPredicate.modifier = convertPredicate(predicate.modifier);
		
		if(predicate.modifier == ir::PredicateOperand::StraightPredicate ||
			predicate.modifier == ir::PredicateOperand::InversePredicate)
		{
			report("     converting non-trivial predicate with virtual "
				"register " << predicate.virtualRegister->id
				<< " (" << predicate.virtualRegister->type->name << ")");
			result.asPredicate.reg = predicate.virtualRegister->id;
		}
		else
		{
			result.asPredicate.reg = 0;
		}
		
		result.asOperand.mode = as::Operand::Predicate;

		break;
	}
	case ir::Operand::Indirect:
	{
		const ir::IndirectOperand& indirect =
			static_cast<const ir::IndirectOperand&>(operand);

		result.asIndirect.reg    = indirect.virtualRegister->id;
		result.asIndirect.type   = convertType(indirect.virtualRegister->type);
		result.asIndirect.offset = indirect.offset;
		
		result.asOperand.mode = as::Operand::Indirect;

		break;
	}
	case ir::Operand::Address:
	{
		const ir::AddressOperand& address =
			static_cast<const ir::AddressOperand&>(operand);
		
		result.asOperand.mode = as::Operand::Symbol;

		if(address.globalValue->type().isBasicBlock())
		{
			result.asSymbol.symbolTableOffset =
				getBasicBlockSymbolTableOffset(address.globalValue);
		}
		else
		{
			result.asSymbol.symbolTableOffset =
				getSymbolTableOffset(address.globalValue);
		}

		break;
	}
	case ir::Operand::Argument:	
	{
		const ir::ArgumentOperand& argument =
			static_cast<const ir::ArgumentOperand&>(operand);

		result.asSymbol.symbolTableOffset =
			getSymbolTableOffset(argument.argument);
		
		result.asOperand.mode = as::Operand::Symbol;

		break;
	}
	}

	return result;
}

static bool isComplexInstruction(const ir::Instruction& instruction)
{
	switch(instruction.opcode)
	{
	case ir::Instruction::St:   // fall through
	case ir::Instruction::Bra:  // fall through
	case ir::Instruction::Ret:  // fall through
	case ir::Instruction::Call: // fall through
	case ir::Instruction::Phi: // fall through
	{
		return true;
	}
	default: break;;
	}
	
	return false;
}

void BinaryWriter::convertComplexInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	switch(instruction.opcode)
	{
	case ir::Instruction::Bra:
	{
		convertBraInstruction(container, instruction);
		break;
	}
	case ir::Instruction::Call:
	{
		convertCallInstruction(container, instruction);
		break;
	}
	case ir::Instruction::St:
	{
		convertStInstruction(container, instruction);
		break;
	}
	case ir::Instruction::Ret:
	{
		convertRetInstruction(container, instruction);
		break;
	}
	case ir::Instruction::Phi:
	{
		convertPhiInstruction(container, instruction);
		break;
	}
	default: assertM(false, "Translation for "
		<< instruction.toString() << " not implemented.");
	}
}

void BinaryWriter::convertUnaryInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::UnaryInstruction& unary =
		static_cast<const ir::UnaryInstruction&>(instruction);

	container.asUnaryInstruction.d = convertOperand(*unary.d());
	container.asUnaryInstruction.a = convertOperand(*unary.a());
}

void BinaryWriter::convertBinaryInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::BinaryInstruction& binary =
		static_cast<const ir::BinaryInstruction&>(instruction);

	container.asBinaryInstruction.d = convertOperand(*binary.d());
	container.asBinaryInstruction.a = convertOperand(*binary.a());
	container.asBinaryInstruction.b = convertOperand(*binary.b());
}

static ComparisonInstruction::Comparison convertComparison(
	ir::ComparisonInstruction::Comparison comparison)
{
	switch(comparison)
	{
	case ir::ComparisonInstruction::OrderedEqual:
	{
		return ComparisonInstruction::OrderedEqual;
	}
	case ir::ComparisonInstruction::OrderedNotEqual:
	{
		return ComparisonInstruction::OrderedNotEqual;
	}
	case ir::ComparisonInstruction::OrderedLessThan:
	{
		return ComparisonInstruction::OrderedLessThan;
	}
	case ir::ComparisonInstruction::OrderedLessOrEqual:
	{
		return ComparisonInstruction::OrderedLessOrEqual;
	}
	case ir::ComparisonInstruction::OrderedGreaterThan:
	{
		return ComparisonInstruction::OrderedGreaterThan;
	}
	case ir::ComparisonInstruction::OrderedGreaterOrEqual:
	{
		return ComparisonInstruction::OrderedGreaterOrEqual;
	}
	case ir::ComparisonInstruction::UnorderedEqual:
	{
		return ComparisonInstruction::UnorderedEqual;
	}
	case ir::ComparisonInstruction::UnorderedNotEqual:
	{
		return ComparisonInstruction::UnorderedNotEqual;
	}
	case ir::ComparisonInstruction::UnorderedLessThan:
	{
		return ComparisonInstruction::UnorderedLessThan;
	}
	case ir::ComparisonInstruction::UnorderedLessOrEqual:
	{
		return ComparisonInstruction::UnorderedLessOrEqual;
	}
	case ir::ComparisonInstruction::UnorderedGreaterThan:
	{
		return ComparisonInstruction::UnorderedGreaterThan;
	}
	case ir::ComparisonInstruction::UnorderedGreaterOrEqual:
	{
		return ComparisonInstruction::UnorderedGreaterOrEqual;
	}
	case ir::ComparisonInstruction::IsANumber:
	{
		return ComparisonInstruction::IsANumber;
	}
	case ir::ComparisonInstruction::NotANumber:
	{
		return ComparisonInstruction::NotANumber;
	}
	case ir::ComparisonInstruction::InvalidComparison:
	{
		return ComparisonInstruction::InvalidComparison;
	}
	}
	
	return ComparisonInstruction::InvalidComparison;
}

void BinaryWriter::convertComparisonInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::ComparisonInstruction& comparison =
		static_cast<const ir::ComparisonInstruction&>(instruction);

	container.asComparisonInstruction.d = convertOperand(*comparison.d());
	container.asComparisonInstruction.a = convertOperand(*comparison.a());
	container.asComparisonInstruction.b = convertOperand(*comparison.b());
	
	container.asComparisonInstruction.comparison =
		convertComparison(comparison.comparison);
}

InstructionContainer BinaryWriter::convertToContainer(
	const Instruction& instruction)
{
	report("    " << instruction.toString());

	InstructionContainer container;

	container.asInstruction.opcode = convertOpcode(instruction.opcode);
	container.asInstruction.guard  =
		convertOperand(*instruction.guard()).asPredicate;

	if(isComplexInstruction(instruction))
	{
		convertComplexInstruction(container, instruction);
	}
	else if(instruction.isComparison())
	{
		convertComparisonInstruction(container, instruction);
	}
	else if(instruction.isUnary())
	{
		convertUnaryInstruction(container, instruction);
	}
	else if(instruction.isBinary())
	{
		convertBinaryInstruction(container, instruction);
	}
	else
	{
		assertM(false, "Translation for " << instruction.toString()
			<< " not implemented.");
	}
	
	return container;
}

size_t BinaryWriter::getSymbolTableOffset(const ir::Argument* a)
{
	return getSymbolTableOffset(a->mangledName());
}

size_t BinaryWriter::getSymbolTableOffset(const ir::Variable* g)
{
	return getSymbolTableOffset(g->name());
}

size_t BinaryWriter::getBasicBlockSymbolTableOffset(const ir::Variable* g)
{
	auto offset = m_basicBlockOffsets.find(g->name());
	assert(offset != m_basicBlockOffsets.end());

	auto symbol = m_basicBlockSymbols.find(offset->second);
	
	if(symbol == m_basicBlockSymbols.end())
	{
		uint64_t symbolOffset = m_symbolTable.size() *
			sizeof(SymbolTableEntry) + getSymbolTableOffset();

		symbol = m_basicBlockSymbols.insert(std::make_pair(
			offset->second, symbolOffset)).first;

		addSymbol(SymbolTableEntry::BasicBlockType, 0x0, 0x0,
			ir::Global::InvalidLevel, g->name(), offset->second, 0,
			g->type().name);
	}

	return symbol->second;
}

size_t BinaryWriter::getSymbolTableOffset(const std::string& name)
{
	auto symbol = getSymbol(name);
	
	assertM(symbol != m_symbolTable.end(),
		"Invalid symbol name " << name << "");

	return getSymbolTableOffset() +
		std::distance(m_symbolTable.begin(), symbol) *
		sizeof(SymbolTableEntry);
}

BinaryWriter::SymbolTableEntryVector::iterator
	BinaryWriter::getSymbol(const std::string& name)
{
	for(SymbolVector::iterator symbol = m_symbolTable.begin();
		symbol != m_symbolTable.end(); ++symbol)
	{
		std::string symbolName(&m_stringTable[symbol->stringOffset]);

		if(symbolName == name)
		{
			return symbol;
		}
	}

	return m_symbolTable.end();
}

void BinaryWriter::addSymbol(unsigned int type, unsigned int linkage,
	unsigned int visibility, unsigned int level, const std::string& name,
	uint64_t offset, uint64_t size, const std::string& typeName,
	const std::string& attributeList)
{
	report("   adding symbol '" << name
		<< "' with type name '" << typeName << "' and attributes '"
		<< attributeList << "'");
	
	SymbolTableEntry symbol;

	symbol.type                  = type;
	symbol.attributes.linkage    = linkage;
	symbol.attributes.visibility = visibility;
	symbol.attributes.level      = level;
	symbol.stringOffset          = m_stringTable.size();
	symbol.offset                = offset;
	symbol.size                  = size;

	std::copy(name.begin(), name.end(), std::back_inserter(m_stringTable));
	m_stringTable.push_back('\0');
	
	//	Add the type name string
	symbol.typeOffset = m_stringTable.size();
		
	std::copy(typeName.begin(), typeName.end(),
		std::back_inserter(m_stringTable));
	m_stringTable.push_back('\0');
	
	// Add the attribute name string
	symbol.attributeOffset = m_stringTable.size();
		
	std::copy(attributeList.begin(), attributeList.end(),
		std::back_inserter(m_stringTable));
	m_stringTable.push_back('\0');

	// Add the symbol
	m_symbolTable.push_back(symbol);
}

void BinaryWriter::addGlobal(const ir::Global& global)
{
	ir::Constant::DataVector blob;
		
	report("  " << global.name());

	if(global.hasInitializer())
	{
		const ir::Constant* initializer = global.initializer();
		blob = initializer->data();
	}
	else
	{
		blob.resize(global.bytes());
	}

	addSymbol(SymbolTableEntry::VariableType, global.linkage(),
		global.visibility(), global.level(), global.name(), m_data.size(),
		global.bytes(), global.type().name);
	
	std::copy(blob.begin(), blob.end(), std::back_inserter(m_data));
}

void BinaryWriter::patchSymbol(const std::string& name,
	uint64_t offset, uint64_t size)
{
	auto symbol = getSymbol(name);
	
	assertM(symbol != m_symbolTable.end(),
		"Invalid symbol name " << name << "");

	symbol->offset = offset;
	symbol->size   = size;
}

void BinaryWriter::alignData(uint64_t alignment)
{
	uint64_t newSize = align(m_data.size(), alignment);

	m_data.resize(newSize);
}

void BinaryWriter::convertStInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::St& st = static_cast<const ir::St&>(instruction);

	container.asSt.d = convertOperand(*st.d());
	container.asSt.a = convertOperand(*st.a());
}

void BinaryWriter::convertBraInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::Bra& bra = static_cast<const ir::Bra&>(instruction);

	container.asBra.target = convertOperand(*bra.target());
	container.asBra.target = convertOperand(*bra.target());
	
	switch(bra.modifier)
	{
	case ir::Bra::UniformBranch:
	{
		container.asBra.modifier = Bra::UniformBranch;
		break;
	}
	case ir::Bra::MultitargetBranch:
	{
		container.asBra.modifier = Bra::MultitargetBranch;
		break;
	}
	case ir::Bra::InvalidModifier:
	{
		container.asBra.modifier = Bra::InvalidModifier;
		break;
	}
	}
}

void BinaryWriter::convertCallInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::Call& call = static_cast<const ir::Call&>(instruction);

	container.asCall.target = convertOperand(*call.target());
	
	alignData(sizeof(OperandContainer));
	
	auto returnArguments = call.returned();
	
	container.asCall.returnArguments      = returnArguments.size();
	container.asCall.returnArgumentOffset = m_data.size();
	
	for(auto operand : returnArguments)
	{
		addOperandToDataSection(convertOperand(*operand));
	}
	
	auto arguments = call.arguments();
	
	container.asCall.arguments      = arguments.size();
	container.asCall.argumentOffset = m_data.size();
	
	for(auto operand : arguments)
	{
		addOperandToDataSection(convertOperand(*operand));
	}
}

void BinaryWriter::convertRetInstruction(
	InstructionContainer& container, const ir::Instruction& instruction)
{
	// Currently a NOP
	// TODO: fix this
}

void BinaryWriter::convertPhiInstruction(
	InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::Phi& phi = static_cast<const ir::Phi&>(instruction);

	container.asPhi.destination = convertOperand(*phi.d());
	
	alignData(sizeof(OperandContainer));
	
	auto sources = phi.sources();
	auto blocks  = phi.blocks();
	
	container.asPhi.sources       = sources.size();
	container.asPhi.sourcesOffset = m_data.size();
	
	for(auto source : sources)
	{
		addOperandToDataSection(convertOperand(*source));
	}
	
	for(auto block : blocks)
	{
		addOperandToDataSection(convertOperand(
			ir::AddressOperand(const_cast<ir::BasicBlock*>(block), nullptr)));
	}
}

void BinaryWriter::addOperandToDataSection(const OperandContainer& operand)
{
	const char* begin = reinterpret_cast<const char*>(&operand);
	const char* end   = begin + sizeof(OperandContainer);
	
	std::copy(begin, end, std::back_inserter(m_data));
}

uint64_t BinaryWriter::align(uint64_t address, uint64_t alignment)
{
	uint64_t remainder  = address % alignment;
	uint64_t difference = alignment - remainder;

	return remainder == 0 ? address : address + difference;
}

uint64_t BinaryWriter::pageAlign(uint64_t address)
{
	return align(address, PageSize);
}

}

}

