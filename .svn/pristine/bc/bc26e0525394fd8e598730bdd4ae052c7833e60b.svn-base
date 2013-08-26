/*!	\file   BinaryWriter.cpp
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The implementation file for the helper class that traslates compiler IR to a binary.
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

	binary.write((const char*)&m_header, sizeof(BinaryHeader));
	binary.write((const char*)m_symbolTable.data(), getSymbolTableSize());
	binary.write((const char*)m_instructions.data(), getInstructionStreamSize());
	binary.write((const char*)m_data.data(), getDataSize());
	binary.write((const char*)m_stringTable.data(), getStringTableSize());

}

void BinaryWriter::populateData()
{
	report(" Adding global variables...");

	for (ir::Module::const_global_iterator i = m_module->global_begin(); i != m_module->global_end(); ++i)
	{
		ir::Constant::DataVector blob;
		
		report("  " << i->name());

		if (i->hasInitializer())
		{
			const ir::Constant* initializer = i->initializer();
			blob = initializer->data();
		}
		else
		{
			blob.resize(i->bytes());
		}

		addSymbol(0x1, i->linkage(), i->visibility(), i->name(), m_data.size(), i->bytes());

		std::copy(blob.begin(), blob.end(), std::back_inserter(m_data));
	}
}

void BinaryWriter::populateInstructions()
{
	report(" Adding functions.");
	for(ir::Module::const_iterator function = m_module->begin(); function != m_module->end(); ++function)
	{
		report("  " << function->name());

		for(ir::Function::const_argument_iterator argument = function->argument_begin();
			argument != function->argument_end(); ++argument)
		{
			addSymbol(0x3, 0x0, 0x0, argument->mangledName(), m_data.size(), 0x0);
			m_data.resize(m_data.size() + argument->type().bytes());
		}

		unsigned int instructionsBegin =
			m_instructions.size() * sizeof(InstructionContainer);
		unsigned int instructionOffset = m_instructions.size();	
		for(ir::Function::const_iterator bb = function->begin(); bb != function->end(); ++bb)
		{
			m_basicBlockOffsets.insert(std::make_pair(bb->name(),
				instructionOffset * sizeof(InstructionContainer)));

			instructionOffset += bb->size();
		}

		unsigned int instructionsSize =
			instructionOffset * sizeof(InstructionContainer) - instructionsBegin;
	
		for(ir::Function::const_iterator bb = function->begin(); bb != function->end(); ++bb)
		{
			report("   Basic Block " << bb->name());
			for(ir::BasicBlock::const_iterator inst = bb->begin(); inst != bb->end(); ++inst)
			{
				m_instructions.push_back(convertToContainer(**inst));
			}
		}

		addSymbol(0x2, function->linkage(), function->visibility(),
			function->name(), instructionsBegin, instructionsSize);
		
		m_basicBlockOffsets.clear();
		m_basicBlockSymbols.clear();
	}
}

void BinaryWriter::linkSymbols()
{
	for (symbol_iterator symb = m_symbolTable.begin(); symb != m_symbolTable.end(); ++symb)
	{
		if (symb->type == 1)
		{
			symb->offset += getInstructionOffset();
		}
		else if (symb->type == 2)
		{
			symb->offset += getDataOffset();
		}
	}
}

void BinaryWriter::populateHeader()
{
	m_header.dataPages     = (m_data.size() + PageSize - 1) / PageSize; 
	m_header.codePages     = ((m_instructions.size()*sizeof(InstructionContainer)) + PageSize - 1) / PageSize;
	m_header.symbols       = m_symbolTable.size(); 
	m_header.stringPages   = (m_stringTable.size() + PageSize - 1) / PageSize;
	m_header.dataOffset    = getDataOffset();
	m_header.codeOffset    = getInstructionOffset();
	m_header.symbolOffset  = getSymbolTableOffset();
	m_header.stringsOffset = getStringTableOffset();
}

size_t BinaryWriter::getHeaderOffset() const
{
	return 0;
}

size_t BinaryWriter::getInstructionOffset() const
{
	return sizeof(m_header);
}

size_t BinaryWriter::getDataOffset() const
{
	return getInstructionStreamSize() + getInstructionOffset();
}

size_t BinaryWriter::getSymbolTableOffset() const
{
	return getDataSize() + getDataOffset();
}

size_t BinaryWriter::getStringTableOffset() const
{
	 return getSymbolTableSize() + getSymbolTableOffset();
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

static archaeopteryx::ir::Instruction::Opcode convertOpcode(ir::Instruction::Opcode opcode)
{
	typedef archaeopteryx::ir::Instruction AInstruction;
	
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

static archaeopteryx::ir::DataType convertType(const ir::Type* type)
{
	if(type->isInteger())
	{
		const ir::IntegerType* integer = static_cast<const ir::IntegerType*>(type);

		switch(integer->bits())
		{
		case 1:
		{
			return archaeopteryx::ir::i1;
		}
		case 8:
		{
			return archaeopteryx::ir::i8;
		}
		case 16:
		{
			return archaeopteryx::ir::i16;
		}
		case 32:
		{
			return archaeopteryx::ir::i32;
		}
		case 64:
		{
			return archaeopteryx::ir::i64;
		}
		default: assertM(false, "Invalid integer bit width.");
		}
	}
	else if(type->isFloatingPoint())
	{
		if(type->isSinglePrecisionFloat())
		{
			return archaeopteryx::ir::f32;
		}
		else
		{
			return archaeopteryx::ir::f64;
		}
	}

	assertM(false, "Data type conversion not implemented in binary writer");

	return archaeopteryx::ir::InvalidDataType;
}

static archaeopteryx::ir::PredicateOperand::PredicateModifier convertPredicate(
	ir::PredicateOperand::PredicateModifier modifier)
{
	switch(modifier)
	{
	case ir::PredicateOperand::StraightPredicate:
	{
		return archaeopteryx::ir::PredicateOperand::StraightPredicate;
	}
	case ir::PredicateOperand::InversePredicate:
	{
		return archaeopteryx::ir::PredicateOperand::InversePredicate;
	}
	case ir::PredicateOperand::PredicateTrue:
	{
		return archaeopteryx::ir::PredicateOperand::PredicateTrue;
	}
	case ir::PredicateOperand::PredicateFalse:
	{
		return archaeopteryx::ir::PredicateOperand::PredicateFalse;
	}
	default: break;
	}

	assertM(false, "Invalid predicate.");

	return archaeopteryx::ir::PredicateOperand::InvalidPredicate;
}

archaeopteryx::ir::OperandContainer BinaryWriter::convertOperand(const ir::Operand& operand)
{
	archaeopteryx::ir::OperandContainer result;

	switch(operand.mode())
	{
	case ir::Operand::Register:
	{
		const ir::RegisterOperand& reg = static_cast<const ir::RegisterOperand&>(operand);

		result.asRegister.reg  = reg.virtualRegister->id;
		result.asRegister.type = convertType(reg.virtualRegister->type);
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Register;
		break;
	}
	case ir::Operand::Immediate:
	{
		const ir::ImmediateOperand& immediate = static_cast<const ir::ImmediateOperand&>(operand);
		
		result.asImmediate.type = convertType(immediate.type);
		result.asImmediate.uint = immediate.uint;
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Immediate;
		break;
	}
	case ir::Operand::Predicate:
	{
		const ir::PredicateOperand& predicate = static_cast<const ir::PredicateOperand&>(operand);
		
		result.asPredicate.modifier = convertPredicate(predicate.modifier);
		
		if(predicate.modifier == ir::PredicateOperand::StraightPredicate ||
			predicate.modifier == ir::PredicateOperand::InversePredicate)
		{
			result.asPredicate.reg = predicate.virtualRegister->id;
		}
		else
		{
			result.asPredicate.reg = 0;
		}
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Predicate;

		break;
	}
	case ir::Operand::Indirect:
	{
		const ir::IndirectOperand& indirect = static_cast<const ir::IndirectOperand&>(operand);

		result.asIndirect.reg    = indirect.virtualRegister->id;
		result.asIndirect.type   = convertType(indirect.virtualRegister->type);
		result.asIndirect.offset = indirect.offset;
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Indirect;

		break;
	}
	case ir::Operand::Address:
	{
		const ir::AddressOperand& address = static_cast<const ir::AddressOperand&>(operand);
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Symbol;

		if(address.globalValue->type().isBasicBlock())
		{
			result.asSymbol.symbolTableOffset = getBasicBlockSymbolTableOffset(address.globalValue);
		}
		else
		{
			result.asSymbol.symbolTableOffset = getSymbolTableOffset(address.globalValue);
		}

		break;
	}
	case ir::Operand::Argument:	
	{
		const ir::ArgumentOperand& argument = static_cast<const ir::ArgumentOperand&>(operand);

		result.asSymbol.symbolTableOffset = getSymbolTableOffset(argument.argument);
		
		result.asOperand.mode = archaeopteryx::ir::Operand::Symbol;

		break;
	}
	}

	return result;
}

static bool isComplexInstruction(const ir::Instruction& instruction)
{
	switch(instruction.opcode)
	{
	case ir::Instruction::St:
	case ir::Instruction::Bra:
	case ir::Instruction::Ret:
	{
		return true;
	}
	default: break;;
	}
	
	return false;
}

void BinaryWriter::convertComplexInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	switch(instruction.opcode)
	{
	case ir::Instruction::Bra:
	{
		convertBraInstruction(container, instruction);
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
	default: assertM(false, "Translation for " << instruction.toString() << " not implemented.");
	}
}

void BinaryWriter::convertUnaryInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::UnaryInstruction& unary =
		static_cast<const ir::UnaryInstruction&>(instruction);

	container.asUnaryInstruction.d = convertOperand(*unary.d);
	container.asUnaryInstruction.a = convertOperand(*unary.a);
}

void BinaryWriter::convertBinaryInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::BinaryInstruction& binary =
		static_cast<const ir::BinaryInstruction&>(instruction);

	container.asBinaryInstruction.d = convertOperand(*binary.d);
	container.asBinaryInstruction.a = convertOperand(*binary.a);
	container.asBinaryInstruction.b = convertOperand(*binary.b);
}

BinaryWriter::InstructionContainer BinaryWriter::convertToContainer(const Instruction& instruction)
{
	report("    " << instruction.toString());

	InstructionContainer container;

	container.asInstruction.opcode = convertOpcode(instruction.opcode);
	container.asInstruction.guard  = convertOperand(*instruction.guard).asPredicate;

	if(isComplexInstruction(instruction))
	{
		convertComplexInstruction(container, instruction);
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
		assertM(false, "Translation for " << instruction.toString() << " not implemented.");
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
		symbol = m_basicBlockSymbols.insert(std::make_pair(
			offset->second, m_symbolTable.size())).first;
	
		addSymbol(0x4, 0, 0, g->name(), offset->second, 0);
	}

	return symbol->second;
}

size_t BinaryWriter::getSymbolTableOffset(const std::string& name)
{
	for(SymbolVector::iterator symbol = m_symbolTable.begin();
		symbol != m_symbolTable.end(); ++symbol)
	{
		std::string symbolName(&m_stringTable[symbol->stringOffset]);

		if(symbolName == name)
		{
			return getSymbolTableOffset() +
				std::distance(m_symbolTable.begin(), symbol) * sizeof(SymbolTableEntry);
		}
	}

	assertM(false, "Invalid symbol name " << name << "");

	return -1;
}

void BinaryWriter::addSymbol(unsigned int type, unsigned int linkage,
	unsigned int visibility, const std::string& name, uint64_t offset,
	uint64_t size)
{
	SymbolTableEntry symbol;

	symbol.type                  = type;
	symbol.attributes.linkage    = linkage;
	symbol.attributes.visibility = visibility;
	symbol.stringOffset          = m_stringTable.size();
	symbol.offset                = offset;
	symbol.size                  = size;

	m_symbolTable.push_back(symbol);
	
	std::copy(name.begin(), name.end(), std::back_inserter(m_stringTable));
	m_stringTable.push_back('\0');
}

void BinaryWriter::convertStInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::St& st = static_cast<const ir::St&>(instruction);

	container.asSt.d = convertOperand(*st.d);
	container.asSt.a = convertOperand(*st.a);
}

void BinaryWriter::convertRetInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	// Currently a NOP
}

void BinaryWriter::convertBraInstruction(archaeopteryx::ir::InstructionContainer& container,
	const ir::Instruction& instruction)
{
	const ir::Bra& bra = static_cast<const ir::Bra&>(instruction);

	container.asBra.target = convertOperand(*bra.target);
	
	switch(bra.modifier)
	{
	case ir::Bra::UniformBranch:
	{
		container.asBra.modifier = archaeopteryx::ir::Bra::UniformBranch;
		break;
	}
	case ir::Bra::MultitargetBranch:
	{
		container.asBra.modifier = archaeopteryx::ir::Bra::MultitargetBranch;
		break;
	}
	case ir::Bra::InvalidModifier:
	{
		container.asBra.modifier = archaeopteryx::ir::Bra::InvalidModifier;
		break;
	}
	}
}

}

}

