/*! \file   AssemblyWriter.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday March 4, 2012
	\brief  The source file for the AssemblyWriter class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/AssemblyWriter.h>

#include <vanaheimr/ir/interface/Module.h>
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace as
{

AssemblyWriter::AssemblyWriter()
{

}

void AssemblyWriter::write(std::ostream& stream, const ir::Module& module)
{
	report("Writing assembly for module '" << module.name << "'");

	for(auto function : module)
	{
		writeFunction(stream, function);
	}
	
	for(auto global = module.global_begin();
		global != module.global_end(); ++global)
	{
		writeGlobal(stream, *global);
	}
}

void AssemblyWriter::writeFunction(std::ostream& stream,
	const ir::Function& function)
{
	report(" For function '" << function.name() << "'");

	stream << ".function ";
	
	writeLinkage(stream, function); 
	
	stream << " " << function.name() << "(";
	
	for(ir::Function::const_argument_iterator
		argument = function.argument_begin();
		argument != function.argument_end(); ++argument)
	{
		if(argument != function.argument_begin()) stream << ", ";

		writeArgument(stream, *argument);
	}
	
	stream << ")\n{\n";
	
	for(ir::Function::const_iterator block = function.begin();
		block != function.end(); ++block)
	{
		if(block == function.exit_block())  continue;
		if(block == function.entry_block()) continue;
		writeBasicBlock(stream, *block);
	}
	
	stream << "}\n";
}

void AssemblyWriter::writeGlobal(std::ostream& stream, const ir::Global& global)
{
	report(" For global '" << global.name() << "'");
	
	stream << ".global ";
	
	writeLinkage(stream, global);
	writeType(stream, global.type());
	
	stream << global.name() << " ";
	
	if(global.hasInitializer())
	{
		stream << " = ";
		writeInitializer(stream, *global.initializer());
	}
}

void AssemblyWriter::writeLinkage(std::ostream& stream,
	const ir::Variable& variable)
{
	switch(variable.linkage())
	{
	case ir::Variable::ExternalLinkage:
	{
		stream << ".external ";
		break;
	}
	case ir::Variable::LinkOnceAnyLinkage:
	{
		stream << ".inline ";
		break;
	}
	case ir::Variable::LinkOnceODRLinkage:
	{
		stream << ".inline_strict ";
		break;
	}
	case ir::Variable::WeakAnyLinkage:
	{
		stream << ".weak ";
		break;
	}
	case ir::Variable::InternalLinkage:
	{
		stream << ".internal ";
		break;
	}
	case ir::Variable::PrivateLinkage:
	{
		stream << ".private ";
		break;
	}
	}
	
	switch(variable.visibility())
	{
	case ir::Variable::VisibleVisibility:
	{
		stream << ".visible ";
		break;
	}
	case ir::Variable::ProtectedVisibility:
	{
		stream << ".protected ";
		break;
	}
	case ir::Variable::HiddenVisibility:
	{
		stream << ".hidden ";
		break;
	}
	}
}

void AssemblyWriter::writeArgument(std::ostream& stream,
	const ir::Argument& argument)
{
	writeType(stream, argument.type());
	
	stream << " " << argument.name();
}

void AssemblyWriter::writeBasicBlock(std::ostream& stream,
	const ir::BasicBlock& block)
{
	stream << "\t BB_" << block.id() << ":\n";
	
	for(auto instruction : block)
	{
		stream << "\t\t";
		stream << instruction->toString();
		stream << "\n";
	}
}

void AssemblyWriter::writeType(std::ostream& stream, const ir::Type& type)
{
	if(type.isPrimitive())
	{
		if(type.isInteger())
		{
			const ir::IntegerType& integerType =
				static_cast<const ir::IntegerType&>(type);
				
			stream << ".i" << integerType.bits() << " ";
		}
		else if(type.isSinglePrecisionFloat())
		{
			stream << ".float ";
		}
		else if(type.isDoublePrecisionFloat())
		{
			stream << ".double ";
		}
		else
		{
			throw std::runtime_error("Invalid primitive type " + type.name());
		}
	}
	else
	{
		assertM(false, "Not implemented.");
	}
}

void AssemblyWriter::writeInitializer(std::ostream& stream,
	const ir::Constant& constant)
{
	assertM(false, "Not implemented.");
}

void AssemblyWriter::writeOpcode(std::ostream& stream, unsigned int opcode)
{
	stream << ir::Instruction::toString((ir::Instruction::Opcode)opcode);
}

void AssemblyWriter::writeOperand(std::ostream& stream, const ir::Operand& o)
{
	switch(o.mode())
	{
	case ir::Operand::Register:
	{
		const ir::RegisterOperand& operand =
			static_cast<const ir::RegisterOperand&>(o);
		
		writeVirtualRegister(stream, *operand.virtualRegister);
		
		break;
	}
	case ir::Operand::Immediate:
	{
		const ir::ImmediateOperand& operand =
			static_cast<const ir::ImmediateOperand&>(o);
		
		writeType(stream, *operand.type);
		
		stream << "0x" << std::hex << operand.uint << std::dec;
		
		break;
	}
	case ir::Operand::Predicate:
	{
		const ir::PredicateOperand& operand =
			static_cast<const ir::PredicateOperand&>(o);
		
		switch(operand.modifier)
		{
		case ir::PredicateOperand::InversePredicate:
		{
			stream << "!";
			
			// fall through
		}
		case ir::PredicateOperand::StraightPredicate:
		{
			stream << "@";
			
			writeVirtualRegister(stream, *operand.virtualRegister);

			break;
		}
		case ir::PredicateOperand::PredicateTrue:
		{
			stream << "@pt";
			break;
		}
		case ir::PredicateOperand::PredicateFalse:
		{
			stream << "!@pt";
			break;
		}
		}
			
		break;
	}
	case ir::Operand::Indirect:
	{
		const ir::IndirectOperand& operand =
			static_cast<const ir::IndirectOperand&>(o);
		
		stream << "[ ";
		writeVirtualRegister(stream, *operand.virtualRegister);
	
		stream << " + " << std::hex << operand.offset << std::dec << " ]";
		
		break;
	}
	case ir::Operand::Address:
	{
		const ir::AddressOperand& operand =
			static_cast<const ir::AddressOperand&>(o);
		
		writeType(stream, operand.globalValue->type());
		
		stream << " ";
		
		stream << operand.globalValue->name();
		
		break;
	}
	case ir::Operand::Argument:
	{
		const ir::ArgumentOperand& operand =
			static_cast<const ir::ArgumentOperand&>(o);
		
		writeType(stream, operand.argument->type());
		
		stream << " ";
		
		stream << operand.argument->name();
		
		break;
	}
	}
}

void AssemblyWriter::writeVirtualRegister(std::ostream& stream,
	const ir::VirtualRegister& v)
{
	writeType(stream, *v.type);
	stream << "%r" << v.id;
}

}

}

