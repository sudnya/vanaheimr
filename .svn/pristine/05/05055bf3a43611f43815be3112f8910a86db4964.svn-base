/*! \file   Type.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The source file for the Type class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/math.h>

// Standard Library Includes
#include <typeinfo>
#include <sstream>

namespace vanaheimr
{

namespace ir
{

Type::Type(const std::string& n, Compiler* c)
: _name(n), _compiler(c)
{

}

const std::string& Type::name() const
{
	return _name;
}

bool Type::isPrimitive() const
{
	return isInteger() || isFloatingPoint();
}

bool Type::isInteger() const
{
	return typeid(*this) == typeid(IntegerType);
}

bool Type::isFloatingPoint() const
{
	return isSinglePrecisionFloat() || isDoublePrecisionFloat();
}

bool Type::isSinglePrecisionFloat() const
{
	return typeid(FloatType) == typeid(*this);
}

bool Type::isDoublePrecisionFloat() const
{
	return typeid(DoubleType) == typeid(*this);
}

bool Type::isBasicBlock() const
{
	return typeid(BasicBlockType) == typeid(*this);
}

IntegerType::IntegerType(Compiler* c, unsigned int bits)
: Type(integerName(bits), c), _bits(bits)
{
	
}

bool IntegerType::isBitWidthAPowerOfTwo() const
{
	return hydrazine::isPowerOfTwo(bits());
}

unsigned int IntegerType::bits() const
{
	return _bits;
}

size_t IntegerType::bytes() const
{
	return (bits() + 7) / 8;
}

std::string IntegerType::integerName(unsigned int bits)
{
	std::stringstream stream;
	
	stream << "i" << bits;
	
	return stream.str();
}

FloatType::FloatType(Compiler* c)
: Type("float", c)
{

}

size_t FloatType::bytes() const
{
	return 4;
}

DoubleType::DoubleType(Compiler* c)
: Type("double", c)
{

}

size_t DoubleType::bytes() const
{
	return 8;
}

AggregateType::AggregateType(Compiler* c)
: Type("", c)
{

}

ArrayType::ArrayType(Compiler* c, const Type* t, unsigned int elementCount)
: AggregateType(c), _pointedToType(t), _elementCount(elementCount)
{

}

const Type* ArrayType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;

	return _pointedToType;
}

bool ArrayType::isIndexValid(unsigned int index) const
{
	return index == 0;
}

unsigned int ArrayType::numberOfSubTypes() const
{
	return 1;
}

unsigned int ArrayType::elementsInArray() const
{
	return _elementCount;
}

StructureType::StructureType(Compiler* c, const TypeVector& types)
: AggregateType(c), _types(types)
{

}

const Type* StructureType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;
	
	return _types[index];
}

bool StructureType::isIndexValid(unsigned int index) const
{
	return index < numberOfSubTypes();
}

unsigned int StructureType::numberOfSubTypes() const
{
	return _types.size();
}

PointerType::PointerType(Compiler* c, const Type* t)
: AggregateType(c), _pointedToType(t)
{

}

const Type* PointerType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;
	
	return _pointedToType;
}

bool PointerType::isIndexValid(unsigned int index) const
{
	return index == 0;
}

unsigned int PointerType::numberOfSubTypes() const
{
	return 1;
}

FunctionType::FunctionType(Compiler* c, const Type* returnType,
	const TypeVector& argumentTypes)
: Type(functionPrototypeName(returnType, argumentTypes), c),
	_returnType(returnType), _argumentTypes(argumentTypes)
{

}

const Type* FunctionType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;
	
	if(_returnType != 0)
	{
		if(index == 0)
		{
			return _returnType;
		}
		
		index -= 1;
	}
	
	return _argumentTypes[index];
}

bool FunctionType::isIndexValid(unsigned int i) const
{
	return i < numberOfSubTypes();
}

unsigned int FunctionType::numberOfSubTypes() const
{
	unsigned int indices = _returnType == 0 ? 1 : 0;
	
	indices += _argumentTypes.size();
	
	return indices;
}

FunctionType::iterator FunctionType::begin() const
{
	return _argumentTypes.begin();
}

FunctionType::iterator FunctionType::end() const
{
	return _argumentTypes.end();
}

std::string FunctionType::functionPrototypeName(const Type* returnType,
	const TypeVector& argumentTypes)
{
	std::stringstream stream;

	if(returnType != 0)
	{
		stream << returnType->name() << " ";
	}
	
	stream << "(";
	
	for(TypeVector::const_iterator type = argumentTypes.begin();
		type != argumentTypes.end(); ++type)
	{
		if(type != argumentTypes.end()) stream << ", ";
		
		stream << (*type)->name();
	}
	
	stream << ")";
	
	return stream.str();
}

BasicBlockType::BasicBlockType(Compiler* c)
: Type("_ZTBasicBlock", c)
{

}

size_t BasicBlockType::bytes() const
{
	return 0;
}

}

}

