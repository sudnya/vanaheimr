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
: name(n), _compiler(c) 
{

}

Type::~Type()
{

}

bool Type::isPrimitive() const
{
	return isInteger() || isFloatingPoint() || isVoid();
}

bool Type::isAggregate() const
{
	return dynamic_cast<const AggregateType*>(this) != nullptr;
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

bool Type::isFunction() const
{
	return typeid(FunctionType) == typeid(*this);
}

bool Type::isStructure() const
{
	return typeid(StructureType) == typeid(*this);
}

bool Type::isArray() const
{
	return typeid(ArrayType) == typeid(*this);
}

bool Type::isAlias() const
{
	return typeid(AliasedType) == typeid(*this);
}

bool Type::isVoid() const
{
	return typeid(VoidType) == typeid(*this);
}

size_t Type::alignment() const
{
	return bytes();
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

Type* IntegerType::clone() const
{
	return new IntegerType(*this);
}

std::string IntegerType::integerName(unsigned int bits)
{
	std::stringstream stream;
	
	stream << "i" << bits;
	
	return stream.str();
}

FloatType::FloatType(Compiler* c)
: Type("f32", c)
{

}

size_t FloatType::bytes() const
{
	return 4;
}

Type* FloatType::clone() const
{
	return new FloatType(*this);
}

DoubleType::DoubleType(Compiler* c)
: Type("f64", c)
{

}

size_t DoubleType::bytes() const
{
	return 8;
}

Type* DoubleType::clone() const
{
	return new DoubleType(*this);
}

AggregateType::AggregateType(Compiler* c, const std::string& name)
: Type(name, c)
{

}

static std::string arrayTypeName(const Type* t, unsigned int count)
{
	std::stringstream stream;
	
	stream << t->name << "[" << count << "]";
	
	return stream.str();
}

ArrayType::ArrayType(Compiler* c, const Type* t, unsigned int elementCount)
: AggregateType(c, arrayTypeName(t, elementCount)), _pointedToType(t),
	_elementCount(elementCount)
{

}

const Type* ArrayType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;

	return _pointedToType;
}

const Type*& ArrayType::getTypeAtIndex(unsigned int index)
{
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

size_t ArrayType::bytes() const
{
	return _pointedToType->bytes() * _elementCount;
}

Type* ArrayType::clone() const
{
	return new ArrayType(*this);
}

static std::string structureTypeName(const Type::TypeVector& types)
{
	std::stringstream stream;
	
	stream << "{";

	for(auto type = types.begin(); type != types.end(); ++type)
	{
		if(type != types.begin()) stream << ", ";
	
		stream << (*type)->name;
	}
	
	stream << "}";
	
	return stream.str();
}

StructureType::StructureType(Compiler* c, const TypeVector& types)
: AggregateType(c, structureTypeName(types)), _types(types)
{

}

const Type* StructureType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return 0;
	
	return _types[index];
}

const Type*& StructureType::getTypeAtIndex(unsigned int index)
{
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

size_t StructureType::bytes() const
{
	size_t count = 0;

	for(auto type : _types)
	{
		count += type->bytes();
	}

	return count;
}

Type* StructureType::clone() const
{
	return new StructureType(*this);
}

PointerType::PointerType(Compiler* c, const Type* t)
: AggregateType(c, t->name + "*"), _pointedToType(t)
{

}

const Type* PointerType::getTypeAtIndex(unsigned int index) const
{
	return _pointedToType;
}

const Type*& PointerType::getTypeAtIndex(unsigned int index)
{
	return _pointedToType;
}

bool PointerType::isIndexValid(unsigned int index) const
{
	return true;
}

unsigned int PointerType::numberOfSubTypes() const
{
	return 1;
}

size_t PointerType::bytes() const
{
	// TODO Get the target pointer size

	return sizeof(void*);
}

Type* PointerType::clone() const
{
	return new PointerType(*this);
}

FunctionType::FunctionType(Compiler* c, const Type* returnType,
	const TypeVector& argumentTypes)
: Type(functionPrototypeName(returnType, argumentTypes), c),
	_returnType(returnType), _argumentTypes(argumentTypes)
{

}

const Type* FunctionType::getTypeAtIndex(unsigned int index) const
{
	if(!isIndexValid(index)) return nullptr;
	
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

const Type*& FunctionType::getTypeAtIndex(unsigned int index)
{
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
		stream << "(" << returnType->name << ") ";
	}
	else
	{
		stream << "() ";
	}
	
	stream << "(";
	
	for(TypeVector::const_iterator type = argumentTypes.begin();
		type != argumentTypes.end(); ++type)
	{
		if(type != argumentTypes.begin()) stream << ", ";
		
		stream << (*type)->name;
	}
	
	stream << ")";
	
	return stream.str();
}

size_t FunctionType::bytes() const
{
	return 0;
}

Type* FunctionType::clone() const
{
	return new FunctionType(*this);
}

BasicBlockType::BasicBlockType(Compiler* c)
: Type("_ZTBasicBlock", c)
{

}

size_t BasicBlockType::bytes() const
{
	return 0;
}

Type* BasicBlockType::clone() const
{
	return new BasicBlockType(*this);
}

VariadicType::VariadicType(Compiler* c)
: Type("_ZTVariadic", c)
{

}

size_t VariadicType::bytes() const
{
	return 0;
}

Type* VariadicType::clone() const
{
	return new VariadicType(*this);
}

OpaqueType::OpaqueType(Compiler* c)
: Type("_ZOpaque", c)
{

}

size_t OpaqueType::bytes() const
{
	return 0;
}

Type* OpaqueType::clone() const
{
	return new OpaqueType(*this);
}

VoidType::VoidType(Compiler* c)
: Type("void", c)
{

}

size_t VoidType::bytes() const
{
	return 0;
}

Type* VoidType::clone() const
{
	return new VoidType(*this);
}

AliasedType::AliasedType(Compiler* c, const std::string& name)
: Type(name, c)
{

}

size_t AliasedType::bytes() const
{
	return 0;
}

Type* AliasedType::clone() const
{
	return new AliasedType(*this);
}

}

}

