/*! \file   Type.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The header file for the Type class.
	
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }

namespace vanaheimr
{

namespace ir
{

/*! An arbitrary Vanaheimr type */
class Type
{
public:
	typedef std::vector<const Type*> TypeVector;
	typedef compiler::Compiler       Compiler;
	
public:
	Type(const std::string& name, Compiler* compiler);

public:
	virtual const std::string& name() const;

public:
	bool isPrimitive()            const;
	bool isInteger()              const;
	bool isFloatingPoint()        const;
	bool isSinglePrecisionFloat() const;
	bool isDoublePrecisionFloat() const;
	bool isBasicBlock()           const;

public:
	virtual size_t bytes() const = 0;

private:
	std::string _name;
	Compiler*   _compiler;
	
};

/*! \brief A type for an arbitrary bit-width integer */
class IntegerType : public Type
{
public:
	IntegerType(Compiler* c, unsigned int bits);

public:
	bool isBitWidthAPowerOfTwo() const;
	unsigned int bits() const;
	size_t bytes() const;

public:
	static std::string integerName(unsigned int bits);

private:
	unsigned int _bits;
};

/*! \brief A type for an IEEE compliant 32-bit floating point type */
class FloatType : public Type
{
public:
	FloatType(Compiler* c);

public:
	size_t bytes() const;
};

/*! \brief A type for an IEEE compliant 64-bit floating point type */
class DoubleType : public Type
{
public:
	DoubleType(Compiler* c);

public:
	size_t bytes() const;
};

/*! \brief Common functionality for aggregates (structures and arrays) */
class AggregateType : public Type
{
public:
	AggregateType(Compiler* c);

public:
	virtual const Type*  getTypeAtIndex  (unsigned int index) const = 0;
	virtual bool         isIndexValid    (unsigned int index) const = 0;
	virtual unsigned int numberOfSubTypes(                  ) const = 0;

};

/*! \brief A type for an array of other elements */
class ArrayType : public AggregateType
{
public:
	ArrayType(Compiler* c, const Type* t, unsigned int elementCount);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	unsigned int elementsInArray() const;

private:
	const Type*  _pointedToType;
	unsigned int _elementCount;
};

/*! \brief A type for an arbitrary aggregation of types */
class StructureType : public AggregateType
{
public:
	StructureType(Compiler* c, const TypeVector& types);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

private:
	TypeVector _types;

};

/*! \brief A type for a pointer */
class PointerType : public AggregateType
{
public:
	PointerType(Compiler* c, const Type* t);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

private:
	const Type* _pointedToType;

};

/*! \brief A type for a function */
class FunctionType : public Type
{
public:
	typedef TypeVector::const_iterator iterator;

public:
	FunctionType(Compiler* c, const Type* returnType,
		const TypeVector& argumentTypes);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;
	
public:
	iterator begin() const;
	iterator end()   const;

public:
	static std::string functionPrototypeName(const Type* returnType,
		const TypeVector& argumentTypes);

private:
	const Type* _returnType;
	TypeVector  _argumentTypes;
};

/*! \brief A type for a basic block */
class BasicBlockType : public Type
{
public:
	BasicBlockType(Compiler* c);

public:
	size_t bytes() const;
};

}

}

