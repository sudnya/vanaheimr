/*! \file   Type.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The header file for the Type class.
	
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <list>

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
	typedef std::list<std::string>   StringList;
	typedef compiler::Compiler       Compiler;
	
public:
	Type(const std::string& name, Compiler* compiler);
	virtual ~Type();

public:
	bool isPrimitive()            const;
	bool isAggregate()            const;
	bool isInteger()              const;
	bool isFloatingPoint()        const;
	bool isSinglePrecisionFloat() const;
	bool isDoublePrecisionFloat() const;
	bool isBasicBlock()           const;
	bool isFunction()             const;
	bool isStructure()            const;
	bool isArray()                const;
	bool isAlias()                const;
	bool isVoid()                 const;

public:
	virtual size_t alignment() const;

public:
	virtual size_t bytes() const = 0;
	virtual Type*  clone() const = 0;

public:
	std::string name;

protected:
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
	Type*  clone() const;

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
	Type*  clone() const;
};

/*! \brief A type for an IEEE compliant 64-bit floating point type */
class DoubleType : public Type
{
public:
	DoubleType(Compiler* c);

public:
	size_t bytes() const;
	Type*  clone() const;
};

/*! \brief Common functionality for aggregates (structures and arrays) */
class AggregateType : public Type
{
public:
	AggregateType(Compiler* c, const std::string& name = "");

public:
	virtual const Type*  getTypeAtIndex  (unsigned int index) const = 0;
	virtual const Type*& getTypeAtIndex  (unsigned int index)       = 0;
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
	const Type*& getTypeAtIndex  (unsigned int index);
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	unsigned int elementsInArray() const;

public:
	size_t bytes() const;
	Type*  clone() const;

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
	const Type*& getTypeAtIndex  (unsigned int index);
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	size_t bytes() const;
	Type*  clone() const;

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
	const Type*& getTypeAtIndex  (unsigned int index);
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	size_t bytes() const;
	Type*  clone() const;

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
	const Type*& getTypeAtIndex  (unsigned int index);
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	size_t bytes() const;
	Type*  clone() const;
	
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
	Type*  clone() const;
};

/*! \brief A type for a variadic function argument */
class VariadicType : public Type
{
public:
	VariadicType(Compiler* c);

public:
	size_t bytes() const;
	Type*  clone() const;
};

/*! \brief A type that is not visible to the compiler */
class OpaqueType : public Type
{
public:
	OpaqueType(Compiler* c);

public:
	size_t bytes() const;
	Type*  clone() const;
};

/*! \brief Corresponds to the C++ void type */
class VoidType : public Type
{
public:
	VoidType(Compiler* c);

public:
	size_t bytes() const;
	Type*  clone() const;
};

/*! \brief A type that mapped to another type with a specific name.

	Note that Aliased types are not legal in the pure form of the IR.
*/
class AliasedType : public Type
{
public:
	AliasedType(Compiler* c, const std::string& name);

public:
	size_t bytes() const;
	Type*  clone() const;

};

}

}

