/*! \file   Constant.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The header file for the Constant family of classes.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class BasicBlock; } }
namespace vanaheimr { namespace ir { class Type;       } }

namespace vanaheimr
{

namespace ir
{

/*! \brief A base class for any constant */
class Constant
{
public:
	typedef std::vector<uint8_t> DataVector;

public:
	Constant(const Type* type);
	virtual ~Constant();

public:
	/*! \brief Get the size of the constant in bytes */
	virtual size_t bytes() const;

public:
	/*! \brief Is the constant equivalent to 0 */
	virtual bool isNullValue() const = 0;
	
	/*! \brief Get a binary representation of the constant */
	virtual DataVector data() const = 0;
	
	/*! \brief Create a new copy of the constant */
	virtual Constant* clone() const = 0;
	
public:
	const Type* type() const;
	
public:
	static Constant* parseConstantFromString(const std::string& value);
	
private:
	const Type* _type;
};

/*! \brief Floating point data */
class FloatingPointConstant : public Constant
{
public:
	FloatingPointConstant(float f);
	FloatingPointConstant(double d);

public:
	float  asFloat()  const;
	double asDouble() const;

public:
	virtual bool isNullValue() const;
	virtual DataVector data() const;

public:
	virtual Constant* clone() const;
	
private:
	union
	{
		float  _float;
		double _double;
	};

};

/*! \brief Integer data */
class IntegerConstant : public Constant
{
public:
	IntegerConstant(uint64_t i, unsigned int bits = 64);	

public:
	operator uint64_t() const;
	operator int64_t()  const;
	operator uint32_t() const;
	operator int32_t()  const;
	
public:
	virtual bool isNullValue() const;
	virtual DataVector data() const;

public:
	virtual Constant* clone() const;

private:
	// TODO: Arbitrary precision
	uint64_t     _value;
};

/*! \brief A pointer constant */
class PointerConstant : public Constant
{
public:
	PointerConstant(uint64_t i);	
	PointerConstant(void* i);	

public:
	operator uint64_t() const;
	operator int64_t()  const;
	operator uint32_t() const;
	operator int32_t()  const;
	operator void*()    const;
	
public:
	PointerConstant& operator=(uint64_t i);
	PointerConstant& operator=(void*    i);

public:
	virtual bool isNullValue() const;
	virtual DataVector data() const;

public:
	virtual Constant* clone() const;

private:
	uint64_t _pointer;
};

/*! \brief A structure constant */
class StructureConstant : public Constant
{
public:
	StructureConstant(const Type* aggregateType);

public:
	Constant*       getMember(unsigned int index);
	const Constant* getMember(unsigned int index) const;

public:
	void setMember(unsigned int index, Constant*);

public:
	unsigned int numberOfSubTypes() const;

public:
	virtual bool isNullValue() const;
	virtual DataVector data() const;

public:
	virtual Constant* clone() const;

private:
	typedef std::vector<Constant*> ConstantVector;

private:
	ConstantVector _members;
};

/*! \brief An array constant for normal data types */
class ArrayConstant : public Constant
{
public:
	ArrayConstant(const void* data, uint64_t size, const Type* t);
	ArrayConstant(uint64_t size, const Type* t);

public:
	Constant*       getMember(unsigned int index);
	const Constant* getMember(unsigned int index) const;

public:
	uint64_t size() const;

public:
	void* storage();

public:
	virtual bool isNullValue() const;
	virtual DataVector data() const;

public:
	virtual Constant* clone() const;

private:
	DataVector _value;
};

}

}

