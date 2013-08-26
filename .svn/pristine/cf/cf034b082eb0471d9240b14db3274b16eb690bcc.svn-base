/*! \file   ApplicationBinaryInterface.h
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ApplicationBinaryInterface class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/util/interface/LargeMap.h>

// Standard Library Includes
#include <string>
#include <vector>
#include <cstdint>

// Forward Declarations
namespace vanaheimr { namespace ir { class Type; } }

namespace vanaheimr
{

/*! \brief A namespace for abstract application binary interface functions */
namespace abi
{

/*! \brief A class representing the ABI for a target machine */
class ApplicationBinaryInterface
{
public:
	class MemoryRegion
	{
	public:
		enum Binding
		{
			Register, // address is bound to a register
			Fixed,    // at a fixed address
			Indirect  // accessed via a load through another 
		};

	public:
		MemoryRegion(const std::string& name, unsigned int bytes,
			unsigned int alginment, unsigned int level, Binding binding);

	public:
		Binding binding() const;
	
		bool isRegister() const;
		bool    isFixed() const;
		bool isIndirect() const;
	
	public:
		std::string  name;
		unsigned int bytes;
		unsigned int alignment;
		unsigned int level;

	private:
		Binding _binding;
	};
	
	typedef MemoryRegion::Binding MemoryBinding;
	
	class RegisterBoundRegion : public MemoryRegion
	{
	public:
		RegisterBoundRegion(const std::string& name, unsigned int bytes,
			unsigned int alginment, unsigned int level,
			const std::string& registerName);
		
	public:
		std::string registerName; // The register containing the reference
	};
	
	class FixedAddressRegion : public MemoryRegion
	{
	public:
		FixedAddressRegion(const std::string& name, unsigned int bytes,
			unsigned int alginment, unsigned int level,
			uint64_t address);
		
	public:
		uint64_t address; // The global memory space address
	};
	
	class IndirectlyAddressedRegion : public MemoryRegion
	{
	public:
		IndirectlyAddressedRegion(const std::string& name, unsigned int bytes,
			unsigned int alginment, unsigned int level,
			const std::string& region, unsigned int offset);
	
	public:
		std::string  region; // The name of the region containing the reference
		unsigned int offset; // The offset within the region
	};

	/*! A variable in the program that is bound to an ABI abstraction */
	class BoundVariable
	{
	public:
		enum Binding
		{
			Register,
			Memory
		};

	public:
		BoundVariable(const std::string& name, const ir::Type* type,
			Binding binding);

	public:
		Binding binding() const;

	public:
		std::string     name;
		const ir::Type* type;

	private:
		Binding _binding;
	};
	
	typedef BoundVariable::Binding VariableBinding;
	
	class RegisterBoundVariable: public BoundVariable
	{
	public:
		RegisterBoundVariable(const std::string& name, const ir::Type* type,
			const std::string& registerName);
	
	public:
		std::string registerName; // The name of the bound register
	};
	
	class MemoryBoundVariable: public BoundVariable
	{
	public:
		MemoryBoundVariable(const std::string& name, const ir::Type* type,
			const std::string& regionName);
	
	public:
		std::string  region; // The memory region containing the variable
		unsigned int offset; // The offset within the region
	};
	
	typedef util::LargeMap<std::string, MemoryRegion*>  MemoryRegionMap;
	typedef util::LargeMap<std::string, BoundVariable*> BoundVariableMap;
	
	typedef MemoryRegionMap::const_iterator  const_region_iterator;
	typedef BoundVariableMap::const_iterator const_variable_iterator;

public:
	ApplicationBinaryInterface();
	/*! \brief Destroy all regions and variables */
	~ApplicationBinaryInterface();

public:
	/*! \brief Is the ABI valid?  return false if not and
		set message to indicate why */
	bool validate(std::string& message);

public:
	/* \brief Get a pointer to a region or 0 if it does not exist */
	MemoryRegion*  findRegion(const std::string& name);
	/* \brief Get a pointer to a variable or 0 if it does not exist */
	BoundVariable* findVariable(const std::string& name);

	/* \brief Get a pointer to a region or 0 if it does not exist */
	const MemoryRegion*  findRegion(const std::string& name) const;
	/* \brief Get a pointer to a variable or 0 if it does not exist */
	const BoundVariable* findVariable(const std::string& name) const;

public:
	MemoryRegion*  insert(MemoryRegion*  region  );
	BoundVariable* insert(BoundVariable* variable);

public:
	const_region_iterator regions_begin() const;
	const_region_iterator regions_end() const;
	
	const_variable_iterator variables_begin() const;
	const_variable_iterator variables_end() const;

public:
	ApplicationBinaryInterface(const ApplicationBinaryInterface&) = delete;
	ApplicationBinaryInterface&
		operator=(const ApplicationBinaryInterface&) = delete;

public:
	/*! \brief Get the global singleton corresponding to the named ABI */
	static const ApplicationBinaryInterface* getABI(const std::string& name); 

public:
	/*! \brief The mandatory alignment for all stack variables, in bytes */
	uint64_t stackAlignment;
	
private:
	MemoryRegionMap      _regions;
	BoundVariableMap     _variables;
	
};

typedef ApplicationBinaryInterface::FixedAddressRegion    FixedAddressRegion;
typedef ApplicationBinaryInterface::BoundVariable         BoundVariable;
typedef ApplicationBinaryInterface::RegisterBoundVariable RegisterBoundVariable;
typedef ApplicationBinaryInterface::MemoryRegion          MemoryRegion;

}

}

