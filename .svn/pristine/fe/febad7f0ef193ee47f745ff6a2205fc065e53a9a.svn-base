/*! \file   ApplicationBinaryInterface.cpp
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ApplicationBinaryInterface class.
*/

// Vanaheimr Includes
#include <vanaheimr/abi/interface/ApplicationBinaryInterface.h>
#include <vanaheimr/abi/interface/ArchaeopteryxABI.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace abi
{

ApplicationBinaryInterface::MemoryRegion::MemoryRegion(const std::string& n,
	unsigned int b, unsigned int a, unsigned int l, Binding bi)
: name(n), bytes(b), alignment(a), level(l), _binding(bi)
{

}

ApplicationBinaryInterface::MemoryBinding
	ApplicationBinaryInterface::MemoryRegion::binding() const
{
	return _binding;
}

bool ApplicationBinaryInterface::MemoryRegion::isRegister() const
{
	return binding() == Register;
}

bool ApplicationBinaryInterface::MemoryRegion::isFixed() const
{
	return binding() == Fixed;
}

bool ApplicationBinaryInterface::MemoryRegion::isIndirect() const
{
	return binding() == Indirect;
}

ApplicationBinaryInterface::RegisterBoundRegion::RegisterBoundRegion(
	const std::string& name, unsigned int bytes,
	unsigned int alignment, unsigned int level,
	const std::string& r)
: MemoryRegion(name, bytes, alignment, level, Register), registerName(r)
{

}

ApplicationBinaryInterface::FixedAddressRegion::FixedAddressRegion(
	const std::string& name, unsigned int bytes, unsigned int alignment,
	unsigned int level, uint64_t a)
: MemoryRegion(name, bytes, alignment, level, Fixed), address(a)
{

}

ApplicationBinaryInterface::IndirectlyAddressedRegion::IndirectlyAddressedRegion(
	const std::string& name, unsigned int bytes,
	unsigned int alignment, unsigned int level,
	const std::string& r, unsigned int o)
: MemoryRegion(name, bytes, alignment, level, Indirect), region(r), offset(o)
{

}

ApplicationBinaryInterface::BoundVariable::BoundVariable(
	const std::string& n, const ir::Type* t, Binding b)
: name(n), type(t), _binding(b)
{

}

ApplicationBinaryInterface::VariableBinding
	ApplicationBinaryInterface::BoundVariable::binding() const
{
	return _binding;
}

ApplicationBinaryInterface::RegisterBoundVariable::RegisterBoundVariable(
	const std::string& name, const ir::Type* type, const std::string& r)
: BoundVariable(name, type, Register), registerName(r)
{

}

ApplicationBinaryInterface::MemoryBoundVariable::MemoryBoundVariable(
	const std::string& name, const ir::Type* type, const std::string& r)
: BoundVariable(name, type, Memory), region(r)
{

}

ApplicationBinaryInterface::ApplicationBinaryInterface()
: stackAlignment(8)
{

}

ApplicationBinaryInterface::~ApplicationBinaryInterface()
{
	for(auto region : _regions)
	{
		delete region.second;
	}
	
	for(auto variable : _variables)
	{
		delete variable.second;
	}
}

bool ApplicationBinaryInterface::validate(std::string& message)
{
	// TODO

	return true;
}

ApplicationBinaryInterface::MemoryRegion*
	ApplicationBinaryInterface::findRegion(const std::string& name)
{
	auto region = _regions.find(name);
	
	if(region == _regions.end()) return nullptr;
	
	return region->second;
}

ApplicationBinaryInterface::BoundVariable*
	ApplicationBinaryInterface::findVariable(const std::string& name)
{
	auto variable = _variables.find(name);
	
	if(variable == _variables.end()) return nullptr;
	
	return variable->second;
}	

const ApplicationBinaryInterface::MemoryRegion* 
	ApplicationBinaryInterface::findRegion(const std::string& name) const
{
	auto region = _regions.find(name);
	
	if(region == _regions.end()) return nullptr;
	
	return region->second;
}

const ApplicationBinaryInterface::BoundVariable*
	ApplicationBinaryInterface::findVariable(const std::string& name) const
{
	auto variable = _variables.find(name);
	
	if(variable == _variables.end()) return nullptr;
	
	return variable->second;
}

ApplicationBinaryInterface::MemoryRegion* ApplicationBinaryInterface::insert(
	MemoryRegion* region)
{
	assert(findRegion(region->name) == nullptr);
	
	_regions.insert(std::make_pair(region->name, region));
	
	return region;
}

ApplicationBinaryInterface::BoundVariable* ApplicationBinaryInterface::insert(
	BoundVariable* variable)
{
	assert(findVariable(variable->name) == nullptr);
	
	_variables.insert(std::make_pair(variable->name, variable));

	return variable;
}

ApplicationBinaryInterface::const_region_iterator
	ApplicationBinaryInterface::regions_begin() const
{
	return _regions.begin();
}

ApplicationBinaryInterface::const_region_iterator
	ApplicationBinaryInterface::regions_end() const
{
	return _regions.end();
}

ApplicationBinaryInterface::const_variable_iterator
	ApplicationBinaryInterface::variables_begin() const
{
	return _variables.begin();
}

ApplicationBinaryInterface::const_variable_iterator
	ApplicationBinaryInterface::variables_end() const
{
	return _variables.end();
}

class ABISingleton
{
public:
	typedef util::LargeMap<std::string,
		const ApplicationBinaryInterface*> ABIMap;

public:
	ABISingleton();

public:
	const ApplicationBinaryInterface* getABI(const std::string& name);
	
private:
	ABIMap _abis;

};

ABISingleton::ABISingleton()
{
	_abis.insert(std::make_pair("archaeopteryx", getArchaeopteryxABI()));
}

const ApplicationBinaryInterface* ABISingleton::getABI(const std::string& name)
{
	auto namedABI = _abis.find(name);
	
	if(namedABI == _abis.end()) return nullptr;
	
	return namedABI->second;
}

static ABISingleton singleton;

const ApplicationBinaryInterface*
	ApplicationBinaryInterface::getABI(const std::string& name)
{
	return singleton.getABI(name);
}

}

}

