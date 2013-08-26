/*! \file   MachineModel.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineModel class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/machine/interface/RegisterFile.h>
#include <vanaheimr/machine/interface/Operation.h>

// Standard Library Includes
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

// Forward Declarations
namespace vanaheimr { namespace machine { class PhysicalRegister; } }
namespace vanaheimr { namespace machine { class TranslationTable; } }

namespace vanaheimr
{

/*! \brief A namespace for the Vanaheimr machine model */
namespace machine
{

/*! \brief A model of a vanaheimr processor */
class MachineModel
{
public:
	typedef unsigned int             RegisterId;
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Construct a machine model */
	MachineModel(const std::string& name = "Vanaheimr");
	/*! \brief Virtual Destructor */
	virtual ~MachineModel();

public:
	MachineModel(const MachineModel&) = delete;
	const MachineModel& operator=(const MachineModel&) = delete;

public:
	/*! \brief Get the named physical register */
	const PhysicalRegister* getPhysicalRegister(RegisterId id) const;
	/*! \brief Get the named physical operation */
	const Operation* getOperation(const std::string& name) const;

public:
	/*! \brief Get the total register count */
	unsigned int totalRegisterCount() const;

public:
	const TranslationTable* translationTable() const;

public:
	/*! \brief Add a physical operation */
	void addOperation(const Operation&);

public:
	/*! \brief Configure the machine model with a set of options */
	virtual void configure(const StringVector& options);

public:
	/*! \brief Clone the machine model */
	virtual MachineModel* clone() const = 0;

public:
	const std::string name;

protected:
	void addRegisterFile(const std::string& name, unsigned int registers);

protected:
	typedef std::unordered_map<unsigned int,
		const PhysicalRegister*> RegisterMap;
	typedef std::map<std::string, RegisterFile> RegisterFileMap;
	typedef std::unordered_map<std::string, Operation> OperationMap;

protected:
	RegisterMap     _idToRegisters;
	RegisterFileMap _registerFiles;
	OperationMap    _machineOperations;

protected:
	TranslationTable* _translationTable;

};

}

}


