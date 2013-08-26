/*! \file   MachineModelFactory.h
	\date   Tuesday January 15, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the MachineModelFactory class.
	
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class MachineModel; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief Used to create passes by name */
class MachineModelFactory
{
public:
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Create a machine model object from the specified name */
	static MachineModel* createMachineModel(const std::string& name,
		const StringVector& options = StringVector());

public:
	/*! \brief Create the default machine model */
	static MachineModel* createDefaultMachine();

public:
	/*! \brief Register a machine model with the factory.
	
		The object is copied by the machine model
	*/
	static void registerMachineModel(const MachineModel*);

};

}

}

