/*! \file   Compiler.h
	\date   Sunday February 12, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Compiler class.
	
*/

// Standard Library Includes
#include <vanaheimr/ir/interface/Module.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class Type;   } }

namespace vanaheimr
{

/*! \brief A namespace for core compiler components */
namespace compiler
{

/*! \brief The global compiler state for vanaheimr. */
class Compiler
{
public:
	typedef std::vector<ir::Type*> TypeVector;
	typedef std::list<ir::Module>  ModuleList;
	
	typedef TypeVector::iterator       iterator;
	typedef TypeVector::const_iterator const_iterator;

	typedef ModuleList::iterator       module_iterator;
	typedef ModuleList::const_iterator const_module_iterator;
	
public:
	Compiler();
	~Compiler();
	
public:
	Compiler( const Compiler&) = delete;
	Compiler& operator=(const Compiler&) = delete;	
	
public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	bool   empty() const;
	size_t size()  const;

public:
	module_iterator       module_begin();
	const_module_iterator module_begin() const;

	module_iterator       module_end();
	const_module_iterator module_end() const;

public:
	module_iterator newModule(const std::string& name);
	
public:
	module_iterator getModule(const std::string& name);
	const_module_iterator getModule(const std::string& name) const;

public:
	/*! \brief Lookup a type by name, return 0 if it doesn't exist */
	ir::Type*       getType(const std::string& name);
	/*! \brief Lookup a type by name, return 0 if it doesn't exist */
	const ir::Type* getType(const std::string& typeName) const;

public:
	/*! \brief Get the basic block type */
	const ir::Type* getBasicBlockType() const;

public:
	static Compiler* getSingleton();

private:
	TypeVector _types;
	ModuleList _modules;

};	

}

}


