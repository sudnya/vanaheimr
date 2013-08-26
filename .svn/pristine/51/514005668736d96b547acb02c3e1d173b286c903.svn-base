/*! \file   PassManager.h
	\date   Thursday September 16, 2010
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the PassManager class
*/

#pragma once

// Standard Library Includes
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <list>

// Forward Declarations
namespace vanaheimr { namespace analysis   { class Analysis; } }
namespace vanaheimr { namespace ir         { class Module;   } }
namespace vanaheimr { namespace ir         { class Function; } }
namespace vanaheimr { namespace transforms { class Pass;     } }

namespace vanaheimr
{

namespace transforms
{

/*! \brief A class to orchestrate the execution of many passes */
class PassManager
{
public:
	typedef analysis::Analysis Analysis;
	typedef ir::Module         Module;
	typedef ir::Function       Function;

public:
	/*! \brief A map from analysis id to an up to date copy */
	typedef std::unordered_map<std::string, Analysis*> AnalysisMap;
	
	typedef std::vector<Pass*> PassVector;
	typedef std::list<PassVector> PassWaveList;

public:
	/*! \brief The constructor creates an empty pass manager associated
		with an existing Module.  
		
		The module is not owned by the PassManager.
		
		\param module The module that this manager is associated with.
	*/
	explicit PassManager(Module* module);
	~PassManager();
		
public:
	/*! \brief Adds a pass that needs to be eventually run
	
		The pass is now owned by the manager.
	
		\param pass The pass being added
	 */
	void addPass(Pass* pass);
	
	/*! \brief Adds an explicit dependence between pass types
	
		The dependence relationship is:
		
			dependentPassName <- passName
			
		or:
			
			dependentPassName depends on passName
	 */
	void addDependence(const std::string& dependentPassName,
		const std::string& passName);
	
	/*! \brief Clears all added passes */
	void clear();
	
public:
	/*! \brief Runs passes on a specific Function contained in the module.
	
		\param name The name of the function to run all passes on.
	*/
	void runOnFunction(const std::string& name);

	/*! \brief Runs passes on a specific Function.
	
		\param function The function to run all passes on.
	*/
	void runOnFunction(Function& function);
	
	/*! \brief Runs passes on the entire module. */
	void runOnModule();

public:
	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& type);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& type) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again the next time 'get' is called */
	void invalidateAnalysis(const std::string& type);

public:
	/*! \brief Get a previously run pass by name */
	Pass* getPass(const std::string& name);

	/*! \brief Get a previously run pass by name (const) */
	const Pass* getPass(const std::string& name) const;

public:	
	/*! \brief Disallow the copy constructor */
	PassManager(const PassManager&) = delete;
	/*! \brief Disallow the assignment operator */
	const PassManager& operator=(const PassManager&) = delete;

private:
	typedef std::multimap<std::string, std::string> DependenceMap;
	typedef std::unordered_map<std::string, Pass*> PassMap;
	typedef std::vector<std::string> StringVector;

private:
	PassWaveList _schedulePasses();
	StringVector _getAllDependentPasses(Pass* p);
	Pass*        _findPass(const std::string& name);

private:
	PassVector    _passes;
	Module*       _module;
	Function*     _function;
	AnalysisMap*  _analyses;
	PassVector    _ownedTemporaryPasses;
	DependenceMap _extraDependences;
	PassMap       _previouslyRunPasses;
};

}

}

