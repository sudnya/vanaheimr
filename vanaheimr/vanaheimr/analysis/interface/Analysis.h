/*! \file   Analysis.h
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Saturday May 7, 2011
	\brief  The header file for the Analysis class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace transforms { class PassManager; } }
namespace vanaheimr { namespace ir         { class Function;    } }
namespace vanaheimr { namespace ir         { class Module;      } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief An analysis that can be constructed for aiding IR transforms */
class Analysis
{
public:
	enum Type
	{
		FunctionAnalysis,
		ModuleAnalysis
	};

public:
	typedef std::vector<std::string> StringVector;

	typedef transforms::PassManager PassManager;
	typedef ir::Function            Function;
	typedef ir::Module              Module;

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	Analysis(Type t, const std::string& name = "",
		const StringVector& dependentAnalyses = StringVector());

	virtual ~Analysis();

public:
	/*! \brief The type of analysis */
	const Type type;

	/*! \brief The name of the analysis */
	const std::string name;

	/*! \brief The analysis dependencies */
	const StringVector required;

public:
	/*! \brief Set the pass manager used to supply dependent analyses */
	void setPassManager(PassManager* m);
	
	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& name);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& name) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again for other users */
	void invalidateAnalysis(const std::string& name);

public:
	virtual void configure(const StringVector& );

private:
	PassManager* _manager;

};

/*! \brief An analysis over a single kernel */
class FunctionAnalysis : public Analysis
{

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	FunctionAnalysis(const std::string& name = "",
		const StringVector& dependentAnalyses = StringVector());

public:
	virtual void analyze(Function& function) = 0;

};

/*! \brief An analysis over a complete module */
class ModuleAnalysis : public Analysis
{

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	ModuleAnalysis(const std::string& name = "",
		const StringVector& dependentAnalyses = StringVector());

public:
	virtual void analyze(Module& module) = 0;

};

}

}

