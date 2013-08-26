/*! \file   Pass.h
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Tuesday September 15, 2009
	\brief  The header file for the Pass class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <list>

// Forward Declarations
namespace vanaheimr { namespace ir         { class Function;    } }
namespace vanaheimr { namespace ir         { class Module;      } }
namespace vanaheimr { namespace ir         { class BasicBlock;  } }
namespace vanaheimr { namespace transforms { class PassManager; } }
namespace vanaheimr { namespace analysis   { class Analysis;    } }

namespace vanaheimr
{

namespace transforms
{
/*! \brief A class modeled after the LLVM notion of an optimization pass.  
	Allows different transformations to be applied to modules */
class Pass
{
public:
	/*! \brief For virtual classes, the type of pass */
	enum Type
	{
		ImmutablePass,
		ModulePass,
		ImmutableFunctionPass,
		FunctionPass,
		BasicBlockPass,
		InvalidPass
	};
	
	typedef std::vector<std::string> StringVector;
	
	// Typedef commonly used base classes into this namespace
	typedef analysis::Analysis Analysis;
	typedef ir::Function       Function;
	typedef ir::Module         Module;
	typedef ir::BasicBlock     BasicBlock;
	
	
public:
	/*! \brief The type of this pass */
	const Type type;
	
	/*! \brief What types of analysis routines does the pass require? */
	const StringVector analyses;
	
	/*! \brief The name of the pass */
	const std::string name;
	
	/*! \brief Classes of passes that this pass falls into
		(e.g. register-allocators) */
	const StringVector classes;

public:
	/*! \brief The default constructor sets the type */
	Pass(Type t = InvalidPass, const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~Pass();
	
public:
	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& type);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& type) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again for other applications */
	void invalidateAnalysis(const std::string& type);

public:
	/*! \brief Get a previously run pass by name */
	Pass* getPass(const std::string& name);

	/*! \brief Get a previously run pass by name (const) */
	const Pass* getPass(const std::string& name) const;

public:
	/*! \brief Get a list of passes that this pass depends on */
	virtual StringVector getDependentPasses() const;

public:
	/*! \brief Configure the pass given a list of options */
	virtual void configure(const StringVector& options);

public:
	/*! \brief Clone the pass */
	virtual Pass* clone() const = 0;

public:
	/*! \brief Report the name of the pass */
	std::string toString() const;

public:
	/*! \brief Set the pass manager used to supply dependent analyses */
	void setPassManager(PassManager* m);

private:
	PassManager* _manager;
	
};


/*! \brief A pass that generates information about a 
	program without modifying it, used to generate data structures */
class ImmutablePass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ImmutablePass(const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~ImmutablePass();
	
public:
	/*! \brief Run the pass on a specific module */
	virtual void runOnModule(const Module& m) = 0;
};

/*! \brief A pass over an entire module */
class ModulePass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ModulePass(const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~ModulePass();
	
public:
	/*! \brief Run the pass on a specific module */
	virtual void runOnModule(Module& m) = 0;		
};

/*! \brief A pass over a single function in a module */
class FunctionPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	FunctionPass(const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~FunctionPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const Module& m);
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnFunction(Function& f) = 0;		
	/*! \brief Finalize the pass */
	virtual void finalize();
};

/*! \brief An immutable pass over a single function in a module */
class ImmutableFunctionPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ImmutableFunctionPass(const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~ImmutableFunctionPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const Module& m);
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnFunction(const Function& k) = 0;		
	/*! \brief Finalize the pass */
	virtual void finalize();
};

/*! \brief A pass over a single basic block in a function */
class BasicBlockPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	BasicBlockPass(const StringVector& analyses = StringVector(),
		const std::string& n = "",
		const StringVector& classes = StringVector());
	/*! \brief Virtual destructor */
	virtual ~BasicBlockPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const Module& m);
	/*! \brief Initialize the pass using a specific function */
	virtual void initialize(const Function& m);
	/*! \brief Run the pass on a specific function in the module */
	virtual void runOnBlock(BasicBlock& b) = 0;		
	/*! \brief Finalize the pass on the function */
	virtual void finalizeFunction();
	/*! \brief Finalize the pass on the module */
	virtual void finalize();
};

typedef std::list<Pass*> PassList;

}

}

