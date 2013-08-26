/*! \file   ptx-to-vir-translator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday February 13, 2012
	\brief  The source file for the ptx-to-vir-translator tool.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Module.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/translation/interface/PTXToVIRTranslator.h>
#include <vanaheimr/translation/interface/OcelotToVIRTraceTranslator.h>

// Ocelot Includes
#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Standard Library Includes
#include <fstream>

namespace vanaheimr
{

static bool isTraceFile(const std::string& path)
{
	auto extension = hydrazine::split(path, ".").back();
	
	return extension == "trace";
}

static std::string translatePTX(const std::string& ptxFileName)
{
	// Load the PTX module
	::ir::Module ptxModule(ptxFileName);
	
	compiler::Compiler* virCompiler = compiler::Compiler::getSingleton();
	
	// Translate the PTX
	translation::PTXToVIRTranslator translator(virCompiler);
	
	try
	{
		translator.translate(ptxModule);
	}
	catch(const std::exception& e)
	{
		std::cerr << "Compilation Failed: PTX to VIR translation failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		
		throw;
	}
	
	return ptxFileName;
}

static std::string translateTrace(const std::string& traceFileName)
{
	// Translate the trace
	compiler::Compiler* virCompiler = compiler::Compiler::getSingleton();
	
	translation::OcelotToVIRTraceTranslator translator(virCompiler);
	
	try
	{
		translator.translate(traceFileName);
	}
	catch(const std::exception& e)
	{
		std::cerr << "Compilation Failed: Trace to VIR translation failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		
		throw;
	}
	
	return translator.translatedModuleName();	
}

/*! \brief Load a PTX module, translate it to VIR, output the result */
static void translate(const std::string& virFileName,
	const std::string& ptxFileName, bool binary)
{
	// is this a ptx or trace file?
	bool isTrace = isTraceFile(ptxFileName);

	std::string ptxModuleName;

	// Translate the ptx
	try
	{
		if(isTrace)
		{
			ptxModuleName = translateTrace(ptxFileName);
		}
		else
		{
			ptxModuleName = translatePTX(ptxFileName);
		}
	}
	catch(const std::exception& e)
	{
		return;
	}

	compiler::Compiler* virCompiler = compiler::Compiler::getSingleton();
	
	// Output the VIR module
	vanaheimr::compiler::Compiler::module_iterator virModule =
		virCompiler->getModule(ptxModuleName);
	assert(virModule != virCompiler->module_end());
	
	virModule->name = virFileName;

	std::ios_base::openmode mode = std::ios_base::out;

	if(binary)
	{
		mode |= std::ios_base::binary;
	}
	
	std::ofstream virFile(virFileName.c_str(), mode);
	
	if(!virFile.is_open())
	{
		std::cerr << "Compilation Failed: could not open VIR file '"
			<< virFileName << "' for writing.\n"; 
		return;
	}
	
	try
	{
		if(binary)
		{
			virModule->writeBinary(virFile);
		}
		else
		{
			virModule->writeAssembly(virFile);
		}
	}
	catch(const std::exception& e)
	{
		std::cerr << "Compilation Failed: binary writing failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		
		std::remove(virFileName.c_str());
	}
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string ptxFileName;
	std::string virFileName;
	bool writeBinary;

	parser.description("This program compiles a PTX file into a VIR binary.");

	parser.parse("-i", "--input",  ptxFileName, "", "The input PTX file path.");
	parser.parse("-o", "--output", virFileName,
		"", "The output VIR file path.");
	parser.parse("-b", "--use-binary-format", writeBinary,
		false, "Output a VIR binary "
		"bytecode file rather than an assembly file.");
	parser.parse();
	
	vanaheimr::translate(virFileName, ptxFileName, writeBinary);

	return 0;
}


