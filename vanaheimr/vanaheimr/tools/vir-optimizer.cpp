/*! \file   vir-optimizer.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday May 7, 2012
	\brief  The source file for the vir-optimizer tool.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassManager.h>
#include <vanaheimr/transforms/interface/PassFactory.h>

#include <vanaheimr/parser/interface/LLVMParser.h>

#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace vanaheimr
{

static void optimizeModule(ir::Module* module, const std::string& optimizations)
{
	auto optimizationList = hydrazine::split(optimizations, ",");
	
	transforms::PassManager manager(module);
	
	for(auto optimization : optimizationList)
	{
		auto pass = transforms::PassFactory::createPass(optimization);

		if(pass == nullptr)
		{
			throw std::runtime_error("Failed to create pass named '"
				+ optimization + "'");
		}

		manager.addPass(pass);
	}
	
	manager.runOnModule();
}

static ir::Module* loadBinaryModule(const std::string& inputFileName)
{
	std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
	
	std::ifstream virFile(inputFileName.c_str(), mode);
	
	if(!virFile.is_open())
	{
		std::cerr << "VIR Optimizer Failed: could not open VIR bytecode file '"
			<< inputFileName << "' for reading.\n"; 
	}
	
	try
	{
		as::BinaryReader reader;

		return reader.read(virFile, inputFileName);
	}
	catch(const std::exception& e)
	{
		std::cerr << "VIR Optimizer Failed: binary reading failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
	}

	return nullptr;
}

static ir::Module* loadAssemblyModule(const std::string& inputFileName)
{
	try
	{
		parser::LLVMParser parser(compiler::Compiler::getSingleton());

		parser.parse(inputFileName);
	
		return &*compiler::Compiler::getSingleton()->getModule(inputFileName);
	}
	catch(const std::exception& e)
	{
		std::cerr << "VIR Optimizer Failed: llvm parsing failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
	}

	return nullptr;
}

static std::string getExt(const std::string& path)
{
	auto segments = hydrazine::split(path, ".");

	if(!segments.empty()) return segments.back();

	return "";
}

static bool isAssembly(const std::string& inputFileName)
{
	return getExt(inputFileName) == "llvm";
}

static ir::Module* loadModule(const std::string& inputFileName)
{
	if(isAssembly(inputFileName))
	{
		return loadAssemblyModule(inputFileName);
	}

	return loadBinaryModule(inputFileName);

}

static void optimize(const std::string& inputFileName,
	const std::string& outputFileName,
	const std::string& optimizations)
{	
	
	ir::Module* module = loadModule(inputFileName);

	if(module == nullptr) return;
	
	try
	{
		optimizeModule(module, optimizations);
	}
	catch(const std::exception& e)
	{
		std::cerr << "VIR Optimizer Failed: optimization failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 

		return;
	}
	
	std::ios_base::openmode oMode = std::ios_base::out | std::ios_base::binary;	
	
	std::ofstream outputVirFile(outputFileName.c_str(), oMode);
	
	if(!outputVirFile.is_open())
	{
		std::cerr << "ObjDump Failed: could not open VIR file '"
			<< outputFileName << "' for writing.\n"; 
		
		return;
	}
	
	try
	{
		module->writeBinary(outputVirFile);
	}
	catch(const std::exception& e)
	{
		std::cerr << "ObjDump Failed: binary writing failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		return;
	}
	
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string virFileName;
	std::string outputFileName;
	std::string optimizations;

	bool verbose = false;

	parser.description("This program reads in a VIR binary, optimizes it, "
		"and writes it out again a new binary.");

	parser.parse("-i", "--input" ,  virFileName,
		"", "The input VIR file path.");
	parser.parse("-o", "--output",  outputFileName,
		"", "The output VIR file path.");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");
	parser.parse("", "--optimizations",  optimizations,
		"", "Comma separated list of optimizations (ConvertToSSA).");
	parser.parse();

	if(verbose)
	{
		hydrazine::enableAllLogs();
	}
	
	vanaheimr::optimize(virFileName, outputFileName, optimizations);

	return 0;
}

