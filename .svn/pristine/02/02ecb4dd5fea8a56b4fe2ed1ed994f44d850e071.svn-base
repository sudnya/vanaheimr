/*! 	\file   vir-objdump.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday May 7, 2012
	\brief  The source file for the vir-objdump tool.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Module.h>

#include <vanaheimr/asm/interface/BinaryReader.h>

// Hydrazine Includes
#include <hydrazine/implementation/ArgumentParser.h>

// Standard Library Includes
#include <fstream>

namespace vanaheimr
{

void dump(const std::string& name)
{	
	std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
	
	std::ifstream virFile(name.c_str(), mode);
	
	if(!virFile.is_open())
	{
		std::cerr << "ObjDump Failed: could not open VIR file '"
			<< name << "' for reading.\n"; 
		return;
	}
	
	ir::Module* module = 0;

	try
	{
		as::BinaryReader reader;

		module = reader.read(virFile, name);
	
		module->writeAssembly(std::cout);
	}
	catch(const std::exception& e)
	{
		std::cerr << "ObjDump Failed: binary reading failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		return;
	}
	
	delete module;
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string virFileName;

	parser.description("This program prints out an assembly representation of a VIR binary.");

	parser.parse("-i", "--input",  virFileName, "", "The input VIR file path.");
	parser.parse();
	
	vanaheimr::dump(virFileName);

	return 0;
}

