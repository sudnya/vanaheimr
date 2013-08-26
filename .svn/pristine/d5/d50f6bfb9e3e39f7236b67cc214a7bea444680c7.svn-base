/*! \file   test-lexer.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday April 15, 2013
	\brief  The source file for the test-lexer test.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/Lexer.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace test
{

static void testLexer(const std::string& filename)
{	
	std::ifstream file(filename.c_str());
	
	if(!file.is_open())
	{
		std::cerr << "test-lexer FAILED: could not open input file '"
			<< filename << "' for reading.\n"; 
		
		return;
	}
	
	vanaheimr::parser::Lexer lexer;

	lexer.addWhitespaceRules(" \t\n");
	
	lexer.addTokenRegex("hello");
	lexer.addTokenRegex("he");
	lexer.addTokenRegex("[*]");

	lexer.setStream(&file);

	std::cout << "Lexed tokens: '" << lexer.nextToken() << "'";


	while(!lexer.hitEndOfStream())
	{
		std::cout << ", '" << lexer.nextToken() << "'";
	}
	
	std::cout << "\n";
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string filename;
	
	bool verbose = false;

	parser.description("This program reads in a a text file and attempts "
		"to lex it.");

	parser.parse("-i", "--input" ,  filename,
		"", "The input file path to be lexed.");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");
	parser.parse();

	if(verbose)
	{
		hydrazine::enableAllLogs();
	}
	
	test::testLexer(filename);

	return 0;
}

