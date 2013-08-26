/*! \file   Lexer.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday September 12, 2011
	\brief  The header file for the Lexer class.
*/

#pragma once

/*! \brief A namespace for VIR assembler related classes and functions */
namespace assembler
{

/*! \brief An attempt at a fast data-parallel lexer for the VIR language 

	This intial attempt makes several language simplifications to ease the
	lexing process.  Specifically:
		1) Tokens may not cross new-line boundaries, including comments.
		2) Only one statement is allowed per line.

	Restriction 1) defines a list of short regions that may be parsed
		independently.  The goal is to filter these into thread private memory
		and then lex them independently.

	To start with, we need a definition of the VIR language.
	
	1) Comments begin at ';' characters and continue to the end of the line.
	2) There are about 50 valid tokens for instruction opcodes,
		modifiers, and types.
	
	Here is the general philosophy of the algorithm.
		1) The input is transposed at a stride such that each thread is given
			a long sequence of characters that can be fetched efficiently.
		2) Each thread streams through its sequence using a state transition
			table stored in shared memory, generating a sequence of tokens.
		3) The output token streams are concatenated together. 
*/
class Lexer
{
public:
	class Token
	{
	public:
		enum Type
		{
			
		};
	
	public:
		Type     type;
		uint64_t data;
	};


public:
	/*! \brief The constructor initializes itself from a file */
	__device__ Lexer(util::File* file);
	/*! \brief Destroy the lexer and any generated token stream */
	__device__ ~Lexer();

public:
	/*! \brief Get the generated token stream */
	__device__ const Token* tokenStream() const;
	/*! \brief Get the length of the generated token stream */
	__device__ size_t tokenCount() const;

public:
	/*! \brief Run the lexing pass */
	__device__ void lex();

public:
	/*! \brief Find splitters */
	__device__ void findSplitters();
	/*! \brief Transpose the character streams between the splitters */
	__device__ void transposeStreams();
	/*! \brief Produce token streams, lexing each stream independently */
	__device__ void lexCharacterStreams();
	/*! \brief Gather the token streams */
	__device__ void gatherTokenStreams();
	/*! \brief Cleanup allocated scratchpath memory */
	__device__ void cleanup();

private:
	/*! \brief The file being lexed */
	util::File* _file;
	/*! \brief The data buffer for the file */
	char* _fileData;
	size_t _fileDataSize;
	/*! \brief The set of splitters, one per thread */
	Splitter* _splitters;
	/*! \brief The transposed file data */
	char* _transposedFileData;
	/*! \brief The list of transposed tokens */
	Token* _tokenStreams;
	/*! \brief Finally, the compacted set of tokens */
	Token* _tokens;
	size_t _tokenCount;
};

}

