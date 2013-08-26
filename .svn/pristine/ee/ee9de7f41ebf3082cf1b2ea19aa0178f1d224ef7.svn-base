/*! \file   Lexer.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday September 12, 2011
	\brief  The header file for the Lexer class.
*/

#pragma once

#define threads      128
#define ctas         120
#define transactions 6

/*! \brief A namespace for VIR assembler related classes and functions */
namespace assembler
{

__device__ Lexer::Lexer(util::File* file)
: _file(file), _fileData(0), _fileDataSize(0), _splitters(0),
	_transposedFileData(0), _transposedTokens(0), _tokens(0), _tokenCount(0)
{

}

__device__ Lexer::~Lexer()
{
	delete _file;
	delete[] _fileData;
	delete[] _splitters;
	delete[] _transposedFileData;
	delete[] _transposedTokens;
	delete[] _tokens;
}

__device__ const Lexer::Token* Lexer::tokenStream() const
{
	return _tokens;
}

__device__ size_t Lexer::tokenCount() const
{
	return _tokenCount;
}

__device__ void Lexer::lex()
{
	_fileDataSize       = _file->size();
	_fileData           = new char[_fileDataSize];
	_transposedFileData = new char[_fileDataSize];

	const unsigned int threadCount = ctas * threads;

	_splitters        = new Splitter[threadCount];
	_transposedTokens = new Token[_fileDataSize];
	_tokens           = new Token[_fileDataSize];
	
	util::HostReflection::launch(ctas, threads, "Lexer::findSplitters", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::transposeStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::lexCharacterStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::transposeCharacterStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::gatherTokenStreams", this);
	util::HostReflection::launch(1, 1,
		"Lexer::cleanup", this);
}

__device__ void Lexer::findSplitters()
{
	size_t totalThreads = threads * ctas;
	size_t blockSize = _fileDataSize / totalThreads;
	size_t threadId  = util::threadId();
	
	char*  startingPoint = _fileData + (threadId + 1) * blockSize;
	char*  fileEnd       = _fileData + _fileDataSize;
	
	startingPoint = (totalThreads - 1) == threadId ? fileEnd : startingPoint;
	
	for(; startingPoint < fileEnd; ++startingPoint)
	{
		if(*startingPoint == '\n') break;
	}
	
	_splitters[threadId] = startingPoint;
}

__device__ void Lexer::transposeStreams()
{
	Splitter splitter = _getDataSplitter();
	
	const util::Config config = {threads, ctas, 2 * threads};
	
	util::transpose<char, config>(splitter.begin, splitter.end,
		_transposedFileData);
}

__device__ bool isWhitespace(char character)
{
	return character == ' '  ||
		   character == '\t' ||
}

__device__ static Token nextState(State& state, char character)
{
	Token result = InvalidToken;

	// End a token with whitespace (a separator)
	if(isWhitespace(character))
	{
		if(state == Entry)
		{
			return InvalidToken;
		}

		State temp = state;
		state = Entry;
		return getTokenForState(state);
	}

	// produce a token and do not update the state
	if(state == Entry)
	{
		result = tryLexingSingleCharacter(character);
	}

	// a state update is required
	if(result == InvalidToken)
	{
		result = tryLexingComplex(state, character);	
	}
		
	return result;
}

__device__ static void lex(Token* tokens, unsigned int& bufferIndex,
	State& state, const char* position)
{
	unsigned int base = util::threadId() * localBufferEntries;
	
	Token token = nextState(state, *position);
	
	if(token != InvalidToken)
	{
		tokens[bufferIndex++ + base] = token;
	}
}

__device__ void Lexer::lexCharacterStreams()
{
	State state = Entry;

	const char* begin = _transposedFileData + util::threadId();
	const char* end   = _transposedFileData + _fileDataSize;

	const unsigned int bufferSize = threads * localBufferEntries;

	__shared__ Token buffer[bufferSize];
	__shared__ Token transposed[bufferSize];
	__shared__ unsigned int tokensGenerated[threads];
	__shared__ unsigned int tokenOffsets[threads];

	const util::Config config = {threads, ctas, bufferSize};
	
	for(const char* character = begin; character < end; )
	{
		unsigned int bufferIndex = 0;
		
		// lex locally into the buffer
		for(; character < end; ++character)
		{
			while(bufferIndex < bufferSize)
			{
				lex(buffer, bufferIndex, character);
			}
		}
		
		__syncthreads();
		
		// transpose the buffers 
		util::transposeShared<Token, config>(buffer, buffer + bufferSize,
			transposed);
		
		__syncthreads();
		
		// flush the buffers out
		for(unsigned int i = 0; i < threads; ++i)
		{
			unsigned int size   = tokensGenerated[i];
			unsigned int offset = tokenOffsets[i];
			
			if(util::threadId() < size)
			{
				tokenStreams[offset + util::threadId()] =
					transposed[i * threads + util::threadId()];
			}
		}
	}
}

__device__ void Lexer::gatherTokenStreams()
{
	// each CTA gets to gather from a single token stream
	TokenSplitter input  = _getTokenStreamInput();
	TokenSplitter output = _getTokenStreamOutput();
	
	assert(input.end - input.begin == output.end - output.begin);
	
	const util::Config config = {threads, ctas, 0};
	
	// cta memcpy
	util::ctaCopy<Token, config>(input.begin, input.end, output.begin);
}

__device__ void Lexer::cleanup()
{
	delete[] _fileData;
	_fileData = 0;
	
	delete[] _transposedFileData;
	_trabnsposedFileData = 0;

	delete[] _splitters; 
	_splitters = 0;

	delete[] _transposedFileData;
	_transposedFileData = 0;

	delete[] _transposedTokens;
	_transposedTokens = 0;

	delete[] _tokenStreams;
	_tokenStreams = 0;
}

}

