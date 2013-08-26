/*! \file   Lexer.h
	\date   April 9, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Lexer class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr { namespace parser { class LexerEngine; } }

// Standard Library Includes
#include <string>
#include <list>

namespace vanaheimr
{

namespace parser
{

/*! \brief A generic lexer */
class Lexer
{
public:
	typedef std::list<std::string> StringList;

public:
	Lexer();
	~Lexer();

public:
	Lexer(const Lexer& );
	Lexer& operator=(const Lexer&) = delete;

public:
	/*! brief Set the stream being lexed */
	void setStream(std::istream* stream);	

public:
	/*! \brief Add a rule for lexing whitespace */
	void addWhitespaceRules(const std::string& whitespaceCharacters);	
	
	/*! \brief Add a set of rules for lexing tokens */
	void addTokens(const StringList& regexes);
	
	/*! \brief Define a regular expression for a token */
	void addTokenRegex(const std::string& regex);

public:
	std::string peek();
	std::string location() const;
	std::string nextToken();
	bool hitEndOfStream() const;

	bool scan(const std::string& token);
	void scanThrow(const std::string& token);
	bool scanPeek(const std::string& token);

public:
	size_t   line() const;
	size_t column() const;

public:
	void reset();
	void checkpoint();
	void restoreCheckpoint();
	void discardCheckpoint();

private:
	LexerEngine* _engine;

};

}

}

