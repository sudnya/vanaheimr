/*! \file   ConstantValueParser.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday March 4 13, 2013
	\brief  The header file for the ConstantValueParser class.
*/

#pragma once

// Standard Library Includes
#include <istream>

// Forward Declarations
namespace vanaheimr { namespace ir { class Constant; } }
namespace vanaheimr { namespace ir { class Type;     } }

namespace vanaheimr { namespace parser { class Lexer; } }

namespace vanaheimr
{

namespace parser
{

/*! \brief A class for parsing a type from a string */
class ConstantValueParser
{
public:
	explicit ConstantValueParser(std::istream* stream);
	explicit ConstantValueParser(Lexer* lexer);
	~ConstantValueParser();

public:
	            ConstantValueParser(const ConstantValueParser&) = delete;
	ConstantValueParser&  operator=(const ConstantValueParser&) = delete;

public:
	void parse();
	void parse(const ir::Type* type);

public:
	const ir::Constant* parsedConstant() const;

private:
	// Specialized Parsing
	ir::Constant* _parseConstant();
	ir::Constant* _parseConstant(const ir::Type* type);

	ir::Constant* _parseIntegerConstant();
	ir::Constant* _parseFloatingPointConstant();
	ir::Constant* _parseStringConstant();

private:
	ir::Constant* _parsedConstant;
	
private:
	Lexer*        _lexer;
	std::istream* _stream;
	
};

}

}


