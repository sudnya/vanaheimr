/*! \file   TypeParser.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The header file for the TypeParser class.
*/

#pragma once

// Standard Library Includes
#include <istream>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler;     } }
namespace vanaheimr { namespace ir       { class Type;         } }
namespace vanaheimr { namespace parser   { class TypeAliasSet; } }
namespace vanaheimr { namespace parser   { class Lexer;        } }

namespace vanaheimr
{

namespace parser
{

/*! \brief A class for parsing a type from a stream */
class TypeParser
{
public:
	typedef compiler::Compiler Compiler;
	
public:
	TypeParser(Compiler* c, const TypeAliasSet* a = nullptr);
	~TypeParser();

public:
	            TypeParser(const TypeParser&) = delete;
	TypeParser&  operator=(const TypeParser&) = delete;

public:
	void parse(std::istream* stream);
	void parse(Lexer* _lexer);
	
public:
	const ir::Type* parsedType() const;

private:
	// High Level Parser methods
	ir::Type* _parseType();

	ir::Type* _parseFunction();
	ir::Type* _parseFunction(const ir::Type* base);
	ir::Type* _parseStructure();
	ir::Type* _parseArray();
	ir::Type* _parsePrimitive();
	ir::Type* _parseTypeAlias();

private:
	ir::Type* _getTypeAlias(const std::string& token);

private:
	Compiler*    _compiler;
	ir::Type*    _parsedType;
	
	const TypeAliasSet* _typedefs;

private:
	Lexer* _lexer;
	
};

}

}


