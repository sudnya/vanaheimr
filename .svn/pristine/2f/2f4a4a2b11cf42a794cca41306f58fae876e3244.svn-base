/*! \file   TypeParser.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The source file for the TypeParser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/TypeParser.h>
#include <vanaheimr/parser/interface/TypeAliasSet.h>
#include <vanaheimr/parser/interface/Lexer.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace vanaheimr
{

namespace parser
{

TypeParser::TypeParser(Compiler* c, const TypeAliasSet* a)
: _compiler(c), _parsedType(nullptr), _typedefs(a), _lexer(nullptr)
{

}

TypeParser::~TypeParser()
{

}

void TypeParser::parse(std::istream* stream)
{
	_parsedType = nullptr;

	_lexer = new Lexer();
	_lexer->setStream(stream);
	
	_parsedType = _parseType();

	delete _lexer; _lexer = nullptr;
}

void TypeParser::parse(Lexer* lexer)
{
	_parsedType = nullptr;

	_lexer = lexer;
	
	_parsedType = _parseType();

	_lexer = nullptr;
}

const ir::Type* TypeParser::parsedType() const
{
	assert(_parsedType != nullptr);
	
	return _parsedType;
}

static bool isFunction(const std::string& token)
{
	return token.find("(") == 0;
}

static bool isStructure(const std::string& token)
{
	return token.find("{") == 0;
}

static bool isArray(const std::string& token)
{
	return token.find("[") == 0;
}

static bool isPointer(const std::string& token)
{
	return token.find("*") == 0;
}

static bool isVariadic(const std::string& token)
{
	return token == "...";
}

static bool isTypeAlias(const std::string& token)
{
	return token.find("%") == 0;
}

static bool isOpaqueType(const std::string& token)
{
	return token.find("opaque") == 0;
}

static bool isPrimitive(compiler::Compiler* compiler, const std::string& token)
{
	hydrazine::log("TypeParser::Parser") << "Checking if " << token
		<< " is a primitive type.\n";
	
	ir::Type* primitive = compiler->getType(token);

	if(primitive == nullptr) return false;

	return primitive->isPrimitive() || primitive->isBasicBlock();
}

ir::Type* TypeParser::_parseType()
{
	std::string nextToken = _lexer->peek();
	
	ir::Type* type = nullptr;
	
	if(isFunction(nextToken))
	{
		type = _parseFunction();
	}
	else if(isStructure(nextToken))
	{
		type = _parseStructure();
	}
	else if(isPrimitive(_compiler, nextToken))
	{
		type = _parsePrimitive();
		
		nextToken = _lexer->peek();
		
		if(isFunction(nextToken))
		{
			type = _parseFunction(type);
		}
	}
	else if(isArray(nextToken))
	{
		type = _parseArray();
	}
	else if(isVariadic(nextToken))
	{
		_lexer->scan("...");
		type = *_compiler->getOrInsertType(ir::VariadicType(_compiler));
	}
	else if(isTypeAlias(nextToken))
	{
		type = _parseTypeAlias();
	}
	else if(isOpaqueType(nextToken))
	{
		_lexer->scan("opaque");
		type = *_compiler->getOrInsertType(ir::OpaqueType(_compiler));
	}

	nextToken = _lexer->peek();

	while(isPointer(nextToken))
	{
		_lexer->scan("*");
		type = *_compiler->getOrInsertType(ir::PointerType(_compiler, type));
	
		nextToken = _lexer->peek();
	}
	
	if(type == nullptr)
	{
		throw std::runtime_error("Failed to parse type.");
	}
	
	hydrazine::log("TypeParser::Parser") << "Parsed type " << type->name
		<< ".\n";
	
	return type;
}

ir::Type* TypeParser::_parseFunction()
{
	ir::Type* returnType = nullptr;
	ir::Type::TypeVector argumentTypes;

	if(!_lexer->scan("("))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}

	std::string closeBrace = _lexer->peek();

	if(closeBrace != ")")
	{
		returnType = _parseType();
	}

	if(!_lexer->scan(")"))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	if(!_lexer->scan("("))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}
	
	closeBrace = _lexer->peek();

	if(closeBrace != ")")
	{
		do
		{
			argumentTypes.push_back(_parseType());
		
			std::string comma = _lexer->peek();
			
			if(comma == ",")
			{
				_lexer->scan(",");
			}
			else
			{
				break;
			}
		}
		while(true);
	}

	if(!_lexer->scan(")"))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	return *_compiler->getOrInsertType(ir::FunctionType(
		_compiler, returnType, argumentTypes));
}

ir::Type* TypeParser::_parseFunction(const ir::Type* returnType)
{
	ir::Type::TypeVector argumentTypes;

	if(!_lexer->scan("("))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}
       
	auto closeBrace = _lexer->peek();

	if(closeBrace != ")")
	{
		do
		{
			argumentTypes.push_back(_parseType());
	       
			std::string comma = _lexer->peek();
		       
			if(comma == ",")
			{
				_lexer->scan(",");
			}
			else
			{
				break;
			}
		}
		while(true);
	}

	if(!_lexer->scan(")"))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	return new ir::FunctionType(_compiler, returnType, argumentTypes);
}


ir::Type* TypeParser::_parseStructure()
{
	if(!_lexer->scan("{"))
	{
		throw std::runtime_error("Failed to parse structure "
			"type, expecting '{'.");
	}

	ir::Type::TypeVector types;

	auto closeBrace = _lexer->peek();

	if(closeBrace != "}")
	{
		do
		{
			types.push_back(_parseType());
		
			std::string comma = _lexer->peek();
			
			if(comma == ",")
			{
				_lexer->scan(",");
			}
			else
			{
				break;
			}
		}
		while(true);
	}
	
	if(!_lexer->scan("}"))
	{
		throw std::runtime_error("Failed to parse structure "
			"type, expecting '}'.");
	}

	return *_compiler->getOrInsertType(ir::StructureType(_compiler, types));
}

static bool isNumeric(char c)
{
	return c == '0' || c == '2' || c == '3' || c == '4' || c == '5' ||
		c == '6' || c == '7' || c == '8' || c == '9' || c == '1';
}

static bool isInteger(const std::string& integer)
{
	for(auto character : integer)
	{
		if(!isNumeric(character)) return false;
	}
	
	return true;
}

static unsigned int parseInteger(const std::string& integer)
{
	if(!isInteger(integer))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting an integer.");
	}
	
	std::stringstream stream(integer);
	
	unsigned int value = 0;
	
	stream >> value;
	
	return value;
}

ir::Type* TypeParser::_parseArray()
{
	if(!_lexer->scan("["))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting '['.");
	}
	
	std::string dimensionToken = _lexer->nextToken();
	
	unsigned int dimension = parseInteger(dimensionToken);
	
	if(!_lexer->scan("x"))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting 'x'.");
	}

	auto base = _parseType();
	
	if(!_lexer->scan("]"))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting ']'.");
	}
	
	return *_compiler->getOrInsertType(ir::ArrayType(_compiler,
		base, dimension));
}

ir::Type* TypeParser::_parsePrimitive()
{
	if(_lexer->hitEndOfStream())
	{
		throw std::runtime_error("Hit end of stream while "
			"searching for primitive type.");
	}

	return _compiler->getType(_lexer->nextToken());
}

ir::Type* TypeParser::_parseTypeAlias()
{
	if(!_lexer->scan("%"))
	{
		throw std::runtime_error("Failed to parse type alias, expecting '%'.");
	}
	
	if(_lexer->hitEndOfStream())
	{
		throw std::runtime_error("Hit end of stream while "
			"searching for primitive type.");
	}
	
	std::string token = _lexer->nextToken();
	
	auto alias = _getTypeAlias(token);

	if(alias == nullptr)
	{
		throw std::runtime_error("Failed to parse type alias, unknown "
			"typename '" + token + "'.");
	}

	return alias;
}

ir::Type* TypeParser::_getTypeAlias(const std::string& token)
{
	if(_typedefs == nullptr) return nullptr;

	auto type = _typedefs->getType(token);

 	if(type != nullptr) return *_compiler->getOrInsertType(*type);

	return *_compiler->getOrInsertType(ir::AliasedType(_compiler, token));
}

}

}


