/*! \file   TypeParser.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The source file for the TypeParser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/ConstantValueParser.h>
#include <vanaheimr/parser/interface/Lexer.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Type.h>
#include <vanaheimr/ir/interface/Constant.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace vanaheimr
{

namespace parser
{


ConstantValueParser::ConstantValueParser(std::istream* stream)
: _parsedConstant(nullptr), _lexer(nullptr), _stream(stream)
{

}

ConstantValueParser::ConstantValueParser(Lexer* l)
: _parsedConstant(nullptr), _lexer(l), _stream(nullptr)
{

}

ConstantValueParser::~ConstantValueParser()
{
	delete _parsedConstant;
}

static void addRules(Lexer* lexer)
{
	lexer->addWhitespaceRules(" \t\r\n");

	lexer->addTokens({"\"*\""});
}

void ConstantValueParser::parse()
{
	delete _parsedConstant;

	if(_stream != nullptr)
	{
		_lexer = new Lexer();
		
		addRules(_lexer);
		
		_lexer->setStream(_stream);
	}
	
	try
	{
		_parsedConstant = _parseConstant();
	}
	catch(...)
	{
		if(_stream != nullptr) delete _lexer;
	
		throw;
	}
	
	if(_stream != nullptr) delete _lexer;
}

void ConstantValueParser::parse(const ir::Type* type)
{
	delete _parsedConstant;

	if(_stream != nullptr)
	{
		_lexer = new Lexer();
		
		addRules(_lexer);
		
		_lexer->setStream(_stream);
	}
	
	try
	{
		if(type->isInteger() || type->isFloatingPoint())
		{	
			_parsedConstant = _parseConstant();
		}
		else
		{
			_parsedConstant = _parseConstant(type);
		}
	}
	catch(...)
	{
		if(_stream != nullptr) delete _lexer;
	
		throw;
	}
	
	if(_stream != nullptr) delete _lexer;
}

const ir::Constant* ConstantValueParser::parsedConstant() const
{
	return _parsedConstant;
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

static bool isString(const std::string& string)
{
	if(string.size() < 2) return false;
	
	return string.front() == '\"' && string.back() == '\"';
}

static bool isFloatingPoint(const std::string& token)
{
	return !token.empty() && isNumeric(token[0]) && !isInteger(token);
}

ir::Constant* ConstantValueParser::_parseConstant()
{
	std::string nextToken = _lexer->peek();
	
	ir::Constant* constant = nullptr;
	
	if(isInteger(nextToken))
	{
		constant = _parseIntegerConstant();
	}
	else if(isFloatingPoint(nextToken))
	{
		constant = _parseFloatingPointConstant();
	}
	else if(isString(nextToken))
	{
		constant = _parseStringConstant();
	}
	
	if(constant == nullptr)
	{
		throw std::runtime_error("Failed to parse constant.");
	}

	hydrazine::log("ConstantValueParser::Parser") << "Parsed constant with type '"
		<< constant->type()->name << "'\n";
	
	return constant;
}

static ir::Constant* createZeroInitializer(const ir::Type* type)
{
	if(type->isInteger())
	{
		auto integer = static_cast<const ir::IntegerType*>(type);

		return new ir::IntegerConstant(0, integer->bits());
	}
	else if(type->isSinglePrecisionFloat())
	{
		return new ir::FloatingPointConstant(0.0f);
	}
	else if(type->isDoublePrecisionFloat())
	{
		return new ir::FloatingPointConstant(0.0);
	}
	else if(type->isStructure())
	{
		auto structure = new ir::StructureConstant(type);

		return structure;
	}

	assertM(false, "Zero initializer not implemented for " << type->name);

	return nullptr;
}

static bool isZeroInitializer(const std::string& token)
{
	return token == "zeroinitializer";
}

ir::Constant* ConstantValueParser::_parseConstant(const ir::Type* type)
{
	auto nextToken = _lexer->peek();

	ir::Constant* constant = nullptr;

	if(isZeroInitializer(nextToken))
	{
		_lexer->nextToken();
		constant = createZeroInitializer(type);
	}
	
	if(constant == nullptr)
	{
		throw std::runtime_error("Failed to parse constant.");
	}
	
	hydrazine::log("ConstantValueParser::Parser") << "Parsed constant with type '"
		<< constant->type()->name << "'\n";
	
	return constant;
}

static unsigned int parseInteger(const std::string& integer)
{
	std::stringstream stream(integer);
	
	unsigned int value = 0;
	
	stream >> value;
	
	hydrazine::log("ConstantValueParser::Parser") << " parsed integer constant '"
		<< value << "'\n";
	
	return value;
}

ir::Constant* ConstantValueParser::_parseIntegerConstant()
{
	return new ir::IntegerConstant(parseInteger(_lexer->nextToken()));
}

static double parseFloat(const std::string& floating)
{
	std::stringstream stream(floating);
	
	double value = 0.0;
	
	stream >> value;
	
	hydrazine::log("ConstantValueParser::Parser") << " parsed float constant '"
		<< value << "'\n";
	
	return value;
}

ir::Constant* ConstantValueParser::_parseFloatingPointConstant()
{
	return new ir::FloatingPointConstant(parseFloat(_lexer->nextToken()));
}

ir::Constant* ConstantValueParser::_parseStringConstant()
{
	std::string token = _lexer->nextToken();

	hydrazine::log("ConstantValueParser::Parser") << " parsed string constant '"
		<< token << "'\n";

	return new ir::ArrayConstant(token.c_str(), token.size(),
		compiler::Compiler::getSingleton()->getType("i8"));
}

}

}


