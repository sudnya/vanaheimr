/*! \file   LLVMParser.cpp
	\date   March 3, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LLVM parser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LLVMParser.h>

#include <vanaheimr/parser/interface/TypeParser.h>
#include <vanaheimr/parser/interface/ConstantValueParser.h>
#include <vanaheimr/parser/interface/TypeAliasSet.h>
#include <vanaheimr/parser/interface/Lexer.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace vanaheimr
{

namespace parser
{

LLVMParser::LLVMParser(compiler::Compiler* compiler)
: _compiler(compiler) 
{

}

typedef compiler::Compiler Compiler;

typedef ir::Module       Module;
typedef ir::Function     Function;
typedef ir::BasicBlock   BasicBlock;
typedef ir::Type         Type;
typedef ir::Variable     Variable;
typedef ir::Global       Global;
typedef ir::FunctionType FunctionType;
typedef ir::Constant     Constant;

class LLVMParserEngine
{
public:
	LLVMParserEngine(compiler::Compiler* compiler, const std::string& filename);

public:
	void parse(std::istream& stream);

public:
	std::string moduleName;

private:
	void _parseTypedefs();

	void _parseTopLevelDeclaration(const std::string& declaration);
	
	void _parseGlobalVariable();
	void _parseTypedef();
	void _parseFunction();
	void _parsePrototype();
	void _parseTarget();
	void _parseMetadata();

private:
	typedef std::set<ir::Type*> TypeSet;
	typedef std::list<std::string> StringList;

private:
	void _resolveTypeAliases();
	void _resolveTypeAlias(const std::string&);
	void _resolveTypeAliasesInSubtypes(ir::Type* type, TypeSet& visited);
	
	StringList _parseGlobalAttributes();
	Constant* _parseInitializer(const Type*);
	void _parseAlignment(ir::Global*);

	const Type* _parseType();
	void _addTypeAlias(const std::string& alias, const Type*);
	
	void _parseFunctionAttributes();
	void _parseFunctionBody();
	
private:
	void _resetParser();

private:
	typedef std::unordered_map<std::string, std::string> StringMap;

private:
	// Parser Working State
	Compiler*   _compiler;
	Module*     _module;
	Function*   _function;
	BasicBlock* _block;

	TypeAliasSet _typedefs;
	StringMap    _typedefStrings;

private:
	// Lexer Working State
	Lexer _lexer;
};

void LLVMParser::parse(const std::string& filename)
{
	std::ifstream file(filename.c_str());
	
	if(!file.is_open())
	{
		throw std::runtime_error("LLVM Parser: Could not open file '" +
			filename + "' for reading.");
	}

	LLVMParserEngine engine(_compiler, filename);

	engine.parse(file);

	_moduleName = engine.moduleName;
}

std::string LLVMParser::getParsedModuleName() const
{
	return _moduleName;
}

LLVMParserEngine::LLVMParserEngine(Compiler* compiler,
	const std::string& filename)
: moduleName(filename), _compiler(compiler)
{
	// Load up the lexer with token rules
	
	// Simple Rules
	_lexer.addTokens({"@", "define", "declare", "!", "target", "datalayout",
		"%", "|", "(", ")", ";", ",", "=", "@", "[", "]",
		"{", "}", "triple", "type", "i8", "i32", "i16", "i64", "..."});
	
	// Regex Rules
	_lexer.addTokens({"\".*\"", "\\\\.*", "[:digit:]*", "[:alnum:]*"});

	_lexer.addWhitespaceRules(" \t\n\r");
}

static bool isTopLevelDeclaration(const std::string& token)
{
	return token == "@" || token == "define" || token == "declare" ||
		token == "!" || token == "target" || token == "%";
}

void LLVMParserEngine::parse(std::istream& stream)
{
	_module = &*_compiler->newModule(moduleName);

	_lexer.setStream(&stream);

	_parseTypedefs();

	auto token = _lexer.nextToken();

	while(isTopLevelDeclaration(token))
	{
		_parseTopLevelDeclaration(token);
	
		token = _lexer.nextToken();
	}

	if(!_lexer.hitEndOfStream())
	{
		throw std::runtime_error("At " + _lexer.location() +
			": hit invalid top level declaration '" + token + "'" );
	}
}

void LLVMParserEngine::_parseTypedefs()
{
	hydrazine::log("LLVM::Parser") << "Parsing typedefs\n";
	
	while(!_lexer.hitEndOfStream())
	{
		auto token = _lexer.nextToken();

		if(token != "%") continue;
		
		auto name = _lexer.nextToken();

		if(!_lexer.scan("=")) continue;

		if(!_lexer.scan("type")) continue;

		hydrazine::log("LLVM::Parser") << " Parsed '" << name << "'\n";
		
		//_typedefStrings[name] = _getTypeString();
	}

	_resolveTypeAliases();

	_lexer.reset();
}

void LLVMParserEngine::_parseTopLevelDeclaration(const std::string& token)
{
	if(token == "@")
	{
		_parseGlobalVariable();
	}
	else if(token == "%")
	{
		_parseTypedef();
	}
	else if(token == "define")
	{
		_parseFunction();
	}
	else if(token == "declare")
	{
		_parsePrototype();
	}
	else if(token == "target")
	{
		_parseTarget();
	}
	else
	{
		_parseMetadata();
	}
}

static bool isLinkage(const std::string& token)
{
	return token == "private" ||
		token == "linker_private" ||
		token == "linker_private_weak" ||
		token == "internal" ||
		token == "available_externally" ||
		token == "linkonce" ||
		token == "weak" ||
		token == "common" ||
		token == "appending" ||
		token == "extern_weak" ||
		token == "linkonce_odr" ||
		token == "weak_odr" ||
		token == "linkonce_odr_auto_hide" ||
		token == "external" ||
		token == "dllimport" ||
		token == "dllexport";
}

static Variable::Linkage translateLinkage(const std::string& token)
{
	if(token == "internal") return Variable::InternalLinkage;
	if(token == "external") return Variable::ExternalLinkage;
	if(token == "private")  return Variable::PrivateLinkage;

	assertM(false, "Linkage " + token + " implemented.");

	return Variable::ExternalLinkage;
}

void LLVMParserEngine::_resolveTypeAliases()
{
	hydrazine::log("LLVM::Parser") << "Initializing typedefs before "
		"parsing the remainder.\n";
	
	for(auto alias : _typedefStrings)
	{
		hydrazine::log("LLVM::Parser") << " Parsing type '"
			<< alias.first << "' with aliases.\n";
		
		TypeParser parser(_compiler, &_typedefs);
			
		std::stringstream typeStream(alias.second);
			
		parser.parse(&typeStream);

		auto parsedType = *_compiler->getOrInsertType(*parser.parsedType());

		_addTypeAlias(alias.first, parsedType);
	}
	
	for(auto alias : _typedefStrings)
	{
		_resolveTypeAlias(alias.first);
	}
}

void LLVMParserEngine::_resolveTypeAlias(const std::string& alias)
{
	hydrazine::log("LLVM::Parser") << " Resolving type aliases in '"
		<< alias << "'.\n";
	
	auto aliasType = *_compiler->getOrInsertType(*_typedefs.getType(alias));

	if(aliasType == nullptr)
	{
		throw std::runtime_error("Could not find typedef entry for '" +
			alias + "'.");
	}

	TypeSet visited;

	_resolveTypeAliasesInSubtypes(aliasType, visited);
}

void LLVMParserEngine::_resolveTypeAliasesInSubtypes(
	ir::Type* type, TypeSet& visited)
{
	if(!visited.insert(type).second) return;
	
	if( type->isAlias())     return;
	if(!type->isAggregate()) return;
	
	hydrazine::log("LLVM::Parser") << "  Resolving type aliases in subtype '"
		<< type->name << "'.\n";
	
	auto aggregate = static_cast<ir::AggregateType*>(type);
	
	for(unsigned int i = 0; i < aggregate->numberOfSubTypes(); ++i)
	{
		auto subtype = aggregate->getTypeAtIndex(i);
		
		if(!subtype->isAlias())
		{
			auto originalSubtype = *_compiler->getOrInsertType(*subtype);
			_resolveTypeAliasesInSubtypes(originalSubtype, visited);
			continue;
		}
		
		auto unaliasedType = _typedefs.getType(subtype->name);
		
		if(unaliasedType == nullptr)
		{
			throw std::runtime_error("Could not find typedef entry for '" +
				subtype->name + "'.");
		}
		
		aggregate->getTypeAtIndex(i) = unaliasedType;
	}
}

void LLVMParserEngine::_parseGlobalVariable()
{
	auto name = _lexer.nextToken();

	if(!_lexer.scan("="))
	{
		throw std::runtime_error("At " + _lexer.location() +
			": expecting a '='.");
	}
	
	auto linkage = _lexer.peek();

	if(isLinkage(linkage))
	{
		_lexer.nextToken();
	}
	else
	{
		linkage = "";
	}
	
	auto arributes = _parseGlobalAttributes();

	auto type = _parseType();

	auto global = _module->newGlobal(name, type, translateLinkage(linkage),
		Global::Shared);

	hydrazine::log("LLVM::Parser") << " Parsed global variable '"
		<< global->name() << "'.\n";
	
	auto initializer = _parseInitializer(type);

	if(initializer != nullptr)
	{
		global->setInitializer(initializer);
	}

	_parseAlignment(&*global);
}

void LLVMParserEngine::_parseTypedef()
{
	auto name = _lexer.nextToken();
	
	if(!_lexer.scan("="))
	{
		throw std::runtime_error("At " + _lexer.location() +
			": expecting a '='.");
	}
	
	if(!_lexer.scan("type"))
	{
		throw std::runtime_error("At " + _lexer.location() +
			": expecting 'type'.");
	}

	auto type = _parseType();

	_addTypeAlias(name, type);
}

void LLVMParserEngine::_parseFunction()
{
	_parsePrototype();
	_parseFunctionAttributes();

	_lexer.scanThrow("{");

	_parseFunctionBody();

	_lexer.scanThrow("}");
}

void LLVMParserEngine::_parsePrototype()
{
	auto returnType = _parseType();

	_lexer.scanThrow("@");
	
	auto name = _lexer.nextToken();

	_lexer.scanThrow("(");

	auto end = _lexer.peek();

	Type::TypeVector argumentTypes;

	if(end != ")")
	{
		while(true)
		{
			argumentTypes.push_back(_parseType());
			
			auto next = _lexer.peek();

			if(next != ",") break;
			
			_lexer.scan(",");
		}
		
	}

	_lexer.scanThrow(")");

	auto type = _compiler->getOrInsertType(FunctionType(_compiler,
		returnType, argumentTypes));

	_function = &*_module->newFunction(name, Variable::ExternalLinkage,
		Variable::HiddenVisibility, *type);
}

void LLVMParserEngine::_parseTarget()
{
	hydrazine::log("LLVM::Parser") << "Parsing target\n";

	auto name = _lexer.nextToken();

	_lexer.scanThrow("=");

	auto targetString = _lexer.nextToken();

	hydrazine::log("LLVM::Parser") << " target:'" << name << " = "
		<< targetString << "'\n";

	// TODO: use this
}

void LLVMParserEngine::_parseMetadata()
{
	hydrazine::log("LLVM::Parser") << "Parsing metadata\n";

	assertM(false, "Not Implemented.");
}

static bool isGlobalAttribute(const std::string& token)
{
	if(token == "internal")     return true;
	if(token == "external")     return true;
	if(token == "private")      return true;
	if(token == "unnamed_addr") return true;
	if(token == "global")       return true;
	if(token == "constant")     return true;

	return false;
}

LLVMParserEngine::StringList LLVMParserEngine::_parseGlobalAttributes()
{
	StringList attributes;

	hydrazine::log("LLVM::Parser") << "Parsing global attributes...\n";
	
	auto next = _lexer.peek();

	while(isGlobalAttribute(next))
	{
		attributes.push_back(_lexer.nextToken());

		hydrazine::log("LLVM::Parser") << " parsed '"
			<< attributes.back() << "'\n";

		next = _lexer.peek();
	}

	return attributes;
}

static bool isConstant(const std::string& token)
{
	if(token == "zeroinitializer") return true;
	if(token.find("c\"") == 0)     return true;
	if(token.find("[") == 0)       return true;

	return false;
}

Constant* LLVMParserEngine::_parseInitializer(const Type* type)
{
	auto next = _lexer.peek();

	if(!isConstant(next)) return nullptr;

	ConstantValueParser parser(&_lexer);

	parser.parse(type);

	return parser.parsedConstant()->clone();
} 

void LLVMParserEngine::_parseAlignment(ir::Global* global)
{
	auto next = _lexer.peek();

	while(next == ",")
	{
		_lexer.scan(",");

		_lexer.nextToken();
		_lexer.nextToken();
		
		// TODO: store the alignment

		next = _lexer.peek();
	}
}

const Type* LLVMParserEngine::_parseType()
{
	TypeParser parser(_compiler, &_typedefs);
	
	parser.parse(&_lexer);
	
	hydrazine::log("LLVM::Parser") << "Parsed type '"
		<< parser.parsedType()->name << "'\n";
	
	return parser.parsedType();
}

void LLVMParserEngine::_addTypeAlias(const std::string& alias, const Type* type)
{
	hydrazine::log("LLVM::Parser") << " alias '" << alias << "' -> '"
		<< type->name << "'\n";

	_typedefs.addAlias(alias, type);
}

void LLVMParserEngine::_parseFunctionAttributes()
{
	assertM(false, "Not Implemented.");
}

void LLVMParserEngine::_parseFunctionBody()
{
	assertM(false, "Not Implemented.");
}

void LLVMParserEngine::_resetParser()
{
	_lexer.reset();
	
	_typedefs.clear();
	_typedefStrings.clear();
}

}

}


