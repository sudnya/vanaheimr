/*!	\file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/Variable.h>

// Standard Library Includes
#include <unordered_map>
#include <string>

// Forward Declarations
                      namespace ir       { class Module;           }
                      namespace ir       { class PTXKernel;        }
                      namespace ir       { class Global;           }
                      namespace ir       { class Local;            }
                      namespace ir       { class BasicBlock;       }
                      namespace ir       { class PTXInstruction;   }
                      namespace ir       { class PTXOperand;       }
                      namespace ir       { class PTXKernel;        }
                      namespace ir       { class Parameter;        }
namespace vanaheimr { namespace ir       { class Module;           } }
namespace vanaheimr { namespace ir       { class PredicateOperand; } }
namespace vanaheimr { namespace ir       { class Constant;         } }
namespace vanaheimr { namespace ir       { class Call;             } }
namespace vanaheimr { namespace compiler { class Compiler;         } }

namespace vanaheimr
{

namespace translation
{

class PTXToVIRTranslator
{
public:
	typedef ::ir::Module PTXModule;

public:
	PTXToVIRTranslator(compiler::Compiler* compiler);
	
public:
	/*! \brief Translate the specified PTX module, adding it to the
		vanaheimr compiler */
	void translate(const PTXModule& m);

private:
	typedef ::ir::PTXKernel      PTXKernel;
	typedef ::ir::Global         PTXGlobal;
	typedef ::ir::Local          PTXLocal;
	typedef ::ir::Parameter      PTXParameter;
	typedef ::ir::BasicBlock     PTXBasicBlock;
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;
	typedef int                  PTXDataType;
	
	typedef ir::Variable::Visibility Visibility;
	typedef ir::Call                 Call;

	typedef unsigned int PTXRegisterId;
	typedef unsigned int PTXAttribute;
	typedef unsigned int PTXLinkingDirective;

private:
	void _translateGlobal(const PTXGlobal&);
	void _translateKernel(const PTXKernel&);
	void _translateLocal(const PTXLocal&);
	void _translateParameter(const PTXParameter& argument);
	void _translateRegisterValue(PTXRegisterId, PTXDataType);
	void _recordBasicBlock(const PTXBasicBlock&);
	void _translateBasicBlock(const PTXBasicBlock&);

private:
	void _translateInstruction(const PTXInstruction& );

	bool _translateComplexInstruction(const PTXInstruction& );
	bool _translateSimpleUnaryInstruction(const PTXInstruction& );
	bool _translateSimpleBinaryInstruction(const PTXInstruction& );

	void _translateSt(const PTXInstruction& );
	void _translateSetp(const PTXInstruction& );
	void _translateBra(const PTXInstruction& );
	void _translateExit(const PTXInstruction& );
	void _translateNeg(const PTXInstruction& );
	void _translateNot(const PTXInstruction& );
	
	void _translateSimpleIntrinsic(const PTXInstruction&);

	void _translateCall(const PTXInstruction& );

private:
	typedef std::unordered_map<PTXRegisterId,
		ir::Function::register_iterator> RegisterMap;
	typedef std::unordered_map<std::string,
		ir::Function::iterator> BasicBlockMap;

private:
	ir::Operand*          _newTranslatedOperand(const PTXOperand& ptx);
	ir::PredicateOperand* _translatePredicateOperand(const PTXOperand& ptx);
	ir::PredicateOperand* _translatePredicateOperand(unsigned int condition);

private:
	ir::VirtualRegister*  _getRegister(PTXRegisterId id);
	ir::VirtualRegister*  _getSpecialVirtualRegister(unsigned int id,
	                                                 unsigned int vectorIndex);
	ir::Variable*         _getGlobal(const std::string& name);
	ir::Variable*         _getBasicBlock(const std::string& name);
	ir::Argument*         _getArgument(const std::string& name);
	ir::Operand*          _getSpecialValueOperand(unsigned int id,
	                                              unsigned int vectorIndex);
	ir::VirtualRegister*  _newTemporaryRegister(const std::string& type);
	const ir::Type*       _getType(PTXDataType type);
	const ir::Type*       _getType(const std::string& name);
	ir::Variable::Linkage _translateLinkage(PTXAttribute linkage);
	ir::Variable::Linkage _translateLinkingDirective(PTXLinkingDirective d);
	Visibility            _translateVisibility(PTXLinkingDirective d);
	unsigned int          _translateAddressSpace(unsigned int space);
	ir::Constant*         _translateInitializer(const PTXGlobal& g);
	bool                  _isArgument(const std::string& name);
	
	void                  _addSpecialPrototype(const std::string& name);
	void                  _addPrototype(const std::string& name,
	                                    const Call& call);
	
private:
	compiler::Compiler* _compiler;
	ir::Module*         _module;
	ir::Function*       _function;
	ir::BasicBlock*     _block;
	ir::Instruction*    _instruction;
	
	const PTXModule*      _ptx;
	const PTXInstruction* _ptxInstruction;
	
	RegisterMap   _registers;
	BasicBlockMap _blocks;
	
};

}

}

