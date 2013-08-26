/*! \file   StaticTranslationTableEntry.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

// Forward Declarations
namespace vanaheimr { namespace machine { class Operation; } }

namespace vanaheimr { namespace ir { class Type;     } }
namespace vanaheimr { namespace ir { class Constant; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A static rule for translating a VIR operation into a
	Machine equivalent  */
class StaticTranslationTableEntry : public TranslationTableEntry
{
public:
	StaticTranslationTableEntry(const std::string& _name);

public:
	/*! \brief Translate IR instruction into equivalent machine instructions */
	virtual MachineInstructionVector translateInstruction(
		const ir::Instruction*) const;

public:
	virtual TranslationTableEntry* clone() const;

public:
	typedef ir::Type     Type;
	typedef ir::Constant Constant;

	/*! \brief Describes how to translate a VIR instruction to a machine op */
	class Translation
	{
	public:
		class Argument
		{
		public:
			enum ArgumentType
			{
				Register  = 0,
				Temporary = 1,
				Immediate = 2
			};

		public:
			/*! \brief Construct a register argument */
			Argument(unsigned int index, bool isSource);
			/*! \brief Construct a temporary argument */
			Argument(unsigned int index, const Type*);
			/*! \brief Construct an immediate argument */
			Argument(const Constant* constant);

		public:
			bool isTemporary() const;
			bool isImmediate() const;
			bool isRegister()  const;

		public:
			const Constant* immediate;
			const Type*     type;
			unsigned int    index;
			bool            isSource;
			
		private:
			ArgumentType    _argumentType;
		};

		typedef std::vector<Argument> ArgumentVector;

	public:
		Translation(const Operation* _lop);

	public:
		const Operation* operation;
		ArgumentVector   arguments;

	};

	typedef std::vector<Translation> TranslationVector;
	typedef Translation::ArgumentVector ArgumentVector;

	typedef TranslationVector::iterator       iterator;
	typedef TranslationVector::const_iterator const_iterator;

public:
	unsigned int totalArguments() const;
	unsigned int totalTemporaries() const;

public:
	Translation::ArgumentVector getTemporaries() const;

public:
	iterator       begin();
	const_iterator begin() const;
	
	iterator       end();
	const_iterator end() const;
	
public:
	size_t size()  const;
	bool   empty() const;
	
public:
	TranslationVector translations; // translation into logical ops
};

}

}



