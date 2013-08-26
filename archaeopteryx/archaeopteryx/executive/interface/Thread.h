/*! \file   Thread.h
	\date   Saturday Feburary 26, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Thread class.
*/

#pragma once

/*! \brief A namespace for program execution */
namespace executive
{

// Forward Declarations
class ThreadGroup;

/*! \brief A class representing the state of a single thread */
class Thread
{
public:
    typedef unsigned int ProgramCounter;
    typedef long long unsigned int Register;
    typedef Register* RegisterFile;
    typedef char Byte;
    
public:
    ProgramCounter pc;
    RegisterFile*  registerFile;
    Byte*          localMemory;
    ThreadGroup*   threadGroup;
};

}

