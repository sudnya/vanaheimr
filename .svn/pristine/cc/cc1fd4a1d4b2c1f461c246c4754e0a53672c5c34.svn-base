/*! \file   ControlFlowGraph.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday June 25, 2011
	\brief  The header file for the ControlFlowGraph class.
*/

#pragma once

namespace ir
{

/*! \brief A control flow graph optimized for data parallel execution */
class ControlFlowGraph
{
public:
	/*! \brief A sequence of instructions ending with a terminator */
	class BasicBlock
	{
	public:
		/*! \brief A set of basic blocks */
		class BasicBlockSet
		{
		
		
		};

	public:
		/*! \brief Get an iterator to the first instruction in the block */
		iterator begin();
		/*! \brief Get an iterator to the last instruction in the block */
		iterator end();
		
	private:
		InstructionVector _instructions;
		BasicBlockSet     _predecessors;
		BasicBlockSet     _successors;
	};


public:
	/*! \brief Create a new control flow graph with an entry and exit block */
	ControlFlowGraph();

	/*! \brief Load a control flow graph from a binary */
	explicit ControlFlowGraph(const Module& module);

	/*! \brief Copy constructor */
	ControlFlowGraph(const ControlFlowGraph& graph);
	
	/*! \brief Assignment operator */
	ControlFlowGraph& operator=(const ControlFlowGraph& graph);

public:
	/*! \brief Get an iterator to the entry block */
	iterator entry();
	/*! \brief Get an iterator to the exit block */
	iterator exit();
	
	/*! \brief Get an iterator to the first block */
	iterator begin();
	/*! \brief Get an iterator to one beyond the last block */
	iterator end();

private:
	/*! \brief A bulk-synchronous copy operation */
	void _copy(const ControlFlowGraph& graph);
	
private:
	BasicBlockSet _blocks;
	iterator      _entry;
	iterator      _exit;
};

}


