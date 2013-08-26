/*! \file   InterferenceAnalysis.cpp
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InterferenceAnalysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

#include <vanaheimr/analysis/interface/LiveRangeAnalysis.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

// Standard Library Includes
#include <cassert>
#include <algorithm>

namespace vanaheimr
{

namespace analysis
{

InterferenceAnalysis::InterferenceAnalysis()
: FunctionAnalysis("InterferenceAnalysis", {"LiveRangeAnalysis"})
{

}

bool InterferenceAnalysis::doLiveRangesInterfere(const VirtualRegister& one,
	const VirtualRegister& two) const
{
	const VirtualRegisterSet& interferences = getInterferences(one);

	return interferences.count(const_cast<VirtualRegister*>(&two)) != 0;
}

InterferenceAnalysis::VirtualRegisterSet&
	InterferenceAnalysis::getInterferences(
		const VirtualRegister& virtualRegister)
{
	assert(virtualRegister.id < _interferences.size());

	return _interferences[virtualRegister.id];
}

const InterferenceAnalysis::VirtualRegisterSet&
	InterferenceAnalysis::getInterferences(
		const VirtualRegister& virtualRegister) const
{
	assert(virtualRegister.id < _interferences.size());

	return _interferences[virtualRegister.id];
}

typedef std::pair<ir::BasicBlock*, LiveRange*> BlockToRange;
typedef std::vector<BlockToRange> BlockToRangeVector;
typedef BlockToRangeVector::iterator RangeIterator;
typedef std::pair<RangeIterator, RangeIterator> Range;
typedef std::vector<Range> RangeVector;

static BlockToRangeVector mapBlocksToLiveRanges(LiveRangeAnalysis*);
static RangeVector partition(BlockToRangeVector&);
static void checkIntersectionsInOverlappingRanges(InterferenceAnalysis*,
	const RangeVector&);

// TODO: Segregate ranges into fully covered and partially covered sets.
//       Processor them independently to reduce the complexity of the cross
//       product
void InterferenceAnalysis::analyze(Function& function)
{
	auto ranges = static_cast<LiveRangeAnalysis*>(
		getAnalysis("LiveRangeAnalysis"));
	assert(ranges != nullptr);

	_interferences.resize(function.register_size());

	// map live ranges into partiions that are alive in the same blocks
	auto blocksToRanges = mapBlocksToLiveRanges(ranges);

	auto partitions = partition(blocksToRanges);

	// compute intersections among live ranges in the same partition
	checkIntersectionsInOverlappingRanges(this, partitions);
}
	
static BlockToRangeVector mapBlocksToLiveRanges(LiveRangeAnalysis* ranges)
{
	BlockToRangeVector blocksToRanges;
	
	// TODO: for all
	for(auto range = ranges->begin(); range != ranges->end(); ++range)
	{
		// Add an entry for each block used by the range
		auto blocks = range->allBlocksWithLiveValue();
		
		for(auto block : blocks)
		{
			blocksToRanges.push_back(std::make_pair(block, &*range));
		}
	}
	
	return blocksToRanges;
}

static RangeVector partition(BlockToRangeVector& blocksToRanges)
{
	// Group live ranges by blocks that they reference 
	std::sort(blocksToRanges.begin(), blocksToRanges.end());

	// Partition the live ranges by blocks that they reference
	RangeVector blockRanges;
	
	Range currentRange(blocksToRanges.begin(), blocksToRanges.begin());
	
	// TODO: Partition the array in parallel using recursive binary search
	for(auto blockToRange = blocksToRanges.begin();
		blockToRange != blocksToRanges.end(); ++blockToRange)
	{
		currentRange.second = blockToRange;
		
		if(currentRange.first->first != currentRange.second->first)
		{
			// We found a new block
			blockRanges.push_back(currentRange);
			
			currentRange.first = currentRange.second;
		}
	}
	
	blockRanges.push_back(currentRange);
	
	return blockRanges;
}


static void checkIntersectionsInOverlappingRanges(
	InterferenceAnalysis* interefernceAnalysis,
	const RangeVector& partitions)
{
	// check all partitions (TODO in parallel)
	for(auto partition : partitions)
	{
		for(auto one = partition.first; one != partition.second; ++one)
		{
			for(auto two = partition.first; two != partition.second; ++two)
			{
				if(one == two) continue;

				if(one->second->interferesWith(*two->second))
				{
					interefernceAnalysis->getInterferences(
						*one->second->virtualRegister()).insert(
							two->second->virtualRegister());
				}
			}
		}
	}
}

}

}

