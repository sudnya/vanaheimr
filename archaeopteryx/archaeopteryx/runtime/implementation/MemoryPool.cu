/*	\file   MemoryPool.cu
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 15, 2012
	\brief  The source file for the MemoryPool class
*/

// Archaeopteryx Includes
#include <archaeopteryx/runtime/interface/MemoryPool.h>

#include <archaeopteryx/util/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace rt
{

__device__ bool MemoryPool::allocate(uint64_t size, Address address)
{
	device_report("Attempting to allocate %d bytes at %p\n", size, address);
	
	PageMap::iterator page = _pages.lower_bound(address);

	if(page != _pages.end())
	{
		// check against the next allocation
		if(page->second.address() < address + size)
		{
			device_report(" failed, collision with subsequent "
				"allocation at 0x%p\n", page->second.address());
			return false;
		}
	}

	if(page != _pages.begin())
	{
		--page;

		// check against the previous allocation
		if(page->second.endAddress() > address)
		{
			device_report(" failed, collision with next "
				"allocation at 0x%p\n", page->second.address());
			return false;
		}
	}
	
	_pages.insert(util::make_pair(address, Page(size, address)));

	device_report(" success\n");
	return true;
}

__device__ MemoryPool::Address MemoryPool::allocate(uint64_t size)
{
	// Get the next available address
	Address address = 0;

	// TODO use a more efficient divide-and-conquer algorithm here
	for(PageMap::iterator page = _pages.begin(); page != _pages.end(); ++page)
	{
		if(address + size <= page->second.address())
		{
			break;
		}

		address = page->second.endAddress();
	}

	_pages.insert(util::make_pair(address, Page(size, address)));
	
	return address;
}

__device__ void MemoryPool::deallocate(Address address)
{
	PageMap::iterator page = _pages.find(address);

	if(page == _pages.end()) return;

	_pages.erase(page);
}

__device__ MemoryPool::Address MemoryPool::translate(Address address)
{
	// Split the allocations into less-than/greater-than the address
	PageMap::iterator page = _pages.lower_bound(address);

	if(page != _pages.end())
	{
		// check against the next allocation
		if(page->second.address() == address)
		{
			return address - page->second.address() +
				page->second.physicalAddress();
		}
	}

	if(page != _pages.begin())
	{
		--page;

		// check against the previous allocation
		if(page->second.endAddress() > address)
		{
			return address - page->second.address() +
				page->second.physicalAddress();
		}
	}
	
	return InvalidAddress;
}

__device__ MemoryPool::Page::Page(uint64_t size, Address address)
: _address(address), _data(size)
{

}

__device__ MemoryPool::Address MemoryPool::Page::address() const
{
	return _address;
}

__device__ MemoryPool::Address MemoryPool::Page::endAddress() const
{
	return address() + size();
}

__device__ MemoryPool::Address MemoryPool::Page::physicalAddress() const
{
	return (Address)_data.data();
}

__device__ uint64_t MemoryPool::Page::size() const
{
	return _data.size();
}

}

}

