/*	\file   HostReflectionHost.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The host source file for the HostReflection set of functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/host-interface/HostReflectionHost.h>

// Ocelot Includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cstring>
#include <cassert>
#include <fstream>
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace archaeopteryx
{

namespace util
{

// device/host shared memory region
static char* _deviceHostSharedMemory = 0;

void HostReflectionHost::create(const std::string& module)
{
	assert(_booter == 0);
	_booter = new BootUp(module);
}

void HostReflectionHost::destroy()
{
	delete _booter;
}

void HostReflectionHost::handleOpenFile(HostQueue& queue,
	const Header* header)
{
	struct Payload
	{
		size_t handle;
		size_t size;
	};

	report("    handling open file message");

	const char* nameAndMode = (const char*)(header + 1);

	std::string filename(nameAndMode);
	std::string mode(nameAndMode + filename.size() + 1);

	report("     filename: " << filename);
	report("     mode: "     << mode);

	std::ios_base::openmode openmode = (std::ios_base::openmode)0;
	
	if(mode.find("r") != std::string::npos)
	{
		openmode |= std::ios_base::in;
	}
	
	if(mode.find("+") != std::string::npos)
	{
		openmode |= std::ios_base::app;
	}
	
	if(mode.find("w") != std::string::npos)
	{
		openmode |= std::ios_base::out | std::ios_base::trunc;
	}

	std::fstream* file = new std::fstream(filename.c_str(), openmode);

	report("     handle: " << file);
	report("     good:   " << (file->good() ? "yes" : "no"));
	
	Header reply(*header);
	
	reply.handler = OpenFileReplyHandler;
	reply.size    = sizeof(Header) + sizeof(Payload);
	
	Payload payload;
	
	file->seekg(0, std::ios::end);

	payload.size   = file->tellg();
	payload.handle = (file->good()) ? (size_t)file : 0;
	
	file->seekg(0, std::ios::beg);

	report("     sending reply to thread " << header->threadId);
	hostSendAsynchronous(queue, reply, &payload);
}

void HostReflectionHost::handleTeardownFile(HostQueue& queue,
	const Header* header)
{
	report("    handling teardown file message");

	std::fstream* file(*(std::fstream**)(header + 1));

	report("     handle: " << file);
	
	delete file;

	report("     file closed...");
}

void HostReflectionHost::handleFileWrite(HostQueue& queue,
	const Header* header)
{
	struct WriteHeader
	{
		size_t size;
		size_t pointer;
		size_t handle;
	};

	report("    handling file write message");
	WriteHeader* writeHeader = (WriteHeader*)(header + 1);
	
	std::fstream* file = (std::fstream*)writeHeader->handle;

	size_t bytes = writeHeader->size - sizeof(WriteHeader);

	report("     writing " << bytes << " to file " << file);
	
	file->seekp(writeHeader->pointer);
	file->write((char*)(writeHeader + 1), bytes);
}

void HostReflectionHost::handleFileRead(HostQueue& queue,
	const Header* header)
{
	struct ReadHeader
	{
		size_t size;
		size_t pointer;
		size_t handle;
	};

	report("    handling file read message");

	ReadHeader* readHeader = (ReadHeader*)(header + 1);
	
	std::fstream* file = (std::fstream*)readHeader->handle;

	size_t bytes = readHeader->size;

	report("     reading " << bytes << " from file " << file);
	
	file->seekg(readHeader->pointer);

	char* buffer = new char[bytes];

	file->read(buffer, bytes);

	Header reply(*header);
	
	reply.size = sizeof(Header) + bytes;

	hostSendAsynchronous(queue, reply, buffer);

	delete[] buffer;
}

void HostReflectionHost::handleKernelLaunch(HostQueue& queue,
	const Header* header)
{
	report("    handling kernel launch message");

	PayloadData*  payload          = (PayloadData* )(header           + 1);
	unsigned int* ctas             = (unsigned int*)(payload          + 1);
	unsigned int* threads          = (unsigned int*)(ctas             + 1);
	unsigned int* nameLength       = (unsigned int*)(threads          + 1);
	const char*   kernelName       = (const char*  )(nameLength       + 1);
	
	Payload arguments;
	arguments.data = *payload;
	
	launchFromHost(*ctas, *threads, kernelName, arguments);
}

void HostReflectionHost::hostSendAsynchronous(HostQueue& queue,
	const Header& header, const void* payload)
{
	assert(header.size  >= sizeof(Header));
	assert(queue.size() >= header.size   );

	while(!queue.push(&header, sizeof(Header)));

	while(!queue.push(payload, header.size - sizeof(Header)));
}

size_t HostReflectionHost::maxMessageSize()
{
	return MaxMessageSize;
}

void HostReflectionHost::launchFromHost(unsigned int ctas,
	unsigned int threads, const std::string& name, Payload payload)
{
	KernelLaunch launch = {ctas, threads, name, payload};

	_booter->addLaunch(launch);
}

HostReflectionHost::HostQueue::HostQueue(QueueMetaData* m)
: _metadata(m)
{

}

HostReflectionHost::HostQueue::~HostQueue()
{

}

bool HostReflectionHost::HostQueue::push(const void* data, size_t size)
{
	assert(size < this->size());

	if(size > _capacity()) return false;

	size_t end  = _metadata->size;
	size_t head = _metadata->head;

	size_t remainder = end - head;
	size_t firstCopy = std::min(remainder, size);

	std::memcpy(_metadata->hostBegin + head, data, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy(_metadata->hostBegin, (char*)data + firstCopy, secondCopy);
	_metadata->head = secondCopyNecessary ? secondCopy : head + firstCopy;
	
	return true;
}

bool HostReflectionHost::HostQueue::pull(void* data, size_t size)
{
	if(size > _used()) return false;

	report("   pulling " << size << " bytes from gpu->cpu queue (" << _used()
		<< " used, " << _capacity() << " remaining, " << this->size()
		<< " size)");

	_metadata->tail = _read(data, size);

	report("    after pull (" << _used()
		<< " used, " << _capacity() << " remaining, " << this->size()
		<< " size)");

	return true;
}

bool HostReflectionHost::HostQueue::peek()
{
	return _used() >= sizeof(Header);
}

size_t HostReflectionHost::HostQueue::size() const
{
	return _metadata->size;
}

size_t HostReflectionHost::HostQueue::_used() const
{
	size_t end  = _metadata->size;
	size_t head = _metadata->head;
	size_t tail = _metadata->tail;
	
	size_t greaterOrEqual = head - tail;
	size_t less           = (head) + (end - tail);
	
	bool isGreaterOrEqual = head >= tail;
	
	return (isGreaterOrEqual) ? greaterOrEqual : less;
}

size_t HostReflectionHost::HostQueue::_capacity() const
{
	return size() - _used();
}

size_t HostReflectionHost::HostQueue::_read(void* data, size_t size)
{
	size_t end  = _metadata->size;
	size_t tail = _metadata->tail;

	size_t remainder = end - tail;
	size_t firstCopy = std::min(remainder, size);

	std::memcpy(data, _metadata->hostBegin + tail, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy((char*)data + firstCopy, _metadata->hostBegin, secondCopy);
	
	return secondCopyNecessary ? secondCopy : tail + firstCopy;
}

void HostReflectionHost::BootUp::_addMessageHandlers()
{
	addHandler(OpenFileMessageHandler,     handleOpenFile);
	addHandler(TeardownFileMessageHandler, handleTeardownFile);
	addHandler(FileWriteMessageHandler,    handleFileWrite);
	addHandler(FileReadMessageHandler,     handleFileRead);
	addHandler(KernelLaunchMessageHandler, handleKernelLaunch);
}

HostReflectionHost::BootUp::BootUp(const std::string& module)
: _module(module)
{
	report("Booting up host reflection...");

	// add message handlers
	_addMessageHandlers();

	// allocate memory for the queue
	size_t queueDataSize = maxMessageSize() * 2;
	size_t size = 2 * (queueDataSize + sizeof(QueueMetaData));

	_deviceHostSharedMemory = new char[size];

	// setup the queue meta data
	QueueMetaData* hostToDeviceMetaData =
		(QueueMetaData*)_deviceHostSharedMemory;
	QueueMetaData* deviceToHostMetaData =
		(QueueMetaData*)_deviceHostSharedMemory + 1;

	char* hostToDeviceData = _deviceHostSharedMemory +
		2 * sizeof(QueueMetaData);
	char* deviceToHostData = _deviceHostSharedMemory +
		2 * sizeof(QueueMetaData) + queueDataSize;

	hostToDeviceMetaData->hostBegin = hostToDeviceData;
	hostToDeviceMetaData->size      = queueDataSize;
	hostToDeviceMetaData->head      = 0;
	hostToDeviceMetaData->tail      = 0;
	hostToDeviceMetaData->mutex     = (size_t)-1;

	deviceToHostMetaData->hostBegin = deviceToHostData;
	deviceToHostMetaData->size      = queueDataSize;
	deviceToHostMetaData->head      = 0;
	deviceToHostMetaData->tail      = 0;
	deviceToHostMetaData->mutex     = (size_t)-1;

	// Allocate the queues
	_hostToDeviceQueue = new HostQueue(hostToDeviceMetaData);
	_deviceToHostQueue = new HostQueue(deviceToHostMetaData);

	// Map the memory onto the device
	cudaHostRegister(_deviceHostSharedMemory, size, 0);

	char* devicePointer = 0;
	
	cudaHostGetDevicePointer((void**)&devicePointer,
		_deviceHostSharedMemory, 0);

	// Send the metadata to the device
	QueueMetaData* hostToDeviceMetaDataPointer =
		(QueueMetaData*)devicePointer;
	QueueMetaData* deviceToHostMetaDataPointer =
		(QueueMetaData*)devicePointer + 1;

	hostToDeviceMetaData->deviceBegin = devicePointer +
		2 * sizeof(QueueMetaData);
	deviceToHostMetaData->deviceBegin = devicePointer +
		2 * sizeof(QueueMetaData) + queueDataSize;

	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);

	cudaSetupArgument(&hostToDeviceMetaDataPointer, 8, 0 );
	cudaSetupArgument(&deviceToHostMetaDataPointer, 8, 8 );
	ocelot::launch(_module, "_bootupHostReflection");

	// start up the host worker thread
	_kill   = false;
	_thread = new boost::thread(_runThread, this);
}

HostReflectionHost::BootUp::~BootUp()
{
	report("Destroying host reflection");

	// kill the thread
	_kill = true;
	_thread->join();
	delete _thread;
	
	// destroy the device queues
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);
	
	ocelot::launch(_module, "_teardownHostReflection");
	cudaThreadSynchronize();
	
	// destroy the host queues
	delete _hostToDeviceQueue;
	delete _deviceToHostQueue;
	
	// delete the queue memory
	delete[] _deviceHostSharedMemory;
}

void HostReflectionHost::BootUp::addHandler(int handlerId,
	MessageHandler handler)
{
	assert(_handlers.count(handlerId) == 0);

	_handlers.insert(std::make_pair(handlerId, handler));
}

void HostReflectionHost::BootUp::addKernel(const std::string& name,
	KernelFunctionType kernel)
{
	assert(_kernels.count(name) == 0);
	
	_kernels.insert(std::make_pair(name, kernel));
}

void HostReflectionHost::BootUp::addLaunch(const KernelLaunch& launch)
{
	_launches.push(launch);
}

bool HostReflectionHost::BootUp::_handleMessage()
{
	if(!_deviceToHostQueue->peek())
	{
		return false;
	}
	
	report("  found message in gpu->cpu queue, pulling it...");
	
	Header header;
	
	_deviceToHostQueue->pull(&header, sizeof(Header));

	report("   type     " << header.type);
	report("   threadId " << header.threadId);
	report("   size     " << header.size);
	report("   handler  " << header.handler);
	
	HandlerMap::iterator handler = _handlers.find(header.handler);
	assert(handler != _handlers.end());
	
	if(header.type == Synchronous)
	{
		void* address = 0;
		_deviceToHostQueue->pull(&address, sizeof(void*));
	
		report("   synchronous ack to address: " << address);
		bool value = true;
		
		cudaMemcpyAsync(address, &value, sizeof(bool),
			cudaMemcpyHostToDevice, 0);
		header.size -= sizeof(void*);
	}

	unsigned int size = header.size + sizeof(Header);
	
	Header* message = reinterpret_cast<Header*>(new char[size]);
	
	std::memcpy(message, &header, sizeof(Header));

	_deviceToHostQueue->pull(message + 1, header.size - sizeof(Header));
	
	report("   invoking message handler...");
	handler->second(*_hostToDeviceQueue, message);
	
	delete[] reinterpret_cast<char*>(message);
	
	return true;
}

static bool areAnyCudaKernelsRunning()
{
	cudaEvent_t event;
	
	cudaEventCreate(&event);
	
	cudaEventRecord(event, 0);
	
	bool running = cudaEventQuery(event) == cudaErrorNotReady;
	
	cudaEventDestroy(event);
	
	return running;
}

void HostReflectionHost::BootUp::_run()
{
	report(" Host reflection worker thread started.");

	while(true)
	{
		if(_kill)
		{
			if(!areAnyCudaKernelsRunning())
			{
				if(_launches.empty() && !_handleMessage())
				{
					break;
				}
			}
		}
	
		if(!_launches.empty())
		{
			_launchNextKernel();
		}
		else if(!_handleMessage())
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(20));
		}
		else
		{
			while(_handleMessage());
		}
	}

	report("  Host reflection worker thread joined.");
}

void HostReflectionHost::BootUp::_launchNextKernel()
{
	assert(!_launches.empty());
	KernelLaunch& launch = _launches.front();
	
	report("  launching kernel " << launch.ctas << " ctas, "
		<< launch.threads << " threads, kernel: '" << launch.name
		<< "' in module: '" << _module << "'");
	
	cudaConfigureCall(launch.ctas, launch.threads, 0, 0);
	
	cudaSetupArgument(&launch.arguments, sizeof(PayloadData), 0);
	ocelot::launch(_module, launch.name);

	report("   kernel '" << launch.name << "' finished");

	_launches.pop();
}

void HostReflectionHost::BootUp::_runThread(BootUp* booter)
{
	booter->_run();
}

HostReflectionHost::BootUp* HostReflectionHost::_booter = 0;

}

}

