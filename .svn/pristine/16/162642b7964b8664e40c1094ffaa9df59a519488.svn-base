/*	\file   HostReflection.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The header file for the HostReflection set of functions.
*/

#pragma once

// Boost Includes
#include <boost/thread.hpp>

// Standard Library Includes
#include <map>
#include <queue>

// Macro defines
#define KERNEL_PAYLOAD_BYTES        32
#define KERNEL_PAYLOAD_PARAMETERS    5

namespace util
{

class HostReflection
{
public:
	typedef unsigned int HandlerId;

	class Message
	{
	public:
		__device__ virtual void* payload() const = 0;
		__device__ virtual size_t payloadSize() const = 0;
		__device__ virtual HostReflection::HandlerId handler() const = 0;
	};
	
	enum MessageHandler
	{
		OpenFileMessageHandler     = 0,
		OpenFileReplyHandler       = 0,
		TeardownFileMessageHandler = 1,
		FileWriteMessageHandler    = 2,
		FileReadMessageHandler     = 3,
		FileReadReplyHandler       = 3,
		KernelLaunchMessageHandler = 4,
		InvalidMessageHandler      = -1
	};

	enum MessageType
	{
		Synchronous,
		Asynchronous,
		Invalid,
	};

	class Header
	{
	public:
		MessageType  type;
		unsigned int threadId;
		unsigned int size;
		HandlerId    handler;
	};
	
	class SynchronousHeader : public Header
	{
	public:
		void* address;
	};
	
	class PayloadData
	{
	public:
		char data[KERNEL_PAYLOAD_BYTES];
		unsigned int indexes[KERNEL_PAYLOAD_PARAMETERS];
	};
	
	class Payload
	{
	public:
		PayloadData data;
		
	public:
		template<typename T>
		__device__ T get(unsigned int index);
	};
	
	class KernelLaunch
	{
	public:
		unsigned int ctas;
		unsigned int threads;
		std::string  name;
		Payload      arguments;
	};

	class KernelLaunchMessage : public HostReflection::Message
	{
	public:
		__device__ KernelLaunchMessage(unsigned int ctas, unsigned int threads,
			const char* name, const Payload& payload);
		__device__ ~KernelLaunchMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		unsigned int _stringLength;
		char*        _data;
	};
	
public:
	__device__ static void sendAsynchronous(const Message& m);
	__device__ static void sendSynchronous(const Message& m);
	__device__ static void receive(Message& m);

public:
	__device__ static void launch(unsigned int ctas, unsigned int threads,
		const char* functionName,
		const Payload& payload = Payload());

	template<typename T0, typename T1, typename T2, typename T3, typename T4>
	__device__ static Payload createPayload(const T0& t0,
		const T1& t1, const T2& t2, const T3& t3, const T4& t4);

	template<typename T0, typename T1, typename T2, typename T3>
	__device__ static Payload createPayload(const T0& t0,
		const T1& t1, const T2& t2, const T3& t3);

	template<typename T0, typename T1, typename T2>
	__device__ static Payload createPayload(const T0& t0,
		const T1& t1, const T2& t2);

	template<typename T0, typename T1>
	__device__ static Payload createPayload(const T0& t0, const T1& t1);

	template<typename T0>
	__device__ static Payload createPayload(const T0& t0);

	__device__ static Payload createPayload();

public:
	__host__ __device__ static size_t maxMessageSize();

public:
	__host__ static void create(const std::string& modulename);
	__host__ static void destroy();

public:
	class QueueMetaData
	{
	public:
		char*  hostBegin;
		char*  deviceBegin;

		size_t size;
		size_t head;
		size_t tail;
		size_t mutex;
	};

	class HostQueue
	{
	public:
		__host__ HostQueue(QueueMetaData* metadata);
		__host__ ~HostQueue();

	public:	
		__host__ Message* message();
		__host__ Header*  header();

	public:
		__host__ bool push(const void* data, size_t size);
		__host__ bool pull(void* data, size_t size);

	public:
		__host__ bool peek();
		__host__ size_t size() const;

	private:
		volatile QueueMetaData* _metadata;

	private:
		__host__ size_t _capacity() const;
		__host__ size_t _used() const;

	private:
		__host__ size_t _read(void* data, size_t size);
	};

	class DeviceQueue
	{
	public:
		__device__ DeviceQueue(QueueMetaData* metadata);
		__device__ ~DeviceQueue();

	public:
		__device__ bool push(const void* data, size_t size);
		__device__ bool pull(void* data, size_t size);

	public:
		__device__ bool peek();
		__device__ size_t size() const;
	
	private:
		volatile QueueMetaData* _metadata;
		
	private:
		__device__ size_t _capacity() const;
		__device__ size_t _used() const;
		
	private:
		__device__ bool _lock();
		__device__ void _unlock();
		__device__ size_t _read(void* data, size_t size);
	};

public:
	/*! \brief Handle an open message on the host */
	__host__ static void handleOpenFile(HostQueue& q, const Header*);
	
	/*! \brief Handle a teardown message on the host */
	__host__ static void handleTeardownFile(HostQueue& q, const Header*);
	
	/*! \brief Handle a file write message on the host */
	__host__ static void handleFileWrite(HostQueue& q, const Header*);
	
	/*! \brief Handle a file read message on the host */
	__host__ static void handleFileRead(HostQueue& q, const Header*);

	/*! \brief Handle a kernel launch message on the host */
	__host__ static void handleKernelLaunch(HostQueue& q, const Header*);

public:
	__host__ static void hostSendAsynchronous(HostQueue& q, 
		const Header& h, const void* p);

	__host__ static void launchFromHost(unsigned int ctas, unsigned int threads,
		const std::string& name, Payload = Payload());

private:
	class BootUp
	{
	public:
		typedef void (*MessageHandler)(HostQueue& queue, const Header*);
		typedef std::map<int, MessageHandler> HandlerMap;
	
		typedef void (*KernelFunctionType)(Payload& payload);
		typedef std::map<std::string, KernelFunctionType> KernelMap;
		typedef std::queue<KernelLaunch> LaunchQueue;
	
	public:
		__host__ BootUp(const std::string& n);
		__host__ ~BootUp();

	public:
		__host__ void addHandler(int handlerId, MessageHandler handler);
		__host__ void addKernel(const std::string& name,
			KernelFunctionType kernel);
		__host__ void addLaunch(const KernelLaunch& launch);
		
	private:
		boost::thread* _thread;
		HostQueue*     _hostToDeviceQueue;
		HostQueue*     _deviceToHostQueue;
		bool           _kill;
		std::string    _module;
	
	private:
		HandlerMap  _handlers;
		KernelMap   _kernels;
		LaunchQueue _launches;

	private:
		__host__ void _run();
		__host__ void _launchNextKernel();
		__host__ bool _handleMessage();
		__host__ void _addMessageHandlers();
	
	private:
		static void _runThread(BootUp* kill);
	};

private:
	static BootUp* _booter;

};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/HostReflection.cpp>

