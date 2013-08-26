/*	\file   HostReflectionHost.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The header file for the HostReflection set of functions.
*/

#pragma once

// Archaeopetryx Includes
#include <archaeopteryx/util/interface/HostReflection.h>

// Boost Includes
#include <boost/thread.hpp>

// Standard Library Includes
#include <map>
#include <queue>

namespace archaeopteryx
{

namespace util
{

class HostReflectionHost : public HostReflectionShared
{
public:
	class KernelLaunch
	{
	public:
		unsigned int ctas;
		unsigned int threads;
		std::string  name;
		Payload      arguments;
	};

public:
	static void create(const std::string& modulename);
	static void destroy();

public:
	class HostQueue
	{
	public:
		HostQueue(QueueMetaData* metadata);
		~HostQueue();

	public:
		bool push(const void* data, size_t size);
		bool pull(void* data, size_t size);

	public:
		bool peek();
		size_t size() const;

	private:
		volatile QueueMetaData* _metadata;

	private:
		size_t _capacity() const;
		size_t _used() const;

	private:
		size_t _read(void* data, size_t size);
	};	

public:
	/*! \brief Handle an open message on the host */
	static void handleOpenFile(HostQueue& q, const Header*);
	
	/*! \brief Handle a teardown message on the host */
	static void handleTeardownFile(HostQueue& q, const Header*);
	
	/*! \brief Handle a file write message on the host */
	static void handleFileWrite(HostQueue& q, const Header*);
	
	/*! \brief Handle a file read message on the host */
	static void handleFileRead(HostQueue& q, const Header*);

	/*! \brief Handle a kernel launch message on the host */
	static void handleKernelLaunch(HostQueue& q, const Header*);

public:
	static size_t maxMessageSize();

public:
	static void hostSendAsynchronous(HostQueue& q, 
		const Header& h, const void* p);

	static void launchFromHost(unsigned int ctas, unsigned int threads,
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
		BootUp(const std::string& n);
		~BootUp();

	public:
		void addHandler(int handlerId, MessageHandler handler);
		void addKernel(const std::string& name,
			KernelFunctionType kernel);
		void addLaunch(const KernelLaunch& launch);
		
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
		void _run();
		void _launchNextKernel();
		bool _handleMessage();
		void _addMessageHandlers();
	
	private:
		static void _runThread(BootUp* kill);
	};

private:
	static BootUp* _booter;
};

}

}

