/*	\file   HostReflectionDevice.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The header file for the HostReflection device set of functions.
*/

#pragma once

// Archaeopetryx Includes
#include <archaeopteryx/util/interface/HostReflection.h>

namespace archaeopteryx
{

namespace util
{

class HostReflectionDevice : public HostReflectionShared
{
public:
	class Message
	{
	public:
		__device__ virtual void* payload() const = 0;
		__device__ virtual size_t payloadSize() const = 0;
		__device__ virtual HandlerId handler() const = 0;
	};

	class KernelLaunchMessage : public Message
	{
	public:
		__device__ KernelLaunchMessage(unsigned int ctas, unsigned int threads,
			const char* name, const Payload& payload);
		__device__ ~KernelLaunchMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HandlerId handler() const;
	
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
	__device__ static size_t maxMessageSize();

public:

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

};

}

}

