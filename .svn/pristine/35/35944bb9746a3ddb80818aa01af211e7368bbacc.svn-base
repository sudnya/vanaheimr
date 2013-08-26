/*! \file   File.h
	\date   Saturday April 23, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the File class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/HostReflection.h>

namespace util
{

/*! \brief Perform low level operations on a file from a CUDA kernel */
class File
{
public:
	/*! \brief Create a handle to a file */
	__device__ File(const char* fileName, const char* mode = "rw");

	/*! \brief Close the file */
	__device__ ~File();
	
public:
	/*! \brief Write data from a buffer into the file at the current offset */
	__device__ void write(const void* data, size_t size);

	/*! \brief Read data from the file at the current offset into a buffer */
	__device__ void read(void* data, size_t size);

	/*! \brief Try to write data into the file, return the bytes written */
	__device__ size_t writeSome(const void* data, size_t size);

	/*! \brief Try to read from the file, return the bytes read */
	__device__ size_t readSome(void* data, size_t size);

	/*! \brief Delete the file */
	__device__ void remove();

	/*! \brief Get the size of the file */
	__device__ size_t size() const;
	
	/*! \brief Get the current get pointer */
	__device__ size_t tellg() const;
	
	/*! \brief Get the current put pointer */
	__device__ size_t tellp() const;
	
	/*! \brief Set the position of the get pointer */
	__device__ void seekg(size_t p);
	
	/*! \brief Set the position of the put pointer */
	__device__ void seekp(size_t p);

private:
	typedef size_t Handle;
	
	class OpenMessage : public HostReflection::Message
	{
	public:
		__device__ OpenMessage(const char* filename, const char* mode);
		__device__ ~OpenMessage();
	
	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		char _filename[64];
	};
	
	class OpenReply : public HostReflection::Message
	{
	public:
		__device__ OpenReply();
		__device__ ~OpenReply();

	public:
		__device__ Handle handle() const;
		__device__ size_t size()   const;
		
	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	public:
		class Payload
		{
		public:
			Handle handle;
			size_t size;
		};

	private:
		Payload _data;
	};

	class DeleteMessage : public HostReflection::Message
	{
	public:
		__device__ DeleteMessage(Handle handle);
		__device__ ~DeleteMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		Handle _handle;
	};
	
	class TeardownMessage : public HostReflection::Message
	{
	public:
		__device__ TeardownMessage(Handle handle);
		__device__ ~TeardownMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		Handle _handle;
	};
	
	class WriteMessage : public HostReflection::Message
	{
	public:
		__device__ WriteMessage(const void* data, size_t size,
			size_t pointer, Handle handle);
		__device__ ~WriteMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		class Header
		{
		public:
			size_t      size;
			size_t      pointer;
			Handle      handle;
		};
		
	private:
		void* _payload;
	};
	
	class ReadMessage : public HostReflection::Message
	{
	public:
		__device__ ReadMessage(size_t size, size_t pointer, Handle handle);
		__device__ ~ReadMessage();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		class Payload
		{
		public:
			size_t size;
			size_t pointer;
			Handle handle;
		};
	
	private:
		Payload _payload;
	};

	class ReadReply : public HostReflection::Message
	{
	public:
		__device__ ReadReply(size_t size);
		__device__ ~ReadReply();

	public:
		__device__ virtual void* payload() const;
		__device__ virtual size_t payloadSize() const;
		__device__ virtual HostReflection::HandlerId handler() const;
	
	private:
		size_t _size;
		char* _data;
	};

private:
	Handle _handle;
	size_t _size;
	size_t _put;
	size_t _get;
};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/File.cpp>

