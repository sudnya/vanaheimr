/* 	\file Intrinsics.h
	\date Tuesday February 4, 2013
	\author Gregory Diamos
	\brief The header file for the archeopteryx intrinsic functions.

*/

#pragma once

 // Forward Declarations
namespace vanaheimr { namespace as { class Call; } }
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }

namespace archaeopteryx
{

namespace executive
{

class IntrinsicDatabase;

class Intrinsics
{
public:
	__device__ static bool isIntrinsic(const vanaheimr::as::Call* call,
		CoreSimBlock* block);
	__device__ static void execute(const vanaheimr::as::Call* call,
		CoreSimBlock* block, unsigned int threadId);

public:
	__device__ static void loadIntrinsics();
	__device__ static void unloadIntrinsics();

};

}

}

