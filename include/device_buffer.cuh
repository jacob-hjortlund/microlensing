#pragma once

#include "util/util.cuh"

#include <cmath> //for std::ceil
#include <cstddef> //for std::ptrdiff_t
#include <new> //for std::bad_alloc

#include <cuda_runtime.h>


/******************************************************************************
kind of memory a grow-only buffer lives in
******************************************************************************/
enum class BufferKind { device, managed };

/******************************************************************************
make sure a buffer holds at least n elements. the buffer only grows: if it is
already large enough it is left untouched

\param ptr -- reference to the pointer to the buffer. may be nullptr
\param cap -- reference to the capacity of the buffer, in elements
\param n -- number of elements needed. 0 counts as 1
\param kind -- device memory (cudaMalloc) or managed memory (cudaMallocManaged)
\param headroom -- factor applied to n when growing, so later small increases
				   don't reallocate. at least 1
\param name -- to print in error msg

\return bool -- true if the buffer holds at least n elements, false if
				allocation failed (ptr is then nullptr and cap is 0)
******************************************************************************/
template <typename P>
bool reserve_buffer(P*& ptr, size_t& cap, size_t n, BufferKind kind, double headroom, const char* name)
{
	if (n == 0)
	{
		n = 1;
	}
	if (ptr != nullptr && cap >= n)
	{
		return true;
	}

	if (ptr != nullptr)
	{
		cudaFree(ptr);
		if (cuda_error("cudaFree(*ptr)", false, __FILE__, __LINE__)) {ptr = nullptr; cap = 0; return false;}
	}
	ptr = nullptr;
	cap = 0;

	size_t grown = static_cast<size_t>(std::ceil(static_cast<double>(n) * headroom));
	size_t new_cap = (grown > n) ? grown : n;

	cudaError_t err = (kind == BufferKind::managed)
		? cudaMallocManaged(&ptr, new_cap * sizeof(P))
		: cudaMalloc(&ptr, new_cap * sizeof(P));
	//a failed allocation leaves its error as the last error, which cuda_error reports
	if (cuda_error(name, false, __FILE__, __LINE__) || err != cudaSuccess)
	{
		ptr = nullptr;
		cap = 0;
		return false;
	}
	cap = new_cap;
	return true;
}

/******************************************************************************
free a buffer made by reserve_buffer

\param ptr -- reference to the pointer to the buffer. set to nullptr
\param cap -- reference to the capacity of the buffer. set to 0
\param return_on_error -- whether to check for and return on a CUDA error
\param name -- to print in error msg

\return bool -- false if return_on_error is true and cudaFree failed, else true
******************************************************************************/
template <typename P>
bool release_buffer(P*& ptr, size_t& cap, bool return_on_error, const char* name)
{
	bool failed = false;
	if (ptr != nullptr)
	{
		cudaFree(ptr);
		failed = return_on_error && cuda_error(name, false, __FILE__, __LINE__);
	}
	ptr = nullptr;
	cap = 0;
	return !failed;
}

/******************************************************************************
Thrust allocator backed by one grow-only device buffer, so repeated Thrust
calls don't cudaMalloc and cudaFree their temporary storage. deallocate keeps
the buffer; release frees it
******************************************************************************/
struct ThrustScratch
{
	using value_type = char;

	char* ptr = nullptr;
	size_t cap = 0;

	char* allocate(std::ptrdiff_t n)
	{
		if (!reserve_buffer(ptr, cap, static_cast<size_t>(n), BufferKind::device, 1.0, "thrust scratch"))
		{
			throw std::bad_alloc();
		}
		return ptr;
	}

	void deallocate(char*, size_t) {}

	bool release()
	{
		return release_buffer(ptr, cap, true, "cudaFree(thrust scratch)");
	}
};
