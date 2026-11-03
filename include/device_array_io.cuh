#pragma once

#include "util/util.cuh"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>

/******************************************************************************
writes device arrays to disk in the .bin format of write_array (two int32,
rows and columns, followed by the data), streaming each one through two pinned
host buffers so the copy of chunk k+1 overlaps the write of chunk k

the buffers and the copy stream are made on the first write, each buffer sized
to the smaller of chunk_bytes and that array (only one buffer if the array fits
in one chunk), and are reused by the later writes while the arrays fit: a writer
that writes several maps allocates its buffers once. a later array that needs
more is given new ones. a write reserves its buffers before it opens its file,
so a write that cannot get them leaves no file. a failed file leaves them in
place for the next write; a CUDA error discards them, and the next write makes
new ones. release() or the destructor frees them

the buffers are pinned (page-locked) unless pinned is false. if pinning fails
the writer uses a pageable buffer in its place, which copies more slowly but
writes the same file

\param chunk_bytes -- size in bytes of each pinned buffer (capped at the size
                      of the array)
\param pinned -- whether to pin the buffers (they are pageable if pinning fails
                or this is false)
******************************************************************************/
template <typename T>
class DeviceArrayWriter
{
public:
	explicit DeviceArrayWriter(size_t chunk_bytes = 64ULL << 20, bool pinned = true)
		: chunk_elems(std::max<size_t>(chunk_bytes / sizeof(T), 1)), want_pinned(pinned)
	{

	}

	~DeviceArrayWriter()
	{
		if (!release()) std::cerr << "CUDA error while releasing the buffers of a device array writer\n";
	}

	DeviceArrayWriter(const DeviceArrayWriter&) = delete;
	DeviceArrayWriter& operator=(const DeviceArrayWriter&) = delete;

	/******************************************************************************
	write a device array to disk

	\param d_vals -- pointer to array of values on the device
	\param nrows -- number of rows in array
	\param ncols -- number of columns in array
	\param fname -- location of the file to write to

	\return bool -- true if file is successfully written, false if not (including
	         when a CUDA error was already pending on entry, in which case nothing
	         is written and the pending error is consumed after being reported)
	******************************************************************************/
	bool write(const T* d_vals, int nrows, int ncols, const std::string& fname)
	{
		//a pending CUDA error means the caller's device data cannot be trusted, and
		//cuda_error below reads the same last-error state, so report it and stop
		cudaError_t pending = cudaGetLastError();
		if (pending != cudaSuccess)
		{
			std::cerr << "Error. A CUDA error was pending before writing " << fname << ": "
				<< static_cast<int>(pending) << " (" << cudaGetErrorString(pending) << ")\n";
			return false;
		}

		std::filesystem::path fpath = fname;

		if (fpath.extension() != ".bin")
		{
			std::cerr << "Error. File " << fname << " is not a .bin file.\n";
			return false;
		}

		size_t total = static_cast<size_t>(nrows) * ncols; //in elements
		size_t chunk = std::min(chunk_elems, total);
		size_t nchunks = total > 0 ? (total + chunk - 1) / chunk : 0;

		//reserved before the file is opened, so that a failed reservation leaves no file
		if (total > 0 && !reserve(chunk, nchunks > 1)) return false;

		std::ofstream outfile;
		outfile.open(fname, std::ios_base::binary);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile.write(reinterpret_cast<const char*>(&nrows), sizeof(int));
		outfile.write(reinterpret_cast<const char*>(&ncols), sizeof(int));

		if (total > 0)
		{
			bool ok = true; //the file
			bool cuda_ok = true;

			auto chunk_len = [&](size_t k) { return std::min(chunk, total - k * chunk); };

			cudaMemcpyAsync(bufs[0], d_vals, chunk_len(0) * sizeof(T), cudaMemcpyDeviceToHost, stream);
			if (cuda_error("cudaMemcpyAsync(chunk 0)", false, __FILE__, __LINE__)) cuda_ok = false;
			for (size_t k = 0; cuda_ok && ok && k < nchunks; k++)
			{
				//wait for chunk k to land in its buffer
				cudaStreamSynchronize(stream);
				if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) { cuda_ok = false; break; }

				//start copying chunk k+1 into the other buffer, which was written out last iteration
				if (k + 1 < nchunks)
				{
					cudaMemcpyAsync(bufs[(k + 1) % 2], d_vals + (k + 1) * chunk, chunk_len(k + 1) * sizeof(T), cudaMemcpyDeviceToHost, stream);
					if (cuda_error("cudaMemcpyAsync(chunk k+1)", false, __FILE__, __LINE__)) { cuda_ok = false; break; }
				}

				outfile.write(reinterpret_cast<const char*>(bufs[k % 2]), chunk_len(k) * sizeof(T));
				if (!outfile)
				{
					std::cerr << "Error. Failed to write to file " << fname << "\n";
					ok = false;
				}
			}

			//drain any in-flight copy before the buffers it targets are reused or freed
			cudaStreamSynchronize(stream);
			if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) cuda_ok = false;
			if (!cuda_ok)
			{
				release(); //cuda_error reported the error; the next write makes new buffers
				return false;
			}
			if (!ok) return false;
		}

		outfile.close();
		if (outfile.fail())
		{
			std::cerr << "Error. Failed to close file " << fname << "\n";
			return false;
		}
		return true;
	}

	/******************************************************************************
	free the buffers and the stream (a later write makes new ones)

	\return bool -- true if no CUDA error occurred (the last-error state is
	         consumed)
	******************************************************************************/
	bool release()
	{
		if (stream != nullptr)
		{
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
			stream = nullptr;
		}
		for (int i = 0; i < 2; i++)
		{
			if (bufs[i] != nullptr) free_buffer(i);
			bufs[i] = nullptr;
		}
		cap = 0;
		return cudaGetLastError() == cudaSuccess;
	}

protected:
	int n_allocations = 0; //buffers allocated so far (for tests)

private:
	size_t chunk_elems; //largest buffer, in elements
	bool want_pinned; //whether to try to pin the buffers
	bool buf_pinned[2] = {false, false}; //whether each buffer is pinned (decides how it is freed)
	size_t cap = 0; //size of each buffer, in elements
	T* bufs[2] = {nullptr, nullptr};
	cudaStream_t stream = nullptr;

	void free_buffer(int i)
	{
		if (buf_pinned[i]) cudaFreeHost(bufs[i]);
		else delete[] bufs[i];
	}

	//allocates buffer i of n elements: pinned if wanted, or pageable if that is not wanted or pinning fails
	//(the pinning error is consumed); false (with a message) if there is no memory
	bool alloc_buffer(int i, size_t n)
	{
		buf_pinned[i] = false;
		if (want_pinned)
		{
			if (cudaMallocHost(&bufs[i], n * sizeof(T)) == cudaSuccess)
			{
				buf_pinned[i] = true;
				return true;
			}
			cudaGetLastError();
			bufs[i] = nullptr;
			std::cerr << "Warning. Could not pin " << n * sizeof(T) << " bytes of host memory; a device array writer is using a pageable buffer\n";
		}
		bufs[i] = new (std::nothrow) T[n];
		if (bufs[i] == nullptr)
		{
			std::cerr << "Error. Failed to allocate a host buffer of " << n * sizeof(T) << " bytes for a device array writer\n";
			return false;
		}
		return true;
	}

	/******************************************************************************
	make sure there are buffers of at least n elements (two if two) and a stream;
	pinned unless that is not wanted or fails;
	on failure everything is released
	******************************************************************************/
	bool reserve(size_t n, bool two)
	{
		if (n > cap && cap > 0 && !release())
		{
			std::cerr << "CUDA error while releasing pinned buffers to make larger ones\n";
			return false;
		}
		if (bufs[0] == nullptr)
		{
			if (!alloc_buffer(0, n))
			{
				release();
				return false;
			}
			n_allocations++;
			cap = n;
		}
		if (two && bufs[1] == nullptr)
		{
			if (!alloc_buffer(1, cap))
			{
				release();
				return false;
			}
			n_allocations++;
		}
		if (stream == nullptr)
		{
			cudaStreamCreate(&stream);
			if (cuda_error("cudaStreamCreate", false, __FILE__, __LINE__))
			{
				stream = nullptr;
				release();
				return false;
			}
		}
		return true;
	}
};
