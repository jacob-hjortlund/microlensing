#pragma once

#include "util/util.cuh"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

/******************************************************************************
write a device array to disk in the .bin format of write_array (two int32,
rows and columns, followed by the data), streaming it through two pinned host
buffers so the copy of chunk k+1 overlaps the write of chunk k

\param d_vals -- pointer to array of values on the device
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to
\param chunk_bytes -- size in bytes of each pinned buffer (capped at the size
                      of the array)

\return bool -- true if file is successfully written, false if not (including
         when a CUDA error was already pending on entry, in which case nothing
         is written and the pending error is consumed after being reported)
******************************************************************************/
template <typename T>
bool write_device_array(const T* d_vals, int nrows, int ncols, const std::string& fname, size_t chunk_bytes = 64ULL << 20)
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

	std::ofstream outfile;
	outfile.open(fname, std::ios_base::binary);
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	outfile.write(reinterpret_cast<const char*>(&nrows), sizeof(int));
	outfile.write(reinterpret_cast<const char*>(&ncols), sizeof(int));

	size_t total = static_cast<size_t>(nrows) * ncols; //in elements
	if (total > 0)
	{
		size_t chunk = std::max<size_t>(chunk_bytes / sizeof(T), 1);
		chunk = std::min(chunk, total);
		size_t nchunks = (total + chunk - 1) / chunk;

		T* bufs[2] = {nullptr, nullptr};
		cudaStream_t stream = nullptr;
		bool stream_made = false;
		bool ok = true;

		cudaMallocHost(&bufs[0], chunk * sizeof(T));
		if (cuda_error("cudaMallocHost(bufs[0])", false, __FILE__, __LINE__)) ok = false;
		if (ok && nchunks > 1)
		{
			cudaMallocHost(&bufs[1], chunk * sizeof(T));
			if (cuda_error("cudaMallocHost(bufs[1])", false, __FILE__, __LINE__)) ok = false;
		}
		if (ok)
		{
			cudaStreamCreate(&stream);
			if (cuda_error("cudaStreamCreate", false, __FILE__, __LINE__)) ok = false;
			else stream_made = true;
		}

		auto chunk_len = [&](size_t k) { return std::min(chunk, total - k * chunk); };

		if (ok)
		{
			cudaMemcpyAsync(bufs[0], d_vals, chunk_len(0) * sizeof(T), cudaMemcpyDeviceToHost, stream);
			if (cuda_error("cudaMemcpyAsync(chunk 0)", false, __FILE__, __LINE__)) ok = false;
		}
		for (size_t k = 0; ok && k < nchunks; k++)
		{
			//wait for chunk k to land in its buffer
			cudaStreamSynchronize(stream);
			if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) { ok = false; break; }

			//start copying chunk k+1 into the other buffer, which was written out last iteration
			if (k + 1 < nchunks)
			{
				cudaMemcpyAsync(bufs[(k + 1) % 2], d_vals + (k + 1) * chunk, chunk_len(k + 1) * sizeof(T), cudaMemcpyDeviceToHost, stream);
				if (cuda_error("cudaMemcpyAsync(chunk k+1)", false, __FILE__, __LINE__)) { ok = false; break; }
			}

			outfile.write(reinterpret_cast<const char*>(bufs[k % 2]), chunk_len(k) * sizeof(T));
			if (!outfile)
			{
				std::cerr << "Error. Failed to write to file " << fname << "\n";
				ok = false;
			}
		}

		//drain any in-flight copy before freeing the buffers it targets
		if (stream_made) cudaStreamSynchronize(stream);
		if (stream_made) cudaStreamDestroy(stream);
		if (bufs[0]) cudaFreeHost(bufs[0]);
		if (bufs[1]) cudaFreeHost(bufs[1]);
		if (cudaGetLastError() != cudaSuccess && ok)
		{
			cudaGetLastError();
			std::cerr << "CUDA error while releasing buffers for " << fname << "\n";
			ok = false;
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
