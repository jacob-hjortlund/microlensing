// Tests for write_device_array (include/device_array_io.cuh).
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case.
#include "array_functions.cuh"
#include "device_array_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <utility>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

static std::vector<char> slurp(const std::string& f)
{
	std::ifstream in(f, std::ios::binary);
	return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

template <typename T>
static void run(const char* tname, const std::string& dir)
{
	const int shapes[3][2] = {{1, 1}, {3, 5}, {1000, 1001}};
	const size_t chunks[3] = {64, 4096, 64ULL << 20};
	for (auto& s : shapes)
		for (size_t ck : chunks)
		{
			int nr = s[0], nc = s[1];
			size_t n = static_cast<size_t>(nr) * nc;
			std::vector<T> h(n);
			for (size_t i = 0; i < n; i++) h[i] = static_cast<T>(i * 0.37 + 1.0);
			T* d = nullptr;
			cudaMalloc(&d, n * sizeof(T));
			cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			std::string fa = dir + "/a.bin", fb = dir + "/b.bin";
			bool ok1 = write_array<T>(h.data(), nr, nc, fa);
			bool ok2 = write_device_array<T>(d, nr, nc, fb, ck);
			std::string name = std::string("matches_write_array_") + tname + "_" + std::to_string(nr) + "x" + std::to_string(nc) + "_chunk" + std::to_string(ck);
			bool same = ok1 && ok2 && slurp(fa) == slurp(fb) && !slurp(fa).empty();
			report(name, same, ok2 ? "bytes differ" : "write_device_array returned false");
			cudaFree(d);
		}

	T* d = nullptr;
	cudaMalloc(&d, 16 * sizeof(T));
	cudaMemset(d, 0, 16 * sizeof(T));
	report(std::string("rejects_wrong_extension_") + tname, !write_device_array<T>(d, 4, 4, dir + "/x.txt"), "returned true");
	bool bad = write_device_array<T>(d, 4, 4, "/nonexistent_dir/x.bin", 64);
	bool good = write_device_array<T>(d, 4, 4, dir + "/c.bin", 64) && slurp(dir + "/c.bin").size() == 8 + 16 * sizeof(T);
	report(std::string("unwritable_path_fails_cleanly_") + tname, !bad && good, bad ? "returned true" : "later call failed");
	cudaFree(d);

	// header-only arrays
	for (auto& sh : {std::pair<int, int>(0, 7), std::pair<int, int>(7, 0)})
	{
		std::string fa = dir + "/a.bin", fb = dir + "/b.bin";
		bool ok1 = write_array<T>(nullptr, sh.first, sh.second, fa);
		bool ok2 = write_device_array<T>(nullptr, sh.first, sh.second, fb);
		report(std::string("empty_array_header_only_") + tname + "_" + std::to_string(sh.first) + "x" + std::to_string(sh.second),
			ok1 && ok2 && slurp(fa) == slurp(fb) && slurp(fb).size() == 8, "mismatch or false");
	}

	// failure after buffers exist: /dev/full opens fine and fails on write
	{
		std::string full = dir + "/full.bin";
		std::remove(full.c_str());
		bool linked = symlink("/dev/full", full.c_str()) == 0;
		size_t n = 1 << 20;
		T* dd = nullptr;
		cudaMalloc(&dd, n * sizeof(T));
		cudaMemset(dd, 0, n * sizeof(T));
		bool multi = write_device_array<T>(dd, 1024, 1024, full, 4096);     // many chunks, fails mid-stream
		bool single = write_device_array<T>(dd, 4, 4, full, 64);            // fails at flush/close
		bool after = write_device_array<T>(dd, 1024, 1024, dir + "/d.bin", 4096)
			&& slurp(dir + "/d.bin").size() == 8 + n * sizeof(T);
		bool cuda_clean = cudaGetLastError() == cudaSuccess;
		std::remove(full.c_str());
		report(std::string("dev_full_fails_cleanly_") + tname, linked && !multi && !single && after && cuda_clean,
			!linked ? "symlink failed" : multi ? "mid-stream returned true" : single ? "small returned true" : !after ? "later call failed" : "stale CUDA error");
		cudaFree(dd);
	}

	// pending error is reported, nothing is written
	{
		T* dd = nullptr;
		cudaMalloc(&dd, 16 * sizeof(T));
		std::string f = dir + "/pending.bin";
		std::remove(f.c_str());
		cudaMemcpy(dd, dd, 16 * sizeof(T), (cudaMemcpyKind)12345); // invalid, non-sticky error
		bool r = write_device_array<T>(dd, 4, 4, f);
		bool exists = std::filesystem::exists(f);
		bool again = write_device_array<T>(dd, 4, 4, f); // error was consumed
		report(std::string("pending_error_is_reported_") + tname, !r && !exists && again, r ? "returned true" : exists ? "file written" : "retry failed");
		cudaFree(dd);
	}
}

int main()
{
	std::string dir = "/tmp/ipm_bench/cuda_tests";
	if (const char* e = std::getenv("DEVICE_IO_TEST_DIR")) dir = e;
	std::filesystem::create_directories(dir);
	run<float>("float", dir);
	run<double>("double", dir);
	return n_fail ? 1 : 0;
}
