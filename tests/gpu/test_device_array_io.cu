// Tests for write_device_array (include/device_array_io.cuh).
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case.
#include "array_functions.cuh"
#include "device_array_io.cuh"

#include <cstdio>
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
