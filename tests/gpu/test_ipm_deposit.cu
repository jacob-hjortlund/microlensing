// Test of the deposit member of IPM<T> (include/ipm.cuh) in double: deposit = Deposit::direct against the automatic
// deposit on the example lens (kappa 0.4, shear 0.4, kappa_star 0.2, half-lengths 5) at 1000x1000 pixels and one
// ray per pixel, where many cells map to triangles whose bounding box has more than EDGE_MIN_BBOX pixels, without
// and with parity maps. Needs the GPU; each run takes about a second. IPM<double> has its own program because fmm.cuh
// declares its dynamic shared memory per template, so IPM<float> and IPM<double> cannot share one.
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case and exits 1 if any case fails. The run logs go
// to a temporary file under $TMPDIR (/tmp if unset).
#include "ipm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
	std::fflush(stdout);
}

// ipm.run(verbose) with stdout (std::cout and the shooting kernel's printf) redirected to a temporary file; out gets
// what was written
template <typename T>
static bool run_capturing_stdout(IPM<T>& ipm, int verbose, std::string& out)
{
	out.clear();
	const char* tmp = std::getenv("TMPDIR");
	std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/test_ipm_deposit_XXXXXX";
	const int fd = mkstemp(path.data());
	if (fd < 0)
	{
		std::fprintf(stderr, "cannot create %s\n", path.c_str());
		return false;
	}
	std::fflush(stdout);
	const int saved = dup(STDOUT_FILENO);
	dup2(fd, STDOUT_FILENO);
	const bool ran = ipm.run(verbose);
	cudaDeviceSynchronize(); //flushes the kernel's printf
	std::fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	close(fd);
	std::ifstream in(path, std::ios::binary);
	out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	unlink(path.c_str());
	return ran;
}

// the line of text that contains what, without its newline, or ""
static std::string line_with(const std::string& text, const std::string& what)
{
	const size_t at = text.find(what);
	if (at == std::string::npos) return "";
	const size_t begin = text.rfind('\n', at) + 1; //0 if there is no newline before (npos + 1)
	return text.substr(begin, text.find('\n', at) - begin); //to the end if there is none after
}

// the maps of IPM<double>, by the name the test reports them under
struct Access
{
	const char* name;
	double* (IPM<double>::*get)();
};
static const Access MAPS[3] = {
	{"total", &IPM<double>::get_pixels},
	{"minima", &IPM<double>::get_pixels_minima},
	{"saddles", &IPM<double>::get_pixels_saddles},
};

// the run log (verbosity 2) shows which deposit each run took: the automatic run must have allocated the step buffer
// and summed its rows, and the direct run must have done neither. Both deposits are exact, so the maps must agree
// to rounding: the total map, and with parities also the minima and the saddles, each of which gets the steps of
// its own parity
static void direct_deposit_matches_automatic(int write_parities)
{
	const std::string name = std::string("direct_deposit_matches_automatic") + (write_parities ? "_with_parities" : "");
	const std::string perimeter_line = "Perimeter deposit for bounding boxes above ";
	const std::string row_sum_line = "Row sums clamped ";
	const std::string direct_line = "Direct deposit for every triangle.";
	const size_t npix = 1000ULL * 1000;
	const int nmaps = write_parities ? 3 : 1;
	std::vector<double> maps[2][3];
	std::string logs[2];
	const IPM<double>::Deposit deposits[2] = {IPM<double>::Deposit::automatic, IPM<double>::Deposit::direct};
	for (int k = 0; k < 2; k++)
	{
		IPM<double> ipm;
		ipm.kappa_tot = 0.4;
		ipm.shear = 0.4;
		ipm.kappa_star = 0.2;
		ipm.half_length_y = Complex<double>(5, 5);
		ipm.num_pixels_y = Complex<int>(1000, 1000);
		ipm.num_rays_y = 1;
		ipm.random_seed = 1;
		ipm.write_parities = write_parities;
		ipm.write_stars = 0;
		ipm.write_histograms = 0; //only the maps are compared, and a wrong map's histograms can fail to allocate
		ipm.deposit = deposits[k];
		const bool ran = run_capturing_stdout(ipm, 2, logs[k]);
		for (int m = 0; m < nmaps; m++)
		{
			const double* map = ran ? (ipm.*MAPS[m].get)() : nullptr;
			if (map == nullptr)
			{
				report(name, false, std::string(k == 0 ? "automatic" : "direct")
					+ (ran ? std::string(": no ") + MAPS[m].name + " map" : ": run() returned false"));
				return;
			}
			maps[k][m].assign(map, map + npix);
		}
	}

	const bool automatic_took_perimeter = logs[0].find(perimeter_line) != std::string::npos
		&& logs[0].find(row_sum_line) != std::string::npos && logs[0].find(direct_line) == std::string::npos;
	const bool direct_took_direct = logs[1].find(direct_line) != std::string::npos
		&& logs[1].find(perimeter_line) == std::string::npos && logs[1].find(row_sum_line) == std::string::npos;

	bool finite = true;
	bool within = true;
	std::string differences; //one entry per map
	std::string worst_difference; //the first map above the tolerance
	for (int m = 0; m < nmaps; m++)
	{
		double max_rel = 0;
		size_t worst = 0;
		for (size_t i = 0; i < npix; i++)
		{
			const double a = maps[0][m][i], d = maps[1][m][i];
			finite = finite && std::isfinite(a) && std::isfinite(d);
			const double scale = std::max(std::fabs(a), std::fabs(d));
			const double rel = scale > 0 ? std::fabs(a - d) / scale : 0;
			if (rel > max_rel)
			{
				max_rel = rel;
				worst = i;
			}
		}
		char difference[200];
		std::snprintf(difference, sizeof(difference), "%s: max relative difference %.3e at pixel %zu (%.17g vs %.17g)",
			MAPS[m].name, max_rel, worst, maps[0][m][worst], maps[1][m][worst]);
		differences += (m ? "; " : "") + std::string(difference);
		if (within && max_rel > 1e-9) worst_difference = difference;
		within = within && max_rel <= 1e-9;
	}
	std::printf("INFO %s: automatic: \"%s\" \"%s\"; direct: \"%s\"; %s\n", name.c_str(),
		line_with(logs[0], perimeter_line).c_str(), line_with(logs[0], row_sum_line).c_str(),
		line_with(logs[1], direct_line).c_str(), differences.c_str());
	report(name, automatic_took_perimeter && direct_took_direct && finite && within,
		!automatic_took_perimeter ? "the automatic run did not take the perimeter deposit"
		: !direct_took_direct ? "the direct run allocated the step buffer"
		: !finite ? "a map has a value that is not finite"
		: worst_difference + ", above 1e-9");
}

int main()
{
	direct_deposit_matches_automatic(0);
	direct_deposit_matches_automatic(1);

	std::printf("%s: %d failed\n", n_fail ? "FAILED" : "OK", n_fail);
	return n_fail ? 1 : 0;
}
