#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "scan.hpp"
#include "scan_omp.hpp"

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
template <typename T> void print(const char* label, const std::vector<T>& v);

template <>
void print(const char* label, const std::vector<float>& v)
{
    std::cout << label << ": [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << v[i];
    std::cout << "]\n";
}

template <>
void print(const char* label, const std::vector<Vec3>& v)
{
    std::cout << label << ": [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "")
                  << "(" << v[i].x << "," << v[i].y << "," << v[i].z << ")";
    std::cout << "]\n";
}

static bool approx_eq(float a, float b, float rtol = 1e-4f)
{
    const float diff  = std::fabs(a - b);
    const float scale = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1.f);
    return diff / scale < rtol;
}

static void verify(const std::vector<float>& ref, const std::vector<float>& got,
                   const char* label)
{
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!approx_eq(ref[i], got[i])) {
            std::cerr << "MISMATCH " << label << "[" << i << "]: "
                      << "ref=" << ref[i] << "  got=" << got[i] << "\n";
            std::exit(1);
        }
    }
    std::cout << "  " << label << ": OK\n";
}

template <typename F>
static double time_ms(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // -----------------------------------------------------------------------
    // 1. Small correctness print — all four variants, n = 5
    // -----------------------------------------------------------------------
    std::cout << "=== small correctness print (n=5) ===\n\n";

    std::cout << "-- float --\n";
    {
        const std::vector<float> a = {1, 2, 3, 4, 5};
        print("  input                    ", a);

        auto run_inc = [&](const char* lbl, auto fn) {
            std::vector<float> v = a; fn(v.data(), v.size());
            print(lbl, v);  // expected: [1, 3, 6, 10, 15]
        };
        auto run_exc = [&](const char* lbl, auto fn) {
            std::vector<float> v = a; fn(v.data(), v.size());
            print(lbl, v);  // expected: [0, 1, 3, 6, 10]
        };

        run_inc("  seq  inclusive_scan     ", [](float* d, std::size_t n){ inclusive_scan(d, n); });
        run_inc("  omp  inclusive_scan     ", [](float* d, std::size_t n){ inclusive_scan_omp(d, n, 0.f); });
        run_inc("  fast inclusive_scan     ", [](float* d, std::size_t n){ inclusive_scan_omp_fast(d, n, 0.f); });
        std::cout << "\n";
        run_exc("  seq  exclusive_scan     ", [](float* d, std::size_t n){ exclusive_scan(d, n, 0.f); });
        run_exc("  omp  exclusive_scan     ", [](float* d, std::size_t n){ exclusive_scan_omp(d, n, 0.f); });
        run_exc("  fast exclusive_scan     ", [](float* d, std::size_t n){ exclusive_scan_omp_fast(d, n, 0.f); });
    }

    std::cout << "\n-- Vec3 --\n";
    {
        const std::vector<Vec3> a = {{1,0,0},{0,2,0},{0,0,3},{1,1,1}};
        print("  input             ", a);

        std::vector<Vec3> si = a; inclusive_scan(si.data(), si.size());
        std::vector<Vec3> oi = a; inclusive_scan_omp(oi.data(), oi.size(), Vec3{});
        std::vector<Vec3> fi = a; inclusive_scan_omp_fast(fi.data(), fi.size(), Vec3{});
        print("  seq inclusive", si);  // (1,0,0),(1,2,0),(1,2,3),(2,3,4)
        print("  omp inclusive", oi);
        print("  fast inclusive", fi);

        std::vector<Vec3> se = a; exclusive_scan(se.data(), se.size(), Vec3{});
        std::vector<Vec3> oe = a; exclusive_scan_omp(oe.data(), oe.size(), Vec3{});
        std::vector<Vec3> fe = a; exclusive_scan_omp_fast(fe.data(), fe.size(), Vec3{});
        print("  seq exclusive", se);  // (0,0,0),(1,0,0),(1,2,0),(1,2,3)
        print("  omp exclusive", oe);
        print("  fast exclusive", fe);
    }

    // -----------------------------------------------------------------------
    // 2. Correctness + timing — n = 2^20 floats
    //
    // All values are 1.0f → prefix sums are integers ≤ 2^20 < 2^24,
    // exactly representable in float, so seq == omp bit-for-bit.
    // -----------------------------------------------------------------------
    const std::size_t N = std::size_t{1} << 20;
    std::cout << "\n=== correctness + timing  (n = 2^20 = " << N << ") ===\n";
    std::cout << std::fixed << std::setprecision(2);

    // Warm up the OpenMP thread pool before any timing.
    {
        std::vector<float> w(N, 1.f);
        inclusive_scan_omp_fast(w.data(), N, 0.f);
    }

    auto bench = [&](const char* kind, auto seq_fn, auto omp_fn, auto fast_fn) {
        std::cout << "\n-- " << kind << " --\n";
        const std::vector<float> base(N, 1.f);

        // Each variant gets its own warmup run on a fresh copy so that
        // every timing reflects steady-state cache behaviour, not the
        // cache footprint of the previous variant.
        auto warm_and_time = [&](auto fn) -> double {
            { std::vector<float> w = base; fn(w.data(), N); }  // warmup
            std::vector<float> v = base;
            return time_ms([&]{ fn(v.data(), N); });
        };

        const double t_seq  = warm_and_time(seq_fn);
        const double t_omp  = warm_and_time(omp_fn);
        const double t_fast = warm_and_time(fast_fn);

        // Correctness check (separate runs to keep timing clean).
        std::vector<float> seq_r = base; seq_fn(seq_r.data(), N);
        std::vector<float> omp_r = base; omp_fn(omp_r.data(), N);
        std::vector<float> fast_r = base; fast_fn(fast_r.data(), N);

        verify(seq_r, omp_r,  "  Blelloch vs seq");
        verify(seq_r, fast_r, "  3-phase  vs seq");

        std::cout << "  seq      : " << std::setw(7) << t_seq  << " ms\n";
        std::cout << "  Blelloch : " << std::setw(7) << t_omp  << " ms"
                  << "  (speedup " << (t_seq / t_omp)  << "x)\n";
        std::cout << "  3-phase  : " << std::setw(7) << t_fast << " ms"
                  << "  (speedup " << (t_seq / t_fast) << "x)\n";
    };

    bench("inclusive_scan",
        [](float* d, std::size_t n){ inclusive_scan(d, n); },
        [](float* d, std::size_t n){ inclusive_scan_omp(d, n, 0.f); },
        [](float* d, std::size_t n){ inclusive_scan_omp_fast(d, n, 0.f); });

    bench("exclusive_scan",
        [](float* d, std::size_t n){ exclusive_scan(d, n, 0.f); },
        [](float* d, std::size_t n){ exclusive_scan_omp(d, n, 0.f); },
        [](float* d, std::size_t n){ exclusive_scan_omp_fast(d, n, 0.f); });

    return 0;
}
