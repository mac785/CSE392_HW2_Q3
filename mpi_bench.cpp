// ---------------------------------------------------------------------------
// mpi_bench.cpp — MPI + OpenMP parallel scan benchmark
//
// Covers:
//   (a) Two element types: float32 scalar, 3-D float32 vector (Vec3)
//   (c) Weak and strong scaling for n = 2^20, 2^25, 2^30, 2^34
//
// Usage:
//   mpirun -np <P> ./mpi_bench [--omp <threads>]
//
// Output is one CSV line per experiment (rank 0 only):
//   kind, type, scaling, n_total, n_local, np, n_omp, wall_ms
// ---------------------------------------------------------------------------

#include <mpi.h>
#include <omp.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cassert>

#include "scan_mpi.hpp"   // inclusive_scan_mpi, exclusive_scan_mpi, Vec3

// ============================================================
// Tiny helpers
// ============================================================

static void fill_float(float* arr, std::size_t n, float val = 1.f)
{
    for (std::size_t i = 0; i < n; ++i) arr[i] = val;
}

static void fill_vec3(Vec3* arr, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) arr[i] = Vec3{1.f, 1.f, 1.f};
}

// Verify inclusive scan of all-ones: element i should equal (base+i+1).
static bool verify_float_inclusive(const float* arr, std::size_t local_n,
                                   std::size_t base)
{
    for (std::size_t i = 0; i < local_n; ++i) {
        float expected = static_cast<float>(base + i + 1);
        if (std::fabs(arr[i] - expected) > 0.5f) return false;
    }
    return true;
}

static bool verify_vec3_inclusive(const Vec3* arr, std::size_t local_n,
                                  std::size_t base)
{
    for (std::size_t i = 0; i < local_n; ++i) {
        float expected = static_cast<float>(base + i + 1);
        if (std::fabs(arr[i].x - expected) > 0.5f ||
            std::fabs(arr[i].y - expected) > 0.5f ||
            std::fabs(arr[i].z - expected) > 0.5f) return false;
    }
    return true;
}

// ============================================================
// Timing helper — returns the max wall time across all ranks (seconds)
// ============================================================
static double timed_inclusive_float(std::size_t local_n, int warmup_runs = 1)
{
    MPI_Datatype mpi_float = MPI_FLOAT;
    std::vector<float> buf(local_n);

    // Warm-up
    for (int w = 0; w < warmup_runs; ++w) {
        fill_float(buf.data(), local_n);
        inclusive_scan_mpi(buf.data(), local_n, 0.f, std::plus<float>{}, mpi_float);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Timed run
    fill_float(buf.data(), local_n);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    inclusive_scan_mpi(buf.data(), local_n, 0.f, std::plus<float>{}, mpi_float);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_elapsed * 1e3;   // ms
}

static double timed_inclusive_vec3(std::size_t local_n, MPI_Datatype mpi_vec3,
                                   int warmup_runs = 1)
{
    auto vadd = [](const Vec3& a, const Vec3& b){ return a + b; };
    std::vector<Vec3> buf(local_n);

    for (int w = 0; w < warmup_runs; ++w) {
        fill_vec3(buf.data(), local_n);
        inclusive_scan_mpi(buf.data(), local_n, Vec3{}, vadd, mpi_vec3);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    fill_vec3(buf.data(), local_n);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    inclusive_scan_mpi(buf.data(), local_n, Vec3{}, vadd, mpi_vec3);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_elapsed * 1e3;
}

// ============================================================
// Correctness check (small n, all ranks)
// ============================================================
static void run_correctness(int rank, int /*size*/, MPI_Datatype mpi_vec3)
{
    // Use n_total = 8 * size so every rank gets exactly 8 elements.
    const std::size_t local_n = 8;
    const std::size_t base    = static_cast<std::size_t>(rank) * local_n;

    // -- float --
    {
        std::vector<float> buf(local_n, 1.f);
        inclusive_scan_mpi(buf.data(), local_n, 0.f, std::plus<float>{}, MPI_FLOAT);
        bool ok = verify_float_inclusive(buf.data(), local_n, base);
        int ok_int = ok ? 1 : 0, all_ok;
        MPI_Reduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Correctness  float  inclusive_scan_mpi : "
                      << (all_ok ? "OK" : "FAIL") << "\n";
    }

    // -- Vec3 --
    {
        auto vadd = [](const Vec3& a, const Vec3& b){ return a + b; };
        std::vector<Vec3> buf(local_n, Vec3{1.f, 1.f, 1.f});
        inclusive_scan_mpi(buf.data(), local_n, Vec3{}, vadd, mpi_vec3);
        bool ok = verify_vec3_inclusive(buf.data(), local_n, base);
        int ok_int = ok ? 1 : 0, all_ok;
        MPI_Reduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Correctness  Vec3   inclusive_scan_mpi : "
                      << (all_ok ? "OK" : "FAIL") << "\n";
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Optional: override OMP thread count from command line: --omp N
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--omp") == 0) {
            omp_set_num_threads(std::atoi(argv[i + 1]));
        }
    }

    const int n_omp = omp_get_max_threads();

    // Build Vec3 MPI type
    MPI_Datatype mpi_vec3 = make_vec3_mpi_type();

    // ----------------------------------------------------------------
    // Correctness
    // ----------------------------------------------------------------
    if (rank == 0) std::cout << "=== Correctness check ===\n";
    run_correctness(rank, size, mpi_vec3);
    MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------
    // Scaling benchmarks
    // ----------------------------------------------------------------
    // Problem sizes (total element count for strong; local count for weak).
    // 2^34 is 16 GiB of floats — only attempt if local allocation fits.
    const std::size_t sizes[] = {
        std::size_t(1) << 20,   // ~4 MB float / ~12 MB Vec3
        std::size_t(1) << 25,   // ~128 MB / ~384 MB
        std::size_t(1) << 30,   // ~4 GB / ~12 GB
        std::size_t(1) << 34,   // ~64 GB / ~192 GB  (may be skipped)
    };
    const char* size_names[] = { "2^20", "2^25", "2^30", "2^34" };
    const int   n_sizes       = 4;

    // CSV header (rank 0 only)
    if (rank == 0) {
        std::cout << "\n=== Benchmark results ===\n";
        std::cout << "kind,type,scaling,n_label,n_total,n_local,np,n_omp,wall_ms\n";
        std::cout << std::fixed << std::setprecision(3);
    }

    auto print_row = [&](const char* kind, const char* type, const char* scaling,
                         const char* n_label, std::size_t n_total, std::size_t n_local,
                         double wall_ms)
    {
        if (rank == 0) {
            std::cout << kind     << ","
                      << type     << ","
                      << scaling  << ","
                      << n_label  << ","
                      << n_total  << ","
                      << n_local  << ","
                      << size     << ","
                      << n_omp    << ","
                      << wall_ms  << "\n";
            std::cout.flush();
        }
    };

    for (int si = 0; si < n_sizes; ++si) {
        const std::size_t N         = sizes[si];
        const char*       n_label   = size_names[si];

        // ------------------------------------------------------------------
        // Strong scaling: total = N, local = N / size
        // ------------------------------------------------------------------
        {
            if (N % static_cast<std::size_t>(size) != 0) {
                // Skip: N not evenly divisible by process count
                if (rank == 0)
                    std::cout << "# skip strong " << n_label
                              << " (not divisible by " << size << ")\n";
            } else {
                const std::size_t local_n = N / static_cast<std::size_t>(size);

                // Memory guard: skip if local allocation > 4 GB
                const std::size_t float_bytes = local_n * sizeof(float);
                const std::size_t vec3_bytes  = local_n * sizeof(Vec3);
                const std::size_t limit       = std::size_t(4) << 30;   // 4 GB

                if (float_bytes <= limit) {
                    double t = timed_inclusive_float(local_n);
                    print_row("inclusive", "float", "strong", n_label,
                              N, local_n, t);
                } else if (rank == 0) {
                    std::cout << "# skip strong float " << n_label
                              << " (local=" << local_n << " too large)\n";
                }

                if (vec3_bytes <= limit) {
                    double t = timed_inclusive_vec3(local_n, mpi_vec3);
                    print_row("inclusive", "vec3", "strong", n_label,
                              N, local_n, t);
                } else if (rank == 0) {
                    std::cout << "# skip strong vec3 " << n_label
                              << " (local=" << local_n << " too large)\n";
                }
            }
        }

        // ------------------------------------------------------------------
        // Weak scaling: each rank always processes N elements
        // (total grows linearly with the number of processes)
        // ------------------------------------------------------------------
        {
            const std::size_t local_n  = N;
            const std::size_t n_total  = local_n * static_cast<std::size_t>(size);

            const std::size_t float_bytes = local_n * sizeof(float);
            const std::size_t vec3_bytes  = local_n * sizeof(Vec3);
            const std::size_t limit       = std::size_t(4) << 30;

            if (float_bytes <= limit) {
                double t = timed_inclusive_float(local_n);
                print_row("inclusive", "float", "weak", n_label,
                          n_total, local_n, t);
            } else if (rank == 0) {
                std::cout << "# skip weak float " << n_label << " (too large)\n";
            }

            if (vec3_bytes <= limit) {
                double t = timed_inclusive_vec3(local_n, mpi_vec3);
                print_row("inclusive", "vec3", "weak", n_label,
                          n_total, local_n, t);
            } else if (rank == 0) {
                std::cout << "# skip weak vec3 " << n_label << " (too large)\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Type_free(&mpi_vec3);
    MPI_Finalize();
    return 0;
}
