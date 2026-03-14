// ---------------------------------------------------------------------------
// gpu_bench.cu — GPU inclusive scan benchmark
//
// Covers:
//   (a) Two element types: float32 scalar, 3-D float32 vector (Vec3)
//   (c) Problem sizes n = 2^20, 2^25, 2^30  (2^34 exceeds VRAM)
//
// Three timings are reported for each run via CUDA events:
//   h2d_ms   : host-to-device transfer
//   compute_ms: GPU scan (data already on device, H2D / D2H excluded)
//   d2h_ms   : device-to-host transfer
//
// Usage:
//   ./gpu_bench
//
// Output is one CSV line per experiment:
//   kind, type, n_label, n_total, h2d_ms, compute_ms, d2h_ms
// ---------------------------------------------------------------------------

#include "scan_cuda.cuh"   // inclusive_scan_cuda, exclusive_scan_cuda, Vec3

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// ---------------------------------------------------------------------------
// CUDA event timing helpers
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end()  {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Correctness helpers
// ---------------------------------------------------------------------------
static bool verify_float_inclusive(const float* h, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float expected = static_cast<float>(i + 1);
        if (std::fabs(h[i] - expected) > 0.5f) {
            fprintf(stderr, "  FAIL at [%zu]: expected %.1f got %.1f\n",
                    i, expected, h[i]);
            return false;
        }
    }
    return true;
}

static bool verify_vec3_inclusive(const Vec3* h, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float e = static_cast<float>(i + 1);
        if (std::fabs(h[i].x - e) > 0.5f ||
            std::fabs(h[i].y - e) > 0.5f ||
            std::fabs(h[i].z - e) > 0.5f) {
            fprintf(stderr, "  FAIL at [%zu]: expected (%.1f,%.1f,%.1f) "
                    "got (%.1f,%.1f,%.1f)\n",
                    i, e, e, e, h[i].x, h[i].y, h[i].z);
            return false;
        }
    }
    return true;
}

static bool verify_float_exclusive(const float* h, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float expected = static_cast<float>(i);     // exclusive: prefix[i] = i ones
        if (std::fabs(h[i] - expected) > 0.5f) {
            fprintf(stderr, "  FAIL at [%zu]: expected %.1f got %.1f\n",
                    i, expected, h[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// run_correctness — small n, all-ones input, compare with expected output
// ---------------------------------------------------------------------------
static void run_correctness()
{
    std::cout << "=== Correctness check ===\n";

    const std::size_t n = 64;   // fits comfortably in a single block

    // ---- float inclusive ----
    {
        std::vector<float> h(n, 1.f);
        float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        inclusive_scan_cuda(d, n, 0.f, FloatAdd{});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        std::cout << "  float  inclusive_scan_cuda : "
                  << (verify_float_inclusive(h.data(), n) ? "OK" : "FAIL") << "\n";
    }

    // ---- float exclusive ----
    {
        std::vector<float> h(n, 1.f);
        float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        exclusive_scan_cuda(d, n, 0.f, FloatAdd{});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        std::cout << "  float  exclusive_scan_cuda : "
                  << (verify_float_exclusive(h.data(), n) ? "OK" : "FAIL") << "\n";
    }

    // Test with n > BLOCK_ITEMS (exercises multi-block path)
    {
        const std::size_t n2 = std::size_t(1) << 14;   // 16 384 elements, ~32 blocks
        std::vector<float> h(n2, 1.f);
        float* d; CUDA_CHECK(cudaMalloc(&d, n2 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n2 * sizeof(float), cudaMemcpyHostToDevice));
        inclusive_scan_cuda(d, n2, 0.f, FloatAdd{});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        std::cout << "  float  inclusive_scan_cuda (n=2^14, multi-block) : "
                  << (verify_float_inclusive(h.data(), n2) ? "OK" : "FAIL") << "\n";
    }

    // ---- Vec3 inclusive ----
    {
        std::vector<Vec3> h(n, Vec3{1.f, 1.f, 1.f});
        Vec3* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
        inclusive_scan_cuda(d, n, Vec3{}, Vec3Add{});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        std::cout << "  Vec3   inclusive_scan_cuda : "
                  << (verify_vec3_inclusive(h.data(), n) ? "OK" : "FAIL") << "\n";
    }

    // ---- Vec3 multi-block inclusive ----
    {
        const std::size_t n2 = std::size_t(1) << 14;
        std::vector<Vec3> h(n2, Vec3{1.f, 1.f, 1.f});
        Vec3* d; CUDA_CHECK(cudaMalloc(&d, n2 * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n2 * sizeof(Vec3), cudaMemcpyHostToDevice));
        inclusive_scan_cuda(d, n2, Vec3{}, Vec3Add{});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n2 * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        std::cout << "  Vec3   inclusive_scan_cuda (n=2^14, multi-block) : "
                  << (verify_vec3_inclusive(h.data(), n2) ? "OK" : "FAIL") << "\n";
    }
}

// ---------------------------------------------------------------------------
// timed_float / timed_vec3
//
// Returns {h2d_ms, compute_ms, d2h_ms}.
// Allocates host + device buffers, does one warmup GPU run (data already on
// device), then one timed GPU run, then D2H.
// ---------------------------------------------------------------------------
struct Times { float h2d, compute, d2h; };

static Times timed_float(std::size_t n)
{
    // Host buffer: all-ones
    std::vector<float> h(n, 1.f);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));

    GpuTimer timer;

    // H2D
    timer.begin();
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float h2d_ms = timer.end();

    // Warmup (data already on device from H2D)
    inclusive_scan_cuda(d, n, 0.f, FloatAdd{});
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-fill for timed run
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed compute
    timer.begin();
    inclusive_scan_cuda(d, n, 0.f, FloatAdd{});
    float compute_ms = timer.end();

    // D2H (just timing the transfer; result not used for correctness here)
    timer.begin();
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    float d2h_ms = timer.end();

    CUDA_CHECK(cudaFree(d));
    return {h2d_ms, compute_ms, d2h_ms};
}

static Times timed_vec3(std::size_t n)
{
    std::vector<Vec3> h(n, Vec3{1.f, 1.f, 1.f});
    Vec3* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(Vec3)));

    GpuTimer timer;

    timer.begin();
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    float h2d_ms = timer.end();

    // Warmup
    inclusive_scan_cuda(d, n, Vec3{}, Vec3Add{});
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    inclusive_scan_cuda(d, n, Vec3{}, Vec3Add{});
    float compute_ms = timer.end();

    timer.begin();
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
    float d2h_ms = timer.end();

    CUDA_CHECK(cudaFree(d));
    return {h2d_ms, compute_ms, d2h_ms};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name
              << "  (sm_" << prop.major << prop.minor << ")"
              << "  " << (prop.totalGlobalMem >> 20) << " MiB VRAM\n\n";

    run_correctness();

    // Problem sizes
    const std::size_t sizes[]     = {
        std::size_t(1) << 20,
        std::size_t(1) << 25,
        std::size_t(1) << 30,
    };
    const char* const size_names[] = { "2^20", "2^25", "2^30" };
    const int n_sizes = 3;

    // Memory guard: skip if allocation would exceed 9 GiB
    // (leaves ~3 GiB for display, driver, and temporary block-sum buffers)
    const std::size_t vram_limit = std::size_t(9) << 30;

    std::cout << "\n=== Benchmark results ===\n";
    std::cout << "kind,type,n_label,n_total,h2d_ms,compute_ms,d2h_ms,total_ms\n";
    std::cout << std::fixed << std::setprecision(3);

    for (int si = 0; si < n_sizes; ++si) {
        const std::size_t n       = sizes[si];
        const char*       n_label = size_names[si];

        // ---- float ----
        if (n * sizeof(float) <= vram_limit) {
            Times t = timed_float(n);
            std::cout << "inclusive,float," << n_label << "," << n << ","
                      << t.h2d << "," << t.compute << "," << t.d2h << ","
                      << (t.h2d + t.compute + t.d2h) << "\n";
            std::cout.flush();
        } else {
            std::cout << "# skip float " << n_label << " (exceeds VRAM limit)\n";
        }

        // ---- Vec3 ----
        if (n * sizeof(Vec3) <= vram_limit) {
            Times t = timed_vec3(n);
            std::cout << "inclusive,vec3," << n_label << "," << n << ","
                      << t.h2d << "," << t.compute << "," << t.d2h << ","
                      << (t.h2d + t.compute + t.d2h) << "\n";
            std::cout.flush();
        } else {
            std::cout << "# skip vec3 " << n_label << " (exceeds VRAM limit)\n";
        }
    }

    return 0;
}
