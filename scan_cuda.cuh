#pragma once

// ---------------------------------------------------------------------------
// scan_cuda.cuh — GPU parallel scan (inclusive + exclusive), in-place
//
// Algorithm: recursive multi-block Blelloch scan
//
//  Phase 1 – k_scan_blocks
//    Each block of BLOCK_ITEMS = 512 elements loads into shared memory and
//    runs the Blelloch up-sweep / down-sweep to produce a per-block scan.
//    The block's reduction total is written to d_block_sums[blockIdx.x].
//    Template parameter Inclusive selects the final writeback:
//      true  → out[i] = op(exclusive[i], original[i])   (inclusive result)
//      false → out[i] = exclusive[i]                    (exclusive result)
//
//  Phase 2 – exclusive_scan_cuda (recursive)
//    The array of block totals (length = num_blocks) is scanned exclusively,
//    yielding the offset each block must add to make its result global.
//    Recursion terminates when the sub-array fits in one block.
//
//  Phase 3 – k_add_offsets
//    Each block adds d_offsets[blockIdx.x] to every element in its slice.
//
// Complexity: O(n) work, O(log n) span, O(log_B n) kernel-launch levels
//             where B = BLOCK_ITEMS = 512.
//
// Requires: CUDA 12.8+, compiled with -std=c++17, target ≥ sm_120.
// ---------------------------------------------------------------------------

#ifndef __CUDACC__
#  error "scan_cuda.cuh must be compiled by nvcc"
#endif

#include "scan.hpp"       // Vec3 with SCAN_HD annotations
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

// ---------------------------------------------------------------------------
// CUDA error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Block parameters
// ---------------------------------------------------------------------------
static constexpr int SCAN_BLOCK_SIZE  = 256;                // threads / block
static constexpr int SCAN_BLOCK_ITEMS = SCAN_BLOCK_SIZE * 2; // elements / block

// ---------------------------------------------------------------------------
// Device-callable binary operations
//
// std::plus<T> has no __device__ annotation so cannot be called from kernels.
// These thin wrappers are used in gpu_bench.cu instead.
// ---------------------------------------------------------------------------
struct FloatAdd {
    __device__ __host__
    float operator()(float a, float b) const { return a + b; }
};

struct Vec3Add {
    __device__ __host__
    Vec3 operator()(const Vec3& a, const Vec3& b) const { return a + b; }
};

// ---------------------------------------------------------------------------
// k_scan_blocks
//
// Per-block Blelloch scan in shared memory.
//   - Loads BLOCK_ITEMS elements (2 per thread); out-of-range slots get identity.
//   - Up-sweep builds partial sums; shmem[BLOCK_ITEMS-1] = block total.
//   - Block total is saved to d_block_sums[blockIdx.x] (if not null), then
//     the root is zeroed to convert to exclusive scan.
//   - Down-sweep distributes prefix values.
//   - Writeback:
//       Inclusive=true  → d_data[i] = op(exclusive[i], original[i])
//       Inclusive=false → d_data[i] = exclusive[i]
//
// Launch: <<<num_blocks, SCAN_BLOCK_SIZE, SCAN_BLOCK_ITEMS * sizeof(T)>>>
// ---------------------------------------------------------------------------
template <typename T, typename Op, bool Inclusive>
__global__ void k_scan_blocks(T*          d_data,
                               T*          d_block_sums,
                               std::size_t n,
                               T           identity,
                               Op          op)
{
    extern __shared__ char raw[];
    T* shmem = reinterpret_cast<T*>(raw);

    const int         tid  = threadIdx.x;
    const std::size_t bid  = blockIdx.x;
    const std::size_t base = bid * static_cast<std::size_t>(blockDim.x * 2);
    const std::size_t i0   = base + static_cast<std::size_t>(tid);
    const std::size_t i1   = i0  + static_cast<std::size_t>(blockDim.x);
    const int         m    = blockDim.x * 2;   // = SCAN_BLOCK_ITEMS

    // Load originals into registers (required for inclusive writeback)
    const T orig0 = (i0 < n) ? d_data[i0] : identity;
    const T orig1 = (i1 < n) ? d_data[i1] : identity;
    shmem[tid]             = orig0;
    shmem[tid + blockDim.x] = orig1;
    __syncthreads();

    // Up-sweep (reduce phase): builds partial sums up the tree
    for (int stride = 1; stride < m; stride <<= 1) {
        const int idx = (tid + 1) * (stride * 2) - 1;
        if (idx < m)
            shmem[idx] = op(shmem[idx - stride], shmem[idx]);
        __syncthreads();
    }

    // Save block total then zero root for exclusive scan
    if (tid == 0) {
        if (d_block_sums) d_block_sums[bid] = shmem[m - 1];
        shmem[m - 1] = identity;
    }
    __syncthreads();

    // Down-sweep (distribute phase): propagates prefix values down the tree
    for (int stride = m / 2; stride >= 1; stride >>= 1) {
        const int idx = (tid + 1) * (stride * 2) - 1;
        if (idx < m) {
            T left           = shmem[idx - stride];
            shmem[idx - stride] = shmem[idx];
            shmem[idx]       = op(shmem[idx], left);
        }
        __syncthreads();
    }
    // shmem[i] now holds the per-block exclusive prefix of position i

    // Write back to global memory
    if constexpr (Inclusive) {
        if (i0 < n) d_data[i0] = op(shmem[tid],               orig0);
        if (i1 < n) d_data[i1] = op(shmem[tid + blockDim.x],  orig1);
    } else {
        if (i0 < n) d_data[i0] = shmem[tid];
        if (i1 < n) d_data[i1] = shmem[tid + blockDim.x];
    }
}

// ---------------------------------------------------------------------------
// k_add_offsets
//
// Adds d_offsets[blockIdx.x] to every element in the block's slice.
// Block 0's offset is the identity so it is unchanged (op(identity, x) = x).
// ---------------------------------------------------------------------------
template <typename T, typename Op>
__global__ void k_add_offsets(T*          d_data,
                               const T*    d_offsets,
                               std::size_t n,
                               Op          op)
{
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2;
    const std::size_t i0   = base + threadIdx.x;
    const std::size_t i1   = i0  + blockDim.x;
    const T off = d_offsets[blockIdx.x];
    if (i0 < n) d_data[i0] = op(off, d_data[i0]);
    if (i1 < n) d_data[i1] = op(off, d_data[i1]);
}

// ---------------------------------------------------------------------------
// Forward declarations (inclusive and exclusive call each other)
// ---------------------------------------------------------------------------
template <typename T, typename Op>
void inclusive_scan_cuda(T* d_data, std::size_t n, T identity, Op op);

template <typename T, typename Op>
void exclusive_scan_cuda(T* d_data, std::size_t n, T identity, Op op);

// ---------------------------------------------------------------------------
// inclusive_scan_cuda
//
// d_data : device pointer to n elements (modified in-place)
// n      : element count
// identity : identity element for op  (e.g. 0.f for FloatAdd)
// op     : device-callable binary associative operation
//
// Post: d_data[i] = op(d_data[0], ..., d_data[i])
// ---------------------------------------------------------------------------
template <typename T, typename Op>
void inclusive_scan_cuda(T* d_data, std::size_t n, T identity, Op op)
{
    if (n <= 1) return;

    const std::size_t num_blocks = (n + SCAN_BLOCK_ITEMS - 1) / SCAN_BLOCK_ITEMS;
    const std::size_t shm        = SCAN_BLOCK_ITEMS * sizeof(T);

    if (num_blocks == 1) {
        // Entire array fits in one block — no multi-block coordination needed
        k_scan_blocks<T, Op, true><<<1, SCAN_BLOCK_SIZE, shm>>>(
            d_data, nullptr, n, identity, op);
        return;
    }

    T* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(T)));

    // Phase 1: per-block inclusive scan; block totals → d_block_sums
    k_scan_blocks<T, Op, true><<<num_blocks, SCAN_BLOCK_SIZE, shm>>>(
        d_data, d_block_sums, n, identity, op);

    // Phase 2: exclusive scan of block totals → per-block offsets
    exclusive_scan_cuda(d_block_sums, num_blocks, identity, op);

    // Phase 3: add per-block offsets to make the scan global
    k_add_offsets<T, Op><<<num_blocks, SCAN_BLOCK_SIZE>>>(
        d_data, d_block_sums, n, op);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_block_sums));
}

// ---------------------------------------------------------------------------
// exclusive_scan_cuda
//
// Post: d_data[i] = op(d_data[0], ..., d_data[i-1])
//       d_data[0] = identity
// ---------------------------------------------------------------------------
template <typename T, typename Op>
void exclusive_scan_cuda(T* d_data, std::size_t n, T identity, Op op)
{
    if (n == 0) return;
    if (n == 1) {
        CUDA_CHECK(cudaMemcpy(d_data, &identity, sizeof(T),
                              cudaMemcpyHostToDevice));
        return;
    }

    const std::size_t num_blocks = (n + SCAN_BLOCK_ITEMS - 1) / SCAN_BLOCK_ITEMS;
    const std::size_t shm        = SCAN_BLOCK_ITEMS * sizeof(T);

    if (num_blocks == 1) {
        k_scan_blocks<T, Op, false><<<1, SCAN_BLOCK_SIZE, shm>>>(
            d_data, nullptr, n, identity, op);
        return;
    }

    T* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(T)));

    // Phase 1: per-block exclusive scan; block totals → d_block_sums
    k_scan_blocks<T, Op, false><<<num_blocks, SCAN_BLOCK_SIZE, shm>>>(
        d_data, d_block_sums, n, identity, op);

    // Phase 2: exclusive scan of block totals → per-block offsets
    exclusive_scan_cuda(d_block_sums, num_blocks, identity, op);

    // Phase 3: add per-block offsets
    k_add_offsets<T, Op><<<num_blocks, SCAN_BLOCK_SIZE>>>(
        d_data, d_block_sums, n, op);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_block_sums));
}
