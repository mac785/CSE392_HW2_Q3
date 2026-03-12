#pragma once

// ---------------------------------------------------------------------------
// Parallel scan implementations — OpenMP
//
// Two strategies are provided:
//
//  1. Blelloch (exclusive/inclusive_scan_omp)
//     Classic work-efficient tree scan: O(n) work, O(log n) span.
//     Thread team is spawned once; `#pragma omp for` barriers synchronise
//     the 2·log₂(n) tree levels.  Theoretically optimal for PRAM / GPU,
//     but on cache-coherent CPUs the strided memory access pattern and
//     frequent barriers limit practical speedup.
//
//  2. Three-phase chunked scan (exclusive/inclusive_scan_omp_fast)
//     Better suited to OpenMP on shared-memory CPUs:
//       Phase 1 — each thread does a sequential scan on its contiguous chunk
//                 (cache-friendly, zero inter-thread traffic)
//       Phase 2 — sequential scan over the p chunk totals (p = # threads)
//       Phase 3 — each thread adjusts its chunk with its prefix total
//     Two barriers total; memory access is always linear.
// ---------------------------------------------------------------------------

#include "scan.hpp"

#include <vector>
#include <functional>
#include <cstddef>
#include <omp.h>

// ---------------------------------------------------------------------------
// exclusive_scan_omp
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void exclusive_scan_omp(T* data, std::size_t n, T identity, Op op = Op{})
{
    if (n == 0) return;
    if (n == 1) { data[0] = identity; return; }

    // Pad to the next power of two so the tree is always complete.
    std::size_t m = 1;
    while (m < n) m <<= 1;

    // Working buffer — padding slots hold `identity` so they are neutral.
    std::vector<T> buf(m, identity);

    // ---- single parallel region (one fork, one join) ---------------------
    #pragma omp parallel default(none) shared(buf, data, m, n, identity, op)
    {
        // Copy data into buf.
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            buf[i] = data[i];
        // implicit barrier — all threads finished copy before up-sweep

        // Up-sweep: each outer iteration is a tree level; all threads
        // step through levels in lock-step via the implicit barrier on `for`.
        for (std::size_t d = 1; d < m; d <<= 1) {
            const std::size_t   stride = d << 1;
            const std::ptrdiff_t cnt   = static_cast<std::ptrdiff_t>(m / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t j = 0; j < cnt; ++j) {
                const std::size_t k = static_cast<std::size_t>(j) * stride;
                buf[k + stride - 1] = op(buf[k + d - 1], buf[k + stride - 1]);
            }
            // implicit barrier: level d done before level d+1 starts
        }

        // Zero the root (exclusive scan discards the total).
        // `single` has an implicit barrier — all threads wait here.
        #pragma omp single
        buf[m - 1] = identity;

        // Down-sweep.
        for (std::size_t d = m >> 1; d > 0; d >>= 1) {
            const std::size_t   stride = d << 1;
            const std::ptrdiff_t cnt   = static_cast<std::ptrdiff_t>(m / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t j = 0; j < cnt; ++j) {
                const std::size_t k = static_cast<std::size_t>(j) * stride;
                T t                  = buf[k + d - 1];
                buf[k + d - 1]       = buf[k + stride - 1];
                buf[k + stride - 1]  = op(buf[k + stride - 1], t);
            }
        }

        // Copy result back.
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            data[i] = buf[i];
    }
    // ---- end of parallel region ------------------------------------------
}

// ---------------------------------------------------------------------------
// inclusive_scan_omp
//
// Saves originals, runs exclusive scan inside one parallel region,
// then folds originals back in — all within the same thread team.
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void inclusive_scan_omp(T* data, std::size_t n, T identity, Op op = Op{})
{
    if (n == 0) return;

    std::size_t m = 1;
    while (m < n) m <<= 1;

    std::vector<T> buf(m, identity);
    std::vector<T> orig(n);

    #pragma omp parallel default(none) shared(buf, orig, data, m, n, identity, op)
    {
        // Copy in and preserve originals.
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            orig[i] = data[i];
            buf[i]  = data[i];
        }

        // Up-sweep.
        for (std::size_t d = 1; d < m; d <<= 1) {
            const std::size_t   stride = d << 1;
            const std::ptrdiff_t cnt   = static_cast<std::ptrdiff_t>(m / stride);
            #pragma omp for schedule(static)
            for (std::ptrdiff_t j = 0; j < cnt; ++j) {
                const std::size_t k = static_cast<std::size_t>(j) * stride;
                buf[k + stride - 1] = op(buf[k + d - 1], buf[k + stride - 1]);
            }
        }

        #pragma omp single
        buf[m - 1] = identity;

        // Down-sweep.
        for (std::size_t d = m >> 1; d > 0; d >>= 1) {
            const std::size_t   stride = d << 1;
            const std::ptrdiff_t cnt   = static_cast<std::ptrdiff_t>(m / stride);
            #pragma omp for schedule(static)
            for (std::ptrdiff_t j = 0; j < cnt; ++j) {
                const std::size_t k = static_cast<std::size_t>(j) * stride;
                T t                  = buf[k + d - 1];
                buf[k + d - 1]       = buf[k + stride - 1];
                buf[k + stride - 1]  = op(buf[k + stride - 1], t);
            }
        }

        // Combine exclusive result with original to get inclusive result.
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            data[i] = op(buf[i], orig[i]);
    }
}

// ---------------------------------------------------------------------------
// Three-phase exclusive scan — practical OpenMP speedup
//
//  Phase 1: each thread sequentially scans its contiguous chunk
//            (cache-friendly, no inter-thread traffic)
//  Phase 2: sequential exclusive scan of p chunk totals  (p is tiny)
//  Phase 3: each thread linearly adjusts its chunk
//
// Single parallel region, two barriers.  All memory access is linear.
// Uses omp_get_max_threads() so no extra fork-join to query thread count.
// Pads `totals` to one entry per cache line to prevent false sharing.
// ---------------------------------------------------------------------------

namespace scan_detail {
    // One slot per thread, padded to a full cache line (64 bytes).
    template <typename T>
    struct alignas(64) Slot { T value; };
} // namespace scan_detail

template <typename T, typename Op = std::plus<T>>
void exclusive_scan_omp_fast(T* data, std::size_t n, T identity, Op op = Op{})
{
    if (n == 0) return;

    const int p = omp_get_max_threads();
    std::vector<scan_detail::Slot<T>> slots(static_cast<std::size_t>(p));
    for (auto& s : slots) s.value = identity;

    #pragma omp parallel default(none) shared(data, n, identity, op, slots, p)
    {
        const int         tid   = omp_get_thread_num();
        const std::size_t sz    = static_cast<std::size_t>(p);
        const std::size_t chunk = (n + sz - 1) / sz;
        const std::size_t beg   = std::min(static_cast<std::size_t>(tid)     * chunk, n);
        const std::size_t end   = std::min(static_cast<std::size_t>(tid + 1) * chunk, n);

        // Phase 1: local inclusive scan; chunk total lands in data[end-1].
        if (beg < end) {
            for (std::size_t i = beg + 1; i < end; ++i)
                data[i] = op(data[i - 1], data[i]);
            slots[static_cast<std::size_t>(tid)].value = data[end - 1];
        }

        // Phase 2: one thread exclusive-scans the totals (serial, p ≈ 24).
        #pragma omp barrier
        #pragma omp single
        {
            T acc = identity;
            for (int t = 0; t < p; ++t) {
                T cur = slots[static_cast<std::size_t>(t)].value;
                slots[static_cast<std::size_t>(t)].value = acc;
                acc = op(acc, cur);
            }
        }
        // implicit barrier after `single`

        // Phase 3: convert the local inclusive scan to an exclusive one by
        // injecting the chunk prefix, using a forward pass with a saved
        // temporary so memory access stays linear (cache-friendly).
        //
        //   exclusive[i] = op(prefix, inclusive_local[i-1])
        //   exclusive[beg] = prefix
        //
        const T prefix = slots[static_cast<std::size_t>(tid)].value;
        if (beg < end) {
            T saved  = data[beg];           // inclusive_local[beg] = original[beg]
            data[beg] = prefix;
            for (std::size_t i = beg + 1; i < end; ++i) {
                const T next = data[i];     // inclusive_local[i], read before overwrite
                data[i] = op(prefix, saved);
                saved   = next;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Three-phase inclusive scan
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void inclusive_scan_omp_fast(T* data, std::size_t n, T identity, Op op = Op{})
{
    if (n == 0) return;

    const int p = omp_get_max_threads();
    std::vector<scan_detail::Slot<T>> slots(static_cast<std::size_t>(p));
    for (auto& s : slots) s.value = identity;

    #pragma omp parallel default(none) shared(data, n, identity, op, slots, p)
    {
        const int         tid   = omp_get_thread_num();
        const std::size_t sz    = static_cast<std::size_t>(p);
        const std::size_t chunk = (n + sz - 1) / sz;
        const std::size_t beg   = std::min(static_cast<std::size_t>(tid)     * chunk, n);
        const std::size_t end   = std::min(static_cast<std::size_t>(tid + 1) * chunk, n);

        // Phase 1: local inclusive scan.
        if (beg < end) {
            for (std::size_t i = beg + 1; i < end; ++i)
                data[i] = op(data[i - 1], data[i]);
            slots[static_cast<std::size_t>(tid)].value = data[end - 1];
        }

        #pragma omp barrier
        #pragma omp single
        {
            T acc = identity;
            for (int t = 0; t < p; ++t) {
                T cur = slots[static_cast<std::size_t>(t)].value;
                slots[static_cast<std::size_t>(t)].value = acc;
                acc = op(acc, cur);
            }
        }

        // Phase 3: add the chunk prefix (thread 0 has identity — skip it).
        const T prefix = slots[static_cast<std::size_t>(tid)].value;
        if (beg < end && tid > 0) {
            for (std::size_t i = beg; i < end; ++i)
                data[i] = op(prefix, data[i]);
        }
    }
}
