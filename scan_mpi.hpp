#pragma once

// ---------------------------------------------------------------------------
// scan_mpi.hpp — MPI + OpenMP distributed in-place inclusive/exclusive scan
//
// Algorithm (4 phases) — inclusive variant:
//
//  Phase 1 — Local inclusive scan (OpenMP):
//    Each rank runs inclusive_scan_omp_fast on its local_n elements.
//    After this, local_arr[local_n-1] holds the local reduction total.
//
//  Phase 2 — Gather totals (MPI):
//    MPI_Allgather exchanges each rank's local total so every rank has all p
//    partial sums.  Cost: O(log p) latency + p*sizeof(T) bandwidth.
//
//  Phase 3 — Compute offset (serial, O(p)):
//    Each rank independently computes its own offset as the exclusive prefix
//    of the p totals up to (but not including) its own rank.
//
//  Phase 4 — Apply offset (OpenMP):
//    Each rank adds its offset to every local element in parallel.
//    Rank 0's offset is the identity, so it skips this phase.
//
// Complexity (per rank):
//   Work   : O(n/p)   — phases 1 & 4
//   Span   : O(log(n/p) + p)
//   Comm   : 1 × Allgather of p elements
// ---------------------------------------------------------------------------

#include <mpi.h>
#include "scan_omp.hpp"   // inclusive_scan_omp_fast, exclusive_scan_omp_fast
#include <vector>
#include <cstddef>
#include <functional>

// ---------------------------------------------------------------------------
// Vec3 MPI helpers
// ---------------------------------------------------------------------------

/// Build a contiguous MPI datatype for Vec3 (3 × MPI_FLOAT).
/// Caller must MPI_Type_free() the returned handle when done.
inline MPI_Datatype make_vec3_mpi_type()
{
    MPI_Datatype dt;
    MPI_Type_contiguous(3, MPI_FLOAT, &dt);
    MPI_Type_commit(&dt);
    return dt;
}

// ---------------------------------------------------------------------------
// inclusive_scan_mpi
//
//   local_arr  : this rank's contiguous slice of the global array (in-place)
//   local_n    : number of elements on this rank
//   identity   : identity element for op  (0.f for float, Vec3{} for Vec3)
//   op         : binary associative operation  (T, T) -> T
//   mpi_type   : MPI datatype for T  (MPI_FLOAT or the Vec3 type above)
//
// Post-condition: if base = rank * local_n, then
//   local_arr[i] = op(global[0], global[1], …, global[base + i])
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void inclusive_scan_mpi(T*           local_arr,
                        std::size_t  local_n,
                        T            identity,
                        Op           op,
                        MPI_Datatype mpi_type)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Phase 1: local inclusive scan (OpenMP) ---------------------------
    inclusive_scan_omp_fast(local_arr, local_n, identity, op);

    // ---- Phase 2: exchange local totals via Allgather ---------------------
    T local_total = (local_n > 0) ? local_arr[local_n - 1] : identity;
    std::vector<T> all_totals(static_cast<std::size_t>(size));
    MPI_Allgather(&local_total,      1, mpi_type,
                  all_totals.data(), 1, mpi_type,
                  MPI_COMM_WORLD);

    // ---- Phase 3: exclusive prefix of totals → this rank's offset ---------
    T offset = identity;
    for (int r = 0; r < rank; ++r)
        offset = op(offset, all_totals[static_cast<std::size_t>(r)]);

    // ---- Phase 4: broadcast offset into every local element (OpenMP) ------
    // Rank 0: offset == identity, so no change needed.
    if (rank > 0 && local_n > 0) {
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(local_n); ++i)
            local_arr[i] = op(offset, local_arr[i]);
    }
}

// ---------------------------------------------------------------------------
// exclusive_scan_mpi
//
// Post-condition: if base = rank * local_n, then
//   local_arr[i] = op(global[0], …, global[base + i - 1])
//   local_arr[0] on rank 0 == identity
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void exclusive_scan_mpi(T*           local_arr,
                        std::size_t  local_n,
                        T            identity,
                        Op           op,
                        MPI_Datatype mpi_type)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (local_n == 0) return;

    // Phase 1: local inclusive scan to capture total; save inclusive result.
    inclusive_scan_omp_fast(local_arr, local_n, identity, op);
    // local_arr[i] now holds incl_local[i] = orig[0] + … + orig[i]
    // local_arr[local_n-1] = local total
    const T local_total = local_arr[local_n - 1];

    // Phase 2: exchange totals.
    std::vector<T> all_totals(static_cast<std::size_t>(size));
    MPI_Allgather(&local_total,      1, mpi_type,
                  all_totals.data(), 1, mpi_type,
                  MPI_COMM_WORLD);

    // Phase 3: exclusive prefix → this rank's offset.
    T offset = identity;
    for (int r = 0; r < rank; ++r)
        offset = op(offset, all_totals[static_cast<std::size_t>(r)]);

    // Phase 4: convert local inclusive scan → global exclusive scan.
    //
    // global_excl[base+i] = op(offset, incl_local[i-1])  (i >= 1)
    // global_excl[base+0] = offset
    //
    // Read: incl_local[i-1] = local_arr[i-1]  (still holds inclusive values)
    // Write: local_arr[i]
    // Going backwards avoids overwriting a value we still need to read.
    //
    // Sequential (no dependency issue: local_arr[n-1] reads local_arr[n-2],
    // which we have not yet written in the backward pass).

    // Parallelisable version: save inclusive result, then scatter in parallel.
    std::vector<T> incl(local_arr, local_arr + local_n);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(local_n); ++i) {
        local_arr[i] = (i == 0)
                       ? offset
                       : op(offset, incl[static_cast<std::size_t>(i) - 1]);
    }
}
