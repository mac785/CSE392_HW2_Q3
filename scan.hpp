#pragma once

#include <cstddef>
#include <functional>

// Annotate Vec3 members as callable from both host and device when compiled
// by nvcc; expands to nothing for ordinary C++ compilers.
#ifdef __CUDACC__
#  define SCAN_HD __device__ __host__
#else
#  define SCAN_HD
#endif

// ---------------------------------------------------------------------------
// Vec3 — minimal 3-component vector usable as a scan element type
// ---------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;

    SCAN_HD Vec3() : x(0.f), y(0.f), z(0.f) {}
    SCAN_HD Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    SCAN_HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    SCAN_HD Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    SCAN_HD bool operator==(const Vec3& o) const { return x == o.x && y == o.y && z == o.z; }
};

// ---------------------------------------------------------------------------
// inclusive_scan — overwrites data[i] with op(data[0], …, data[i])
//
//   T    : element type (must be copyable, op must be associative)
//   Op   : binary functor: (T, T) -> T
//   data : pointer to the array (modified in-place)
//   n    : number of elements
//   op   : the binary operation (defaults to std::plus<T>{})
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void inclusive_scan(T* data, std::size_t n, Op op = Op{})
{
    if (n == 0) return;
    for (std::size_t i = 1; i < n; ++i)
        data[i] = op(data[i - 1], data[i]);
}

// ---------------------------------------------------------------------------
// exclusive_scan — overwrites data[i] with op(identity, data[0], …, data[i-1])
//
//   identity : the identity element for op  (e.g. 0 for addition)
//   The first output element is always `identity`.
// ---------------------------------------------------------------------------
template <typename T, typename Op = std::plus<T>>
void exclusive_scan(T* data, std::size_t n, T identity, Op op = Op{})
{
    if (n == 0) return;
    T prev = identity;
    for (std::size_t i = 0; i < n; ++i) {
        T cur = data[i];
        data[i] = prev;
        prev = op(prev, cur);
    }
}
