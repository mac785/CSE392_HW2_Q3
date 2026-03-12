CXX      ?= g++
MPICXX   ?= mpicxx
NVCC     ?= /usr/local/cuda-12.8/bin/nvcc
CXXFLAGS  = -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
NVCCFLAGS = -O3 -std=c++17 -arch=sm_120
INCLUDES  = -I.

# -----------------------------------------------------------------------
.PHONY: all cuda clean

all: scan_test mpi_bench

cuda: gpu_bench

# Shared-memory only (no MPI dependency)
scan_test: main.cpp scan.hpp scan_omp.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# MPI + OpenMP
mpi_bench: mpi_bench.cpp scan_mpi.hpp scan_omp.hpp scan.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# GPU (CUDA 12.8, sm_120 — RTX 5070 Ti Laptop)
gpu_bench: gpu_bench.cu scan_cuda.cuh scan.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f scan_test mpi_bench gpu_bench
