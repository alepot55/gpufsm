// Shared definitions for the gpufsm CUDA translation units.
//
// The kernels are split one family per .cu (dense, bit-packed, worklist, DFA) and
// compiled WITHOUT relocatable device code, so a __device__ function cannot be called
// across translation units. Anything used by more than one family therefore lives in a
// header and is included into each TU that needs it; anything used by exactly one
// family stays in that family's .cu. Keep it that way: moving a __device__ helper into
// a .cu that another TU calls it from is a link error, not a warning.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Wildcard symbol id: a transition labelled with it matches any input byte.
// Must stay in sync with gpufsm.core.nfa.ANY_SYMBOL.
static constexpr int ANY_SYMBOL = 256;

// Packed working set: 64-bit words, so 8 words covers up to 512 states.
static constexpr int BITPACKED_MAX_WORDS = 8;

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error at ") +          \
                __FILE__ ":" + std::to_string(__LINE__) + " -> " +            \
                cudaGetErrorString(_err));                                    \
        }                                                                     \
    } while (0)

// Copies a numpy array to the device; the pointer is owned by the caller's `frees`
// list, which every entry point drains before returning.
//
// KNOWN ISSUE: `frees` is drained only on the happy path, so a CUDA_CHECK throw between
// the allocation and the drain leaks every buffer allocated so far. Replacing it with an
// RAII owner is a mechanical change but needs a CUDA toolchain to verify, so it is
// deliberately not bundled with the file split.
template <typename T>
static const T* dev_copy(const py::array_t<T>& a, std::vector<void*>& frees) {
    auto buf = a.request();
    T* d = nullptr;
    size_t bytes = static_cast<size_t>(buf.size) * sizeof(T);
    cudaMalloc(&d, bytes ? bytes : 1);
    if (bytes) cudaMemcpy(d, buf.ptr, bytes, cudaMemcpyHostToDevice);
    frees.push_back(d);
    return d;
}
