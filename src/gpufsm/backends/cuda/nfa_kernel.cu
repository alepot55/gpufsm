// CUDA backend for gpufsm — clean, self-contained CSR/bit-packed NFA simulation.
//
// This consumes gpufsm's *new* NFA CSR arrays directly (no legacy MyNFA/ANML C++
// types), keeping the backend minimal. The "dense" technique is the faithful,
// correctness-first baseline mirroring gpufsm.reference.simulate (latch-first-match).
// Advanced techniques (shared-resident CSR, packed-bitmap warp kernels, ngAP) are
// added on top once validated on hardware.
//
// Status: structurally complete; compiled only when GPUFSM_BUILD_CUDA=ON and a CUDA
// toolkit is present. Validated on GPU via the @pytest.mark.gpu suite.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static constexpr int ANY_SYMBOL = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error at ") +          \
                __FILE__ ":" + std::to_string(__LINE__) + " -> " +            \
                cudaGetErrorString(_err));                                    \
        }                                                                     \
    } while (0)

// One thread simulates the whole single stream (baseline). Working sets are
// int8 device buffers (one slot per state), mirroring the reference algorithm.
__global__ void dense_nfa_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const signed char* accept,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    signed char* cur, signed char* nxt,
    int* out_flag, int* out_len) {

    for (int i = 0; i < num_states; ++i) cur[i] = 0;
    cur[start_state] = 1;

    // Epsilon closure: num_states passes guarantee convergence.
    for (int it = 0; it < num_states; ++it) {
        for (int s = 0; s < num_states; ++s) {
            if (cur[s]) {
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k)
                    cur[eps_targets[k]] = 1;
            }
        }
    }

    for (int s = 0; s < num_states; ++s) {
        if (cur[s] && accept[s]) { *out_flag = 1; *out_len = 0; return; }
    }

    for (int pos = 0; pos < input_len; ++pos) {
        int sym = input_symbols[pos];
        for (int i = 0; i < num_states; ++i) nxt[i] = 0;
        for (int s = 0; s < num_states; ++s) {
            if (cur[s]) {
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL))
                        nxt[sym_targets[k]] = 1;
                }
            }
        }
        for (int it = 0; it < num_states; ++it) {
            for (int s = 0; s < num_states; ++s) {
                if (nxt[s]) {
                    for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k)
                        nxt[eps_targets[k]] = 1;
                }
            }
        }
        for (int i = 0; i < num_states; ++i) cur[i] = nxt[i];
        for (int s = 0; s < num_states; ++s) {
            if (cur[s] && accept[s]) { *out_flag = 1; *out_len = pos + 1; return; }
        }
    }
    *out_flag = 0; *out_len = 0;
}

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

// Returns (accepted, match_len, kernel_ms).
static std::tuple<bool, int, float> run_dense(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<signed char> accept,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any) {

    std::vector<void*> frees;
    const int* d_srp = dev_copy(sym_row_ptr, frees);
    const int* d_st = dev_copy(sym_targets, frees);
    const int* d_ss = dev_copy(sym_symbols, frees);
    const int* d_erp = dev_copy(eps_row_ptr, frees);
    const int* d_et = dev_copy(eps_targets, frees);
    const signed char* d_acc = dev_copy(accept, frees);
    const int* d_in = dev_copy(input_symbols, frees);
    int input_len = static_cast<int>(input_symbols.request().size);

    signed char *d_cur, *d_nxt;
    int *d_flag, *d_len;
    CUDA_CHECK(cudaMalloc(&d_cur, num_states)); CUDA_CHECK(cudaMalloc(&d_nxt, num_states));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int))); CUDA_CHECK(cudaMalloc(&d_len, sizeof(int)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    dense_nfa_kernel<<<1, 1>>>(d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, input_len,
                               num_states, start_state, uses_any, d_cur, d_nxt, d_flag, d_len);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    int h_flag = 0, h_len = 0;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_len, d_len, sizeof(int), cudaMemcpyDeviceToHost);

    for (void* p : frees) cudaFree(p);
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_flag); cudaFree(d_len);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {h_flag != 0, h_len, kernel_ms};
}

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "gpufsm CUDA backend (dense CSR NFA kernel)";
    m.def("run_dense", &run_dense,
          "Simulate an NFA (CSR) over an input; returns (accepted, match_len, kernel_ms).");
}
