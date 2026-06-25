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

// Bit-packed technique — the memory-centric thesis artifact.
// The active state-set is a packed bitmask (1 bit/state, 64-bit words) held in
// thread-local registers instead of an int8-per-state buffer in (global-backed)
// local memory. Templating on NWORDS makes the working set a compile-time array:
// for num_states <= 64 (NWORDS==1) it is a single register-resident
// `unsigned long long` with zero global traffic for the state vector — exactly the
// byte->bit + global->register ablation. Same CSR algorithm as the dense kernel.
static constexpr int BITPACKED_MAX_WORDS = 8;  // up to 512 states

template <int NWORDS>
__global__ void bitpacked_nfa_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int* out_flag, int* out_len) {

    unsigned long long cur[NWORDS];
    unsigned long long nxt[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) cur[w] = 0ULL;
    cur[start_state >> 6] |= (1ULL << (start_state & 63));

    // Epsilon closure: num_states passes guarantee convergence.
    for (int it = 0; it < num_states; ++it) {
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    cur[t >> 6] |= (1ULL << (t & 63));
                }
            }
        }
    }

    int out_f = 0, out_l = 0, done = 0;
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nxt[w] = 0ULL;
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
        for (int it = 0; it < num_states; ++it) {
            for (int s = 0; s < num_states; ++s) {
                if (nxt[s >> 6] & (1ULL << (s & 63))) {
                    for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                        int t = eps_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) cur[w] = nxt[w];
        int m = 0;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    *out_flag = out_f; *out_len = out_l;
}

// Multi-stream technique — single->multi-stream ablation axis.
// One block (one thread) per input string: blockIdx.x selects the string,
// strings run concurrently across the SMs. Same bit-packed register-resident
// working set; the read-only CSR is shared by all blocks. Inputs are passed as a
// single concatenated buffer + per-string offsets (CSR-of-strings).
template <int NWORDS>
__global__ void bitpacked_multistream_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int* out_flags, int* out_lens) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    const int* input_symbols = input_data + input_offsets[i];
    int input_len = input_offsets[i + 1] - input_offsets[i];

    unsigned long long cur[NWORDS];
    unsigned long long nxt[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) cur[w] = 0ULL;
    cur[start_state >> 6] |= (1ULL << (start_state & 63));

    for (int it = 0; it < num_states; ++it) {
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    cur[t >> 6] |= (1ULL << (t & 63));
                }
            }
        }
    }

    int out_f = 0, out_l = 0, done = 0;
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nxt[w] = 0ULL;
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
        for (int it = 0; it < num_states; ++it) {
            for (int s = 0; s < num_states; ++s) {
                if (nxt[s >> 6] & (1ULL << (s & 63))) {
                    for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                        int t = eps_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) cur[w] = nxt[w];
        int m = 0;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
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

// Launch the bitpacked kernel specialized for the NFA's word count.
static void launch_bitpacked(
    int nwords,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, int input_len,
    int num_states, int start_state, int uses_any, int* flag, int* len) {
    switch (nwords) {
        case 1: bitpacked_nfa_kernel<1><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 2: bitpacked_nfa_kernel<2><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 3: bitpacked_nfa_kernel<3><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 4: bitpacked_nfa_kernel<4><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 5: bitpacked_nfa_kernel<5><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 6: bitpacked_nfa_kernel<6><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 7: bitpacked_nfa_kernel<7><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 8: bitpacked_nfa_kernel<8><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        default:
            throw std::runtime_error("bitpacked: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
}

// Returns (accepted, match_len, kernel_ms). accept_words is the packed accept set.
static std::tuple<bool, int, float> run_bitpacked(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    std::vector<void*> frees;
    const int* d_srp = dev_copy(sym_row_ptr, frees);
    const int* d_st = dev_copy(sym_targets, frees);
    const int* d_ss = dev_copy(sym_symbols, frees);
    const int* d_erp = dev_copy(eps_row_ptr, frees);
    const int* d_et = dev_copy(eps_targets, frees);
    const unsigned long long* d_acc = dev_copy(accept_words, frees);
    const int* d_in = dev_copy(input_symbols, frees);
    int input_len = static_cast<int>(input_symbols.request().size);

    int *d_flag, *d_len;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int))); CUDA_CHECK(cudaMalloc(&d_len, sizeof(int)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_bitpacked(nwords, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, input_len,
                     num_states, start_state, uses_any, d_flag, d_len);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    int h_flag = 0, h_len = 0;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_len, d_len, sizeof(int), cudaMemcpyDeviceToHost);

    for (void* p : frees) cudaFree(p);
    cudaFree(d_flag); cudaFree(d_len);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {h_flag != 0, h_len, kernel_ms};
}

static void launch_multistream(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int* flags, int* lens) {
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_MS(NW) bitpacked_multistream_kernel<NW><<<blocks, threads>>>( \
        srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any, flags, lens)
    switch (nwords) {
        case 1: LAUNCH_MS(1); break;
        case 2: LAUNCH_MS(2); break;
        case 3: LAUNCH_MS(3); break;
        case 4: LAUNCH_MS(4); break;
        case 5: LAUNCH_MS(5); break;
        case 6: LAUNCH_MS(6); break;
        case 7: LAUNCH_MS(7); break;
        case 8: LAUNCH_MS(8); break;
        default:
            throw std::runtime_error("multistream: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_MS
}

// Returns (flags, lens, kernel_ms) for a batch of strings.
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;

    std::vector<void*> frees;
    const int* d_srp = dev_copy(sym_row_ptr, frees);
    const int* d_st = dev_copy(sym_targets, frees);
    const int* d_ss = dev_copy(sym_symbols, frees);
    const int* d_erp = dev_copy(eps_row_ptr, frees);
    const int* d_et = dev_copy(eps_targets, frees);
    const unsigned long long* d_acc = dev_copy(accept_words, frees);
    const int* d_in = dev_copy(input_data, frees);
    const int* d_off = dev_copy(input_offsets, frees);

    int *d_flags, *d_lens;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * (num_strings ? num_strings : 1)));
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * (num_strings ? num_strings : 1)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        launch_multistream(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off,
                           num_states, start_state, uses_any, d_flags, d_lens);
        CUDA_CHECK(cudaGetLastError());
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    py::array_t<int> flags(num_strings);
    py::array_t<int> lens(num_strings);
    if (num_strings > 0) {
        cudaMemcpy(flags.request().ptr, d_flags, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
        cudaMemcpy(lens.request().ptr, d_lens, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
    }

    for (void* p : frees) cudaFree(p);
    cudaFree(d_flags); cudaFree(d_lens);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {flags, lens, kernel_ms};
}

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "gpufsm CUDA backend (dense + bit-packed + multi-stream CSR NFA kernels)";
    m.def("run_dense", &run_dense,
          "Simulate an NFA (CSR, int8 working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_bitpacked", &run_bitpacked,
          "Simulate an NFA (CSR, packed-bitmask working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_multistream", &run_multistream,
          "Simulate an NFA over a batch (one block/string); returns (flags, lens, kernel_ms).");
}
