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
#include <cstring>
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

// Core single-string bit-packed NFA simulation. The CSR/accept pointers may live
// in GLOBAL **or** SHARED memory — the code is identical, which is exactly what
// lets the global->shared CSR ablation reuse one implementation. Working set is a
// register-resident NWORDS-word bitmask. latch-first-match.
template <int NWORDS>
__device__ __forceinline__ void simulate_one(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int& out_f, int& out_l) {

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

    out_f = 0; out_l = 0; int done = 0;
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
}

template <int NWORDS>
__global__ void bitpacked_nfa_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int* out_flag, int* out_len) {
    int out_f, out_l;
    simulate_one<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                         accept_words, input_symbols, input_len,
                         num_states, start_state, uses_any, out_f, out_l);
    *out_flag = out_f; *out_len = out_l;
}

// Work-efficient frontier epsilon-closure: expand only NEW states (set bits in
// `frontier`) into `set`, until no new states appear. O(reachable) not O(n^2).
template <int NWORDS>
__device__ __forceinline__ void eps_closure_worklist(
    unsigned long long set[NWORDS], const int* eps_row_ptr, const int* eps_targets) {
    unsigned long long frontier[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) frontier[w] = set[w];
    bool any = true;
    while (any) {
        unsigned long long nb[NWORDS];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nb[w] = 0ULL;
        for (int w = 0; w < NWORDS; ++w) {
            unsigned long long b = frontier[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    nb[t >> 6] |= (1ULL << (t & 63));
                }
            }
        }
        any = false;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) {
            nb[w] &= ~set[w];          // keep only genuinely new states
            set[w] |= nb[w];
            frontier[w] = nb[w];
            if (nb[w]) any = true;
        }
    }
}

// Work-efficient single-string simulation: iterate only ACTIVE states (set bits),
// not all num_states, and use a frontier epsilon-closure. Same verdict as
// simulate_one (latch-first-match) but O(active) per symbol instead of O(n^2) —
// the kernel that moves the workload toward the memory-bound regime where the
// memory-layout techniques matter.
template <int NWORDS>
__device__ __forceinline__ void simulate_one_worklist(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int& out_f, int& out_l) {
    (void)num_states;
    unsigned long long cur[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) cur[w] = 0ULL;
    cur[start_state >> 6] |= (1ULL << (start_state & 63));
    eps_closure_worklist<NWORDS>(cur, eps_row_ptr, eps_targets);

    out_f = 0; out_l = 0; int done = 0;
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        unsigned long long nxt[NWORDS];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nxt[w] = 0ULL;
        for (int w = 0; w < NWORDS; ++w) {
            unsigned long long b = cur[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
        eps_closure_worklist<NWORDS>(nxt, eps_row_ptr, eps_targets);
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) cur[w] = nxt[w];
        int m = 0;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
}

// Multi-stream worklist: one thread/string, work-efficient kernel, global CSR.
// __launch_bounds__ raises occupancy for SMALL working sets (NWORDS<=2: cap registers
// to fit 6 blocks/SM — the kernel is latency-bound, so more resident warps hide latency,
// ~2x at <=64 states). For larger NWORDS the same cap forces register spills and hurts,
// so we relax to minBlocks=1 (effectively unconstrained). NWORDS is a compile-time
// template parameter, so the ternary is a constant expression.
template <int NWORDS>
__global__ void __launch_bounds__(256, (NWORDS <= 2 ? 6 : 1)) worklist_multistream_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int* out_flags, int* out_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int out_f, out_l;
    simulate_one_worklist<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                                  accept_words, input_data + input_offsets[i],
                                  input_offsets[i + 1] - input_offsets[i],
                                  num_states, start_state, uses_any, out_f, out_l);
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Frontier epsilon-closure over a GLOBAL-memory bitset (nwords words). S is the set
// being closed; F (frontier) and B (new bits) are per-thread scratch slices.
__device__ __forceinline__ void eps_closure_global(
    unsigned long long* S, unsigned long long* F, unsigned long long* B, int nwords,
    const int* eps_row_ptr, const int* eps_targets) {
    for (int w = 0; w < nwords; ++w) F[w] = S[w];
    bool any = true;
    while (any) {
        for (int w = 0; w < nwords; ++w) B[w] = 0ULL;
        for (int w = 0; w < nwords; ++w) {
            unsigned long long b = F[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    B[t >> 6] |= (1ULL << (t & 63));
                }
            }
        }
        any = false;
        for (int w = 0; w < nwords; ++w) {
            B[w] &= ~S[w];
            S[w] |= B[w];
            F[w] = B[w];
            if (B[w]) any = true;
        }
    }
}

// Work-efficient worklist with the working set in GLOBAL memory — NO state-count cap
// (the register worklist is capped at 512). nwords words per string; cur/nxt/frontier/
// newb are per-string global slices. One thread per string. Same latch-first-match
// verdict as the reference. This is what scales the engine to large (ANMLZoo-sized) automata.
__global__ void worklist_global_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any, int nwords,
    unsigned long long* cur, unsigned long long* nxt,
    unsigned long long* frontier, unsigned long long* newb,
    int* out_flags, int* out_lens) {
    (void)num_states;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    size_t off = (size_t)i * nwords;
    unsigned long long* C = cur + off;
    unsigned long long* N = nxt + off;
    unsigned long long* F = frontier + off;
    unsigned long long* B = newb + off;
    const int* input_symbols = input_data + input_offsets[i];
    int input_len = input_offsets[i + 1] - input_offsets[i];

    for (int w = 0; w < nwords; ++w) C[w] = 0ULL;
    C[start_state >> 6] |= (1ULL << (start_state & 63));
    eps_closure_global(C, F, B, nwords, eps_row_ptr, eps_targets);

    int out_f = 0, out_l = 0, done = 0;
    for (int w = 0; w < nwords; ++w) if (C[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        for (int w = 0; w < nwords; ++w) N[w] = 0ULL;
        for (int w = 0; w < nwords; ++w) {
            unsigned long long b = C[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        N[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
        eps_closure_global(N, F, B, nwords, eps_row_ptr, eps_targets);
        for (int w = 0; w < nwords; ++w) C[w] = N[w];
        int m = 0;
        for (int w = 0; w < nwords; ++w) if (C[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// ---- Compacted active-ID worklist (one thread/string) ------------------------------
// The bitmap worklist iterates all nwords words per symbol even when the active set is
// tiny (measured: brill averages ~1.5 active states over 667 words -> ~99.8% wasted scan).
// This kernel keeps the active set as a COMPACTED array of state IDs (frontier), so per
// symbol it does O(active) work, not O(nwords). A per-string `visited` bitmap dedups the
// next frontier; we clear only the touched bits (O(active)), never the whole bitmap, so the
// per-symbol cost is independent of num_states. One thread per string (isolates the
// compaction effect vs worklist_global; both are 1-thread/string). Same latch-first-match.
// frontier_a/frontier_b are int32[num_states] per string; visited is ull[nwords] per string.
__device__ __forceinline__ int eps_closure_compact(
    int* F, int nf, unsigned long long* V,
    const int* eps_row_ptr, const int* eps_targets) {
    // BFS expansion in place: F[0..nf) is the queue; appends grow nf. V dedups.
    for (int j = 0; j < nf; ++j) {
        int s = F[j];
        for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
            int t = eps_targets[k];
            if (!((V[t >> 6] >> (t & 63)) & 1ULL)) {
                V[t >> 6] |= (1ULL << (t & 63));
                F[nf++] = t;
            }
        }
    }
    return nf;
}

__global__ void worklist_compact_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any, int nwords,
    int* frontier_a, int* frontier_b, unsigned long long* visited,
    int* out_flags, int* out_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int* FA = frontier_a + (size_t)i * num_states;
    int* FB = frontier_b + (size_t)i * num_states;
    unsigned long long* V = visited + (size_t)i * nwords;
    const int* input_symbols = input_data + input_offsets[i];
    int input_len = input_offsets[i + 1] - input_offsets[i];

    for (int w = 0; w < nwords; ++w) V[w] = 0ULL;  // one-time O(nwords) zero
    int nf = 0;
    V[start_state >> 6] |= (1ULL << (start_state & 63));
    FA[nf++] = start_state;
    nf = eps_closure_compact(FA, nf, V, eps_row_ptr, eps_targets);

    int out_f = 0, out_l = 0, done = 0;
    for (int j = 0; j < nf && !done; ++j)
        if ((accept_words[FA[j] >> 6] >> (FA[j] & 63)) & 1ULL) { out_f = 1; out_l = 0; done = 1; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        // clear visited bits of the current frontier (O(active)) -> V all-zero
        for (int j = 0; j < nf; ++j) V[FA[j] >> 6] &= ~(1ULL << (FA[j] & 63));
        int nfb = 0;
        for (int j = 0; j < nf; ++j) {
            int s = FA[j];
            for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                int tsym = sym_symbols[k];
                if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                    int t = sym_targets[k];
                    if (!((V[t >> 6] >> (t & 63)) & 1ULL)) {
                        V[t >> 6] |= (1ULL << (t & 63));
                        FB[nfb++] = t;
                    }
                }
            }
        }
        nfb = eps_closure_compact(FB, nfb, V, eps_row_ptr, eps_targets);
        for (int j = 0; j < nfb; ++j)
            if ((accept_words[FB[j] >> 6] >> (FB[j] & 63)) & 1ULL) {
                out_f = 1; out_l = pos + 1; done = 1; break;
            }
        int* tmp = FA; FA = FB; FB = tmp;  // swap frontiers
        nf = nfb;
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// ---- Block-parallel (warp-per-string) work-efficient worklist ----------------------
// One *warp* (32 lanes) cooperates on one string instead of one thread. The 32 lanes
// partition the nwords words of the working set (lane handles words w with w%32==lane);
// cross-word transition/epsilon scatter uses atomicOr into the shared global next-set.
// Rationale: the 1-thread/string worklist under-utilizes the GPU at small batch (Nsight:
// 17% occupancy, 2 blocks) and issues one string's loads serially. A warp spreads those
// loads across 32 lanes -> more memory-level parallelism, the path toward the memory-bound
// regime for large (ANMLZoo-scale) automata. Same latch-first-match verdict as the oracle.

// Warp-cooperative frontier epsilon-closure over a GLOBAL bitset. All 32 lanes active.
__device__ __forceinline__ void eps_closure_warp(
    unsigned long long* S, unsigned long long* F, unsigned long long* B, int nwords,
    const int* eps_row_ptr, const int* eps_targets, int lane) {
    for (int w = lane; w < nwords; w += 32) F[w] = S[w];
    __syncwarp();
    bool any = true;
    while (any) {
        for (int w = lane; w < nwords; w += 32) B[w] = 0ULL;
        __syncwarp();
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long b = F[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    atomicOr(&B[t >> 6], (1ULL << (t & 63)));
                }
            }
        }
        __syncwarp();
        int any_local = 0;
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long nb = B[w] & ~S[w];
            S[w] |= nb;
            F[w] = nb;
            if (nb) any_local = 1;
        }
        any = __any_sync(0xffffffffu, any_local) != 0;
        __syncwarp();
    }
}

__global__ void worklist_warp_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int start_state, int uses_any, int nwords,
    unsigned long long* cur, unsigned long long* nxt,
    unsigned long long* frontier, unsigned long long* newb,
    int* out_flags, int* out_lens) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= num_strings) return;
    size_t off = (size_t)warp * nwords;
    unsigned long long* C = cur + off;
    unsigned long long* N = nxt + off;
    unsigned long long* F = frontier + off;
    unsigned long long* B = newb + off;
    const int* input_symbols = input_data + input_offsets[warp];
    int input_len = input_offsets[warp + 1] - input_offsets[warp];

    for (int w = lane; w < nwords; w += 32) C[w] = 0ULL;
    __syncwarp();
    if (lane == 0) C[start_state >> 6] |= (1ULL << (start_state & 63));
    __syncwarp();
    eps_closure_warp(C, F, B, nwords, eps_row_ptr, eps_targets, lane);

    int out_f = 0, out_l = 0, done = 0, acc_local = 0;
    for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) acc_local = 1;
    if (__any_sync(0xffffffffu, acc_local)) { out_f = 1; out_l = 0; done = 1; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        for (int w = lane; w < nwords; w += 32) N[w] = 0ULL;
        __syncwarp();
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long b = C[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        atomicOr(&N[t >> 6], (1ULL << (t & 63)));
                    }
                }
            }
        }
        __syncwarp();
        eps_closure_warp(N, F, B, nwords, eps_row_ptr, eps_targets, lane);
        for (int w = lane; w < nwords; w += 32) C[w] = N[w];
        __syncwarp();
        int m_local = 0;
        for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) m_local = 1;
        if (__any_sync(0xffffffffu, m_local)) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    if (lane == 0) { out_flags[warp] = out_f; out_lens[warp] = out_l; }
}

// ---- Shared-memory block-cooperative worklist -------------------------------------
// Same warp-per-string scheme as worklist_warp, but the per-string working set
// (cur/nxt/frontier/newb, 4*nwords words) lives in DYNAMIC SHARED memory instead of
// global. This privatizes the working-set traffic the warp kernel issues to global:
// a test of whether memory-layout privatization helps once the kernel is work-efficient
// (it did NOT in the compute-bound full-scan regime; see multistream_shared). Requires
// warps_per_block * 4 * nwords * 8 bytes of shared memory; the launcher picks
// warps_per_block to fit 48 KB and the technique is only offered when 1 warp/block fits.
__global__ void worklist_shared_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int start_state, int uses_any, int nwords,
    int* out_flags, int* out_lens) {
    extern __shared__ unsigned long long wl_smem[];
    int warps_per_block = blockDim.x >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp = blockIdx.x * warps_per_block + warp_in_block;
    if (warp >= num_strings) return;
    unsigned long long* base = wl_smem + (size_t)warp_in_block * 4 * nwords;
    unsigned long long* C = base;
    unsigned long long* N = base + nwords;
    unsigned long long* F = base + 2 * nwords;
    unsigned long long* B = base + 3 * nwords;
    const int* input_symbols = input_data + input_offsets[warp];
    int input_len = input_offsets[warp + 1] - input_offsets[warp];

    for (int w = lane; w < nwords; w += 32) C[w] = 0ULL;
    __syncwarp();
    if (lane == 0) C[start_state >> 6] |= (1ULL << (start_state & 63));
    __syncwarp();
    eps_closure_warp(C, F, B, nwords, eps_row_ptr, eps_targets, lane);

    int out_f = 0, out_l = 0, done = 0, acc_local = 0;
    for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) acc_local = 1;
    if (__any_sync(0xffffffffu, acc_local)) { out_f = 1; out_l = 0; done = 1; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        for (int w = lane; w < nwords; w += 32) N[w] = 0ULL;
        __syncwarp();
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long b = C[w];
            while (b) {
                int s = w * 64 + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        atomicOr(&N[t >> 6], (1ULL << (t & 63)));
                    }
                }
            }
        }
        __syncwarp();
        eps_closure_warp(N, F, B, nwords, eps_row_ptr, eps_targets, lane);
        for (int w = lane; w < nwords; w += 32) C[w] = N[w];
        __syncwarp();
        int m_local = 0;
        for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) m_local = 1;
        if (__any_sync(0xffffffffu, m_local)) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    if (lane == 0) { out_flags[warp] = out_f; out_lens[warp] = out_l; }
}

// Multi-stream technique — single->multi-stream ablation axis.
// One thread per input string (blockIdx.x*blockDim.x+threadIdx.x); strings run
// concurrently across the SMs. Read-only CSR shared by all threads, in GLOBAL memory.
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
    int out_f, out_l;
    simulate_one<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                         accept_words, input_data + input_offsets[i],
                         input_offsets[i + 1] - input_offsets[i],
                         num_states, start_state, uses_any, out_f, out_l);
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Multi-stream + global->shared CSR ablation. Each block cooperatively stages the
// entire read-only CSR (+ accept words) into shared memory once; every thread then
// reads transitions from shared instead of global. **This layout cannot be
// expressed in Triton** (its compiler owns shared memory) — the core
// abstraction-regret demonstration on the CUDA side. Shared bytes are computed and
// requested by the host launcher; layout matches the host's size computation.
template <int NWORDS>
__global__ void bitpacked_multistream_shared_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int nnz_sym, int nnz_eps,
    int* out_flags, int* out_lens) {

    extern __shared__ unsigned char smem[];
    unsigned long long* s_acc = reinterpret_cast<unsigned long long*>(smem);
    int* s_srp = reinterpret_cast<int*>(s_acc + NWORDS);
    int* s_st  = s_srp + (num_states + 1);
    int* s_ss  = s_st + nnz_sym;
    int* s_erp = s_ss + nnz_sym;
    int* s_et  = s_erp + (num_states + 1);

    for (int j = threadIdx.x; j < NWORDS; j += blockDim.x) s_acc[j] = accept_words[j];
    for (int j = threadIdx.x; j < num_states + 1; j += blockDim.x) {
        s_srp[j] = sym_row_ptr[j];
        s_erp[j] = eps_row_ptr[j];
    }
    for (int j = threadIdx.x; j < nnz_sym; j += blockDim.x) {
        s_st[j] = sym_targets[j];
        s_ss[j] = sym_symbols[j];
    }
    for (int j = threadIdx.x; j < nnz_eps; j += blockDim.x) s_et[j] = eps_targets[j];
    __syncthreads();  // all threads must reach this before the bounds check / return

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int out_f, out_l;
    simulate_one<NWORDS>(s_srp, s_st, s_ss, s_erp, s_et, s_acc,
                         input_data + input_offsets[i],
                         input_offsets[i + 1] - input_offsets[i],
                         num_states, start_state, uses_any, out_f, out_l);
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

// Bytes of dynamic shared memory the shared-CSR kernel needs for this NFA.
static size_t shared_csr_bytes(int nwords, int num_states, int nnz_sym, int nnz_eps) {
    return static_cast<size_t>(nwords) * sizeof(unsigned long long)
         + static_cast<size_t>(num_states + 1) * sizeof(int) * 2
         + static_cast<size_t>(nnz_sym) * sizeof(int) * 2
         + static_cast<size_t>(nnz_eps) * sizeof(int);
}

static void launch_multistream_shared(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int nnz_sym, int nnz_eps,
    size_t shared_bytes, int* flags, int* lens) {
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_MSS(NW)                                                                              \
    do {                                                                                           \
        if (shared_bytes > 48 * 1024)                                                              \
            CUDA_CHECK(cudaFuncSetAttribute(bitpacked_multistream_shared_kernel<NW>,               \
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));     \
        bitpacked_multistream_shared_kernel<NW><<<blocks, threads, shared_bytes>>>(                \
            srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any,    \
            nnz_sym, nnz_eps, flags, lens);                                                        \
    } while (0)
    switch (nwords) {
        case 1: LAUNCH_MSS(1); break;
        case 2: LAUNCH_MSS(2); break;
        case 3: LAUNCH_MSS(3); break;
        case 4: LAUNCH_MSS(4); break;
        case 5: LAUNCH_MSS(5); break;
        case 6: LAUNCH_MSS(6); break;
        case 7: LAUNCH_MSS(7); break;
        case 8: LAUNCH_MSS(8); break;
        default:
            throw std::runtime_error("multistream_shared: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_MSS
}

// Multi-stream + shared-CSR. Returns (flags, lens, kernel_ms). Throws if the CSR
// does not fit the device's opt-in shared-memory budget.
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    int nnz_sym = static_cast<int>(sym_targets.request().size);
    int nnz_eps = static_cast<int>(eps_targets.request().size);
    size_t shared_bytes = shared_csr_bytes(nwords, num_states, nnz_sym, nnz_eps);

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (static_cast<int>(shared_bytes) > max_optin) {
        throw std::runtime_error("multistream_shared: CSR needs " + std::to_string(shared_bytes) +
            " B shared mem > device opt-in max " + std::to_string(max_optin) + " B");
    }

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
        launch_multistream_shared(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc,
                                  d_in, d_off, num_states, start_state, uses_any, nnz_sym, nnz_eps,
                                  shared_bytes, d_flags, d_lens);
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

// Multi-stream + sync→async transfer ablation. Pinned host staging + N CUDA streams
// pipeline H2D(chunk) -> kernel(chunk) -> D2H(chunk) so input transfer overlaps
// compute (and vice-versa), hiding PCIe latency that a single blocking cudaMemcpy
// would expose. The read-only CSR is copied once up front; per-chunk launches reuse
// the global-CSR kernel by shifting the offsets/output pointers (offsets are absolute
// into input_data, so no kernel change is needed). Returns (flags, lens, total_ms)
// where total_ms is the overlapped end-to-end device time.
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_async(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    constexpr int N_STREAMS = 4;
    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    int in_len = static_cast<int>(input_data.request().size);

    py::array_t<int> flags(num_strings < 0 ? 0 : num_strings);
    py::array_t<int> lens(num_strings < 0 ? 0 : num_strings);
    if (num_strings <= 0) return {flags, lens, 0.0f};

    // CSR copied once (small, read-only).
    std::vector<void*> frees;
    const int* d_srp = dev_copy(sym_row_ptr, frees);
    const int* d_st = dev_copy(sym_targets, frees);
    const int* d_ss = dev_copy(sym_symbols, frees);
    const int* d_erp = dev_copy(eps_row_ptr, frees);
    const int* d_et = dev_copy(eps_targets, frees);
    const unsigned long long* d_acc = dev_copy(accept_words, frees);

    // Pin the caller's host buffers IN PLACE (cudaHostRegister) instead of allocating
    // a second pinned buffer and copying into it — that extra full-input host memcpy
    // would dwarf any overlap benefit. Outputs are pinned likewise for async D2H.
    int* h_in = static_cast<int*>(input_data.request().ptr);
    int* h_off = static_cast<int*>(input_offsets.request().ptr);
    int* h_flags = static_cast<int*>(flags.request().ptr);
    int* h_lens = static_cast<int*>(lens.request().ptr);
    if (in_len) CUDA_CHECK(cudaHostRegister(h_in, sizeof(int) * in_len, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_flags, sizeof(int) * num_strings, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_lens, sizeof(int) * num_strings, cudaHostRegisterDefault));

    int *d_in, *d_off, *d_flags, *d_lens;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * (in_len ? in_len : 1)));
    CUDA_CHECK(cudaMalloc(&d_off, sizeof(int) * (num_strings + 1)));
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * num_strings));
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * num_strings));

    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Offsets are needed device-side; copy once (cheap) before the pipeline.
    CUDA_CHECK(cudaMemcpy(d_off, h_off, sizeof(int) * (num_strings + 1), cudaMemcpyHostToDevice));

    int chunk = (num_strings + N_STREAMS - 1) / N_STREAMS;
    int threads = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int c = 0; c < N_STREAMS; ++c) {
        int lo = c * chunk;
        if (lo >= num_strings) break;
        int hi = lo + chunk; if (hi > num_strings) hi = num_strings;
        int nstr = hi - lo;
        int byte_lo = h_off[lo], byte_hi = h_off[hi];
        cudaStream_t st = streams[c % N_STREAMS];
        if (byte_hi > byte_lo) {
            CUDA_CHECK(cudaMemcpyAsync(d_in + byte_lo, h_in + byte_lo,
                sizeof(int) * (byte_hi - byte_lo), cudaMemcpyHostToDevice, st));
        }
        int blocks = (nstr + threads - 1) / threads;
#define LAUNCH_ASYNC(NW) bitpacked_multistream_kernel<NW><<<blocks, threads, 0, st>>>( \
        d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off + lo, nstr, \
        num_states, start_state, uses_any, d_flags + lo, d_lens + lo)
        switch (nwords) {
            case 1: LAUNCH_ASYNC(1); break;
            case 2: LAUNCH_ASYNC(2); break;
            case 3: LAUNCH_ASYNC(3); break;
            case 4: LAUNCH_ASYNC(4); break;
            case 5: LAUNCH_ASYNC(5); break;
            case 6: LAUNCH_ASYNC(6); break;
            case 7: LAUNCH_ASYNC(7); break;
            case 8: LAUNCH_ASYNC(8); break;
            default:
                throw std::runtime_error("multistream_async: num_states > " +
                    std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported");
        }
#undef LAUNCH_ASYNC
        CUDA_CHECK(cudaMemcpyAsync(h_flags + lo, d_flags + lo, sizeof(int) * nstr,
            cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_lens + lo, d_lens + lo, sizeof(int) * nstr,
            cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float total_ms = 0.0f; cudaEventElapsedTime(&total_ms, start, stop);

    // h_flags/h_lens ARE the numpy output buffers — async D2H already wrote them.
    for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
    for (void* p : frees) cudaFree(p);
    cudaFree(d_in); cudaFree(d_off); cudaFree(d_flags); cudaFree(d_lens);
    if (in_len) cudaHostUnregister(h_in);
    cudaHostUnregister(h_flags); cudaHostUnregister(h_lens);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {flags, lens, total_ms};
}

static void launch_worklist(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int* flags, int* lens) {
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_WL(NW) worklist_multistream_kernel<NW><<<blocks, threads>>>( \
        srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any, flags, lens)
    switch (nwords) {
        case 1: LAUNCH_WL(1); break;
        case 2: LAUNCH_WL(2); break;
        case 3: LAUNCH_WL(3); break;
        case 4: LAUNCH_WL(4); break;
        case 5: LAUNCH_WL(5); break;
        case 6: LAUNCH_WL(6); break;
        case 7: LAUNCH_WL(7); break;
        case 8: LAUNCH_WL(8); break;
        default:
            throw std::runtime_error("worklist: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_WL
}

// Work-efficient multi-stream. Returns (flags, lens, kernel_ms).
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist(
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
        launch_worklist(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off,
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

// Work-efficient worklist with a GLOBAL working set — no state-count cap. Returns
// (flags, lens, kernel_ms). accept_words has nwords = ceil(num_states/64) entries.
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_global(
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
    unsigned long long *d_cur, *d_nxt, *d_fr, *d_nb;
    size_t ws = sizeof(unsigned long long) * (size_t)(num_strings ? num_strings : 1) * nwords;
    CUDA_CHECK(cudaMalloc(&d_cur, ws)); CUDA_CHECK(cudaMalloc(&d_nxt, ws));
    CUDA_CHECK(cudaMalloc(&d_fr, ws)); CUDA_CHECK(cudaMalloc(&d_nb, ws));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        worklist_global_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            num_states, start_state, uses_any, nwords, d_cur, d_nxt, d_fr, d_nb, d_flags, d_lens);
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
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_fr); cudaFree(d_nb);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return {flags, lens, kernel_ms};
}

// Block-parallel (warp-per-string) worklist — same global working set as run_worklist_global,
// but launches 32 lanes per string. Returns (flags, lens, kernel_ms).
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_warp(
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
    unsigned long long *d_cur, *d_nxt, *d_fr, *d_nb;
    size_t ws = sizeof(unsigned long long) * (size_t)(num_strings ? num_strings : 1) * nwords;
    CUDA_CHECK(cudaMalloc(&d_cur, ws)); CUDA_CHECK(cudaMalloc(&d_nxt, ws));
    CUDA_CHECK(cudaMalloc(&d_fr, ws)); CUDA_CHECK(cudaMalloc(&d_nb, ws));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        int threads = 256;  // 8 warps/block
        int blocks = (num_strings * 32 + threads - 1) / threads;
        worklist_warp_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            start_state, uses_any, nwords, d_cur, d_nxt, d_fr, d_nb, d_flags, d_lens);
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
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_fr); cudaFree(d_nb);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return {flags, lens, kernel_ms};
}

// Compacted active-ID worklist (one thread/string): O(active) per symbol, not O(nwords).
// Returns (flags, lens, kernel_ms).
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_compact(
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

    int *d_flags, *d_lens, *d_fa, *d_fb;
    unsigned long long* d_vis;
    int ns1 = num_strings ? num_strings : 1;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * ns1));
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * ns1));
    CUDA_CHECK(cudaMalloc(&d_fa, sizeof(int) * (size_t)ns1 * num_states));
    CUDA_CHECK(cudaMalloc(&d_fb, sizeof(int) * (size_t)ns1 * num_states));
    CUDA_CHECK(cudaMalloc(&d_vis, sizeof(unsigned long long) * (size_t)ns1 * nwords));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        worklist_compact_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            num_states, start_state, uses_any, nwords, d_fa, d_fb, d_vis, d_flags, d_lens);
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
    cudaFree(d_flags); cudaFree(d_lens); cudaFree(d_fa); cudaFree(d_fb); cudaFree(d_vis);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return {flags, lens, kernel_ms};
}

// Shared-memory block-cooperative worklist. Working set in dynamic shared memory; the
// launcher picks warps_per_block so warps_per_block*4*nwords*8 fits 48 KB. Raises if even
// one warp's working set (4*nwords*8 bytes) exceeds 48 KB. Returns (flags, lens, kernel_ms).
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    const size_t SMEM_CAP = 48 * 1024;  // bytes
    size_t per_warp = (size_t)4 * nwords * sizeof(unsigned long long);
    if (per_warp > SMEM_CAP) {
        throw std::runtime_error(
            "worklist_shared: working set (" + std::to_string(per_warp) +
            " B) exceeds 48 KB shared memory; use worklist_warp/worklist_global for "
            "num_states > ~1536.");
    }
    int warps_per_block = 1;
    while (warps_per_block < 8 && (warps_per_block + 1) * per_warp <= SMEM_CAP) warps_per_block++;
    int threads = warps_per_block * 32;
    size_t shared_bytes = (size_t)warps_per_block * per_warp;

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
        int blocks = (num_strings + warps_per_block - 1) / warps_per_block;
        worklist_shared_kernel<<<blocks, threads, shared_bytes>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            start_state, uses_any, nwords, d_flags, d_lens);
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

// DFA simulation — the MEMORY-bound automata workload. One thread per string walks a
// dense transition table: cur = trans[cur*256 + symbol] per byte (a random global lookup;
// for large DFAs the table exceeds cache -> memory-bound). latch-first-match.
__global__ void dfa_kernel(
    const int* trans, const signed char* accept,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state,
    int* out_flags, int* out_lens) {
    (void)num_states;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    const int* in = input_data + input_offsets[i];
    int len = input_offsets[i + 1] - input_offsets[i];
    int cur = start_state, out_f = 0, out_l = 0;
    if (accept[cur]) {
        out_f = 1;
    } else {
        for (int p = 0; p < len; ++p) {
            cur = trans[cur * 256 + in[p]];
            if (accept[cur]) { out_f = 1; out_l = p + 1; break; }
        }
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Returns (flags, lens, kernel_ms) for a batch over a DFA.
static std::tuple<py::array_t<int>, py::array_t<int>, float> run_dfa(
    py::array_t<int> trans, py::array_t<signed char> accept,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state) {

    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    std::vector<void*> frees;
    const int* d_trans = dev_copy(trans, frees);
    const signed char* d_acc = dev_copy(accept, frees);
    const int* d_in = dev_copy(input_data, frees);
    const int* d_off = dev_copy(input_offsets, frees);

    int *d_flags, *d_lens;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * (num_strings ? num_strings : 1)));
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * (num_strings ? num_strings : 1)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        dfa_kernel<<<blocks, threads>>>(d_trans, d_acc, d_in, d_off, num_strings,
                                        num_states, start_state, d_flags, d_lens);
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
    m.doc() = "gpufsm CUDA backend (dense + bit-packed + multi-stream [+ shared-CSR/async/worklist] NFA kernels)";
    m.def("run_dense", &run_dense,
          "Simulate an NFA (CSR, int8 working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_bitpacked", &run_bitpacked,
          "Simulate an NFA (CSR, packed-bitmask working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_multistream", &run_multistream,
          "Simulate an NFA over a batch (one thread/string, global CSR); returns (flags, lens, kernel_ms).");
    m.def("run_multistream_shared", &run_multistream_shared,
          "Multi-stream with read-only CSR staged into shared memory; returns (flags, lens, kernel_ms).");
    m.def("run_multistream_async", &run_multistream_async,
          "Multi-stream with pinned host staging + streamed async H2D/kernel/D2H overlap; "
          "returns (flags, lens, total_ms).");
    m.def("run_worklist", &run_worklist,
          "Work-efficient multi-stream (iterate active states + frontier eps-closure); "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_global", &run_worklist_global,
          "Work-efficient worklist with a global working set — no state-count cap; "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_warp", &run_worklist_warp,
          "Block-parallel (warp-per-string) work-efficient worklist; no state-count cap; "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_compact", &run_worklist_compact,
          "Compacted active-ID worklist (one thread/string; O(active) per symbol, no "
          "O(nwords) bitmap scan); returns (flags, lens, kernel_ms).");
    m.def("run_worklist_shared", &run_worklist_shared,
          "Shared-memory block-cooperative worklist (working set in dynamic shared mem; "
          "num_states up to ~1536); returns (flags, lens, kernel_ms).");
    m.def("run_dfa", &run_dfa,
          "DFA simulation (dense transition-table lookup per byte, memory-bound); "
          "returns (flags, lens, kernel_ms).");
}
