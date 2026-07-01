# Cover letter — ACM TACO submission (draft; user finalizes name/affiliation at submit)

To the Editors, ACM Transactions on Architecture and Code Optimization,

We submit for consideration **"From Diagnosis to Cure: Decomposing the Tile-SPMD Abstraction Regret on
Irregular Automata."**

**Contribution.** Tile/SPMD GPU DSLs (e.g., Triton) trail hand-written CUDA on irregular, data-dependent
control flow. We give a constructive, instruction-level account of *why*, and then *close* the gap inside
the real compiler. Concretely: (1) a falsifiable decomposition of the regret into a launch-configuration
artifact, a recoverable component, and an irreducible residual, each Nsight-attributed; (2) identification
of the residual as abstraction-denied intra-warp latency hiding (fewer warp-instructions than CUDA, equal
occupancy, below both roofline ceilings, yet a 15.3x dependent-load stall); (3) a **built in-compiler cure**
— a TritonGPU→LLVM pass, wired into libtriton's `make_llir`, that lowers a detected lock-step region to
per-lane retirement (each lane exits under hardware Independent Thread Scheduling), removing the
per-iteration cross-lane reduce. It is oracle-correct and **4.15x** (39x fewer issued instructions;
2.5–7.3x across trip distributions; and it fires oracle-correct on real SpMV/MoE workloads); and (4) a
generality law over eight oracle-gated irregular workloads showing the regret is created by per-step
scalar control, with a correct sign-flip negative on dense ragged attention.

**Fit for TACO.** The core is code generation and optimization: a real compiler pass, its structural
necessity (an in-tile-IR rewrite is provably blocked; we name the missing IR primitive), and a measured,
mechanism-grounded payoff — squarely in TACO's scope.

**Novelty / relation to prior work.** No prior work compares GPU *programming models* at fixed algorithm on
irregular automata, nor builds an in-compiler tile→thread lowering for a data-dependent per-lane region.
We distinguish carefully from GPU-automata algorithm work (ngAP, HybridSA, BitGen), tile-DSL layout control
(Gluon Linear Layouts, Descend), and thread/tile mixing (Prism, cuTile/Tile-IR, Tawa) in the paper.

**Originality / concurrent work.** This manuscript is original and is not under review at any other journal
or conference. A shorter *companion diagnosis* manuscript (the "two faces" characterization, cited
anonymously) is under separate review; the present paper's constructive cure, in-compiler build, and
generality law are new and non-overlapping (no dual submission of the same content).

**Reproducibility.** Every measurement is gated bit-for-bit against a CPU oracle and traces to a versioned
CSV; all figures regenerate from those CSVs by a single script; the compiler pass is reproduced from a
versioned patch against a pinned Triton commit with a documented build recipe. We intend to submit for
Artifact Evaluation.

**Suggested area / reviewers.** Expertise: GPU code generation and MLIR/LLVM lowering; Triton and tile-DSL
compilation; GPU microarchitecture and warp-level execution (SIMT/ITS, divergence, latency hiding);
irregular/automata GPU workloads. [User: add 3–5 specific non-conflicted names from your network here;
avoid current OpenAI/NVIDIA Triton maintainers we are engaging upstream, to prevent conflict.]

Thank you for your consideration.

[Author name, affiliation, contact — filled at submission; the manuscript PDF is anonymized per TACO policy.]
