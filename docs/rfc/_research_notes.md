# F1 research notes — grounding the Triton per-lane-region RFC (2026-06-30)

Sources verified via web search/fetch on 2026-06-30. Claims below are what the RFC relies on.

## Triton contribution / RFC norms
- Hierarchical governance: contributors → module maintainers → core maintainers → lead core maintainer.
  Changes to IRs/APIs/Passes are "more controversial", evaluated case-by-case by core maintainers; deep
  upstream design changes are expected to be "relatively rare". (triton CONTRIBUTING.md / governance)
- RFCs are filed as GitHub issues with a `[RFC]` title prefix. Live examples:
  `[RFC][AMD] Optimizations for Paged Attention` (issue #8281), `[RFC] Triton Support on Compiler
  Explorer` (issue #7560). 3rd Triton Developer Conference: 2025-10-21 (Microsoft, Mountain View).
  → Implication: post as a `[RFC]` issue; lead with motivation + evidence; propose the IR op but frame
    it as a discussion (IR changes are rare and need core-maintainer buy-in).
  Sources: github.com/triton-lang/triton (CONTRIBUTING.md), issues #8281, #7560.

## NVIDIA cuTile / CUDA Tile IR (CUDA 13.1, 2025)
- CUDA Tile = tile-based programming model + a Virtual ISA "CUDA Tile IR" (MLIR-based) + cuTile Python DSL.
- NVIDIA is building a **CUDA Tile IR backend for OpenAI Triton** (NVIDIA dev blog, 2025).
- That blog and the cuTile material do NOT address irregular/data-dependent control flow, per-lane/scalar
  execution, SIMT fallback, or divergent loops; the backend notes "not all Triton ops yet implemented".
  → Implication: the per-lane / data-dependent-control gap exists in BOTH Triton AND NVIDIA's Tile IR
    path. The RFC is complementary to cuTile, not redundant; it targets exactly what tile IRs omit.
  Sources: developer.nvidia.com/blog (CUDA 13.1; "CUDA Tile IR Backend for OpenAI Triton");
  jinseok-moon.github.io/p/cuda-tile; techpowerup/phoronix CUDA 13.1 announcements.

## Gluon / TLX / warp specialization
- Gluon = Triton's low-level model: exposes layouts (Linear Layouts, arXiv:2505.23819), shared memory,
  warp specialization, target features — convenience traded for control. Gives per-thread *layout*, not
  the per-lane *scalar control* a data-dependent CSR/while loop needs.
- TLX (Meta, 2025) = warp-aware intrinsics + warp-specialization annotations. Warp specialization
  (also Tawa, CGO'26) specializes WARPS for different tasks; it does NOT provide per-lane scalar control
  within a tile. (PyTorch blog "Warp Specialization in Triton"; Triton Gluon docs.)
  → Implication: none of Gluon/TLX/warp-spec close the per-lane data-dependent-control gap; they are
    orthogonal (layout / warp-task specialization), which the RFC's Alternatives section states.

## Existing Triton issues touching data-dependent control flow (no per-lane-region proposal exists)
- #2672 (trip-count-dependent scf.for body elimination), #9122 (invalid SSA from constant-range loop
  values used after exit), #9175 (scalar loads in WMMA layouts), #7125 (scalar vs tensor atomics).
  These are bug/edge-case reports — they confirm data-dependent control flow is a known pain area, but
  NONE proposes a per-lane sub-tile region. → the RFC is novel; cite these as evidence of the pain.
  Sources: github.com/triton-lang/triton issues #2672, #9122, #9175, #7125.
