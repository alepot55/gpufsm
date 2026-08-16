"""In-compiler work: where the abstraction wall actually is.

The ``p2_*`` modules probe the lowering wall and the ThreadRegion detection pass inside
libtriton; ``f3_*`` measures and verifies the reduce-hoist. ``bench_perlane_retire`` is
the standalone benchmark plus oracle check for the per-lane loop retirement pass.
"""

from __future__ import annotations
