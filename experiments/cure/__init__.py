"""The "cure" line of work: diagnose the abstraction regret, then close it.

Four groups, in the order the argument is built:

- :mod:`~experiments.cure.milestones` — the automata workload, from the regret anchor
  to the per-lane ``scalar_program`` primitive that closes it (M0..M10)
- :mod:`~experiments.cure.landmarks` — generality witnesses beyond automata (SpMV, MoE,
  BFS, attention, hash probe, rejection sampling) and the cure applied to each
- :mod:`~experiments.cure.passes` — the in-compiler work: the lowering wall, the
  ThreadRegion detection pass, the selector, and the F3 reduce-hoist
- :mod:`~experiments.cure.validation` — dispersion, out-of-sample and cross-architecture
  checks on the numbers the paper quotes
"""

from __future__ import annotations
