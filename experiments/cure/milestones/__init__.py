"""Milestones M0..M10: the automata workload.

The spine of the argument. M0 pins the Triton-vs-CUDA regret anchor that every later
milestone has to move; M2/M3 probe whether lane packing or occupancy closes it; M4
carries it to the memory-bound DFA; M9 goes past 64 states; M10 implements the cure as
a ``scalar_program`` primitive.
"""

from __future__ import annotations
