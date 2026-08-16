"""Generality witnesses: does the regret exist outside automata?

Each ``landmark_*`` module measures one irregular workload under the tile (Triton) and
thread (CUDA) paradigms; the matching ``cure_*`` module applies the built cure to it.
SpMV is the deliberate negative control: irregular memory, little control divergence.
"""

from __future__ import annotations
