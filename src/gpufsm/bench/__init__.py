"""Measurement harness shared by the CLI, the scripts and the experiments.

Everything a measurement needs that is not the kernel itself lives here, so that the
scripts under ``scripts/`` and ``experiments/`` are thin drivers rather than each
carrying their own copy:

- :mod:`~gpufsm.bench.generators` — the canonical random automata and input batches
- :mod:`~gpufsm.bench.timing` — median + bootstrap CI95, throughput, warmup loops
- :mod:`~gpufsm.bench.oracle` — correctness gates that must pass before any number is reported
- :mod:`~gpufsm.bench.nvcc` — compile a CUDA source to a ``.so`` and load it via ctypes
- :mod:`~gpufsm.bench.csvio` — versioned CSV output with an explicit schema
- :mod:`~gpufsm.bench.sweep` — benchmark every available (backend, technique)
"""

from __future__ import annotations

from .csvio import environment, gpu_slug, print_environment, write_rows
from .generators import (
    DENSE,
    NO_ACCEPT,
    SPARSE_WORKLIST,
    WITH_WILDCARDS,
    NFAShape,
    random_batch,
    random_batch_2d,
    random_byte_batch,
    random_bytes_2d,
    random_nfa,
)
from .oracle import OracleMismatch, matches, require
from .sweep import CSV_FIELDS, sweep, write_csv
from .timing import bootstrap_ci95, gbps, repeat, summarize

__all__ = [
    "NFAShape",
    "DENSE",
    "NO_ACCEPT",
    "SPARSE_WORKLIST",
    "WITH_WILDCARDS",
    "random_nfa",
    "random_batch",
    "random_batch_2d",
    "random_byte_batch",
    "random_bytes_2d",
    "sweep",
    "write_csv",
    "CSV_FIELDS",
    "write_rows",
    "environment",
    "gpu_slug",
    "print_environment",
    "bootstrap_ci95",
    "gbps",
    "repeat",
    "summarize",
    "OracleMismatch",
    "matches",
    "require",
]
