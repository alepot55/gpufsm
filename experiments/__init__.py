"""Measurement experiments — the provenance of the numbers in paper/ and paper2/.

Not part of the installed wheel: this is a package so the modules can import each
other and be run as ``python -m experiments.cure.<group>.<name>`` from the repo root,
instead of depending on the process's working directory.

Everything here reports a number, and every number is gated on agreeing with the CPU
oracle first (:mod:`gpufsm.bench.oracle`). Shared measurement machinery lives in
:mod:`gpufsm.bench`, not in these files.
"""

from __future__ import annotations
