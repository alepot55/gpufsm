"""GPU backend correctness — skipped unless a GPU backend is actually available.

On a GPU box these assert that Triton/CUDA reproduce the CPU reference oracle
(accepted + match_len) on every example and a fuzz of random NFAs.
"""

from __future__ import annotations

import random

import pytest

from gpufsm import ANY_SYMBOL, NFABuilder, available_backends, run, simulate
from gpufsm.examples import EXAMPLES
from gpufsm.registry import Backend as _B
from gpufsm.registry import list_techniques

_GPU_BACKENDS = [b for b in available_backends() if b in (_B.TRITON, _B.CUDA)]
pytestmark = pytest.mark.gpu

skip_no_gpu = pytest.mark.skipif(not _GPU_BACKENDS, reason="no GPU backend available")


def _cases():
    for backend in _GPU_BACKENDS:
        for technique in list_techniques(backend):
            yield backend, technique


@skip_no_gpu
def test_gpu_matches_reference_on_examples():
    for backend, technique in _cases():
        for name in EXAMPLES:
            nfa, inputs = EXAMPLES[name]()
            for data, _ in inputs:
                ref = simulate(nfa, data)
                res = run(nfa, data, backend=backend, technique=technique)
                assert (res.accepted, res.match_len) == ref, (
                    f"{backend.value}/{technique} {name} {data!r}: "
                    f"got ({res.accepted},{res.match_len}) want {ref}"
                )


@skip_no_gpu
def test_gpu_matches_reference_fuzz():
    rng = random.Random(7)
    alphabet = "abc"
    for backend, technique in _cases():
        for _ in range(40):
            b = NFABuilder()
            n = rng.randint(1, 8)
            for _ in range(n):
                b.add_state(accept=rng.random() < 0.25)
            b.set_start(rng.randrange(n))
            for s in range(n):
                for _ in range(rng.randint(0, 2)):
                    sym = ANY_SYMBOL if rng.random() < 0.1 else rng.choice(alphabet)
                    b.add_transition(s, sym, rng.randrange(n))
                for _ in range(rng.randint(0, 1)):
                    b.add_epsilon(s, rng.randrange(n))
            nfa = b.build()
            data = bytes(ord(rng.choice(alphabet)) for _ in range(rng.randint(0, 10)))
            ref = simulate(nfa, data)
            res = run(nfa, data, backend=backend, technique=technique)
            assert (res.accepted, res.match_len) == ref
