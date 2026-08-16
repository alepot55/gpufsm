"""Regenerate ``tests/data/golden.json`` — the refactor safety net.

The golden file pins the *verdicts* of the correctness oracles
(:func:`gpufsm.reference.simulate` for NFAs, :func:`gpufsm.dfa.simulate_dfa` for DFAs)
on a fixed corpus. :mod:`tests.test_golden` replays them; any refactor that changes a
single ``(accepted, match_len)`` fails loudly.

The automata are **serialized in full** (CSR arrays / transition tables), not as
generator seeds. That is deliberate: the corpus must stay valid even when the random
generators are moved, unified or reparameterized, which is exactly what the refactor
does. The generator below is self-contained for the same reason.

Run only to establish a new baseline, and only when the change in verdicts is
understood and intended::

    python -m tests.generate_golden
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from gpufsm.dfa import DFA, DFABuilder, simulate_dfa
from gpufsm.examples import EXAMPLES
from gpufsm.nfa import ANY_SYMBOL, NFA, NFABuilder
from gpufsm.reference import simulate

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.json"
FORMAT_VERSION = 1

_ALPHABET = "abcd"
_FUZZ_CASES = 300
_FUZZ_SEED = 20260816


def nfa_to_json(nfa: NFA) -> dict[str, Any]:
    return {
        "num_states": int(nfa.num_states),
        "start_state": int(nfa.start_state),
        "accept": [bool(x) for x in nfa.accept],
        "sym_row_ptr": [int(x) for x in nfa.sym_row_ptr],
        "sym_targets": [int(x) for x in nfa.sym_targets],
        "sym_symbols": [int(x) for x in nfa.sym_symbols],
        "eps_row_ptr": [int(x) for x in nfa.eps_row_ptr],
        "eps_targets": [int(x) for x in nfa.eps_targets],
    }


def nfa_from_json(d: dict[str, Any]) -> NFA:
    return NFA(
        num_states=int(d["num_states"]),
        start_state=int(d["start_state"]),
        accept=np.array(d["accept"], dtype=bool),
        sym_row_ptr=np.array(d["sym_row_ptr"], dtype=np.int32),
        sym_targets=np.array(d["sym_targets"], dtype=np.int32),
        sym_symbols=np.array(d["sym_symbols"], dtype=np.int32),
        eps_row_ptr=np.array(d["eps_row_ptr"], dtype=np.int32),
        eps_targets=np.array(d["eps_targets"], dtype=np.int32),
    )


def dfa_to_json(dfa: DFA) -> dict[str, Any]:
    return {
        "num_states": int(dfa.num_states),
        "start_state": int(dfa.start_state),
        "accept": [bool(x) for x in dfa.accept],
        "trans": [int(x) for x in dfa.trans],
    }


def dfa_from_json(d: dict[str, Any]) -> DFA:
    return DFA(
        num_states=int(d["num_states"]),
        start_state=int(d["start_state"]),
        accept=np.array(d["accept"], dtype=bool),
        trans=np.array(d["trans"], dtype=np.int32),
    )


def _fuzz_nfa(rng: random.Random, n_states: int) -> NFA:
    """Self-contained fuzz generator — intentionally NOT imported from the harness."""
    b = NFABuilder()
    for _ in range(n_states):
        b.add_state(accept=rng.random() < 0.2)
    b.set_start(rng.randrange(n_states))
    for s in range(n_states):
        for _ in range(rng.randint(0, 3)):
            sym = ANY_SYMBOL if rng.random() < 0.1 else rng.choice(_ALPHABET)
            b.add_transition(s, sym, rng.randrange(n_states))
        for _ in range(rng.randint(0, 2)):
            b.add_epsilon(s, rng.randrange(n_states))
    return b.build()


def _fuzz_dfa(rng: random.Random, n_states: int) -> DFA:
    b = DFABuilder()
    for i in range(n_states):
        b.add_state(accept=(i > 0 and rng.random() < 0.25))
    b.set_start(0)
    for s in range(n_states):
        for ch in _ALPHABET:
            if rng.random() < 0.8:
                b.add_transition(s, ord(ch), rng.randrange(n_states))
    return b.build()


def build_corpus() -> dict[str, Any]:
    nfa_cases: list[dict[str, Any]] = []

    # 1. The canonical hand-built examples, on their labelled inputs.
    for name, factory in sorted(EXAMPLES.items()):
        nfa, inputs = factory()
        payload = nfa_to_json(nfa)
        for i, (data, expected) in enumerate(inputs):
            accepted, match_len = simulate(nfa, data)
            assert accepted == expected, f"{name}[{i}] disagrees with its own label"
            nfa_cases.append(
                {
                    "id": f"example:{name}:{i}",
                    "nfa": payload,
                    "input_hex": data.hex(),
                    "accepted": accepted,
                    "match_len": match_len,
                }
            )

    # 2. Fuzz corpus: random NFAs incl. epsilon cycles and ANY_SYMBOL wildcards.
    rng = random.Random(_FUZZ_SEED)
    for i in range(_FUZZ_CASES):
        nfa = _fuzz_nfa(rng, rng.randint(1, 12))
        data = bytes(ord(rng.choice(_ALPHABET)) for _ in range(rng.randint(0, 16)))
        accepted, match_len = simulate(nfa, data)
        nfa_cases.append(
            {
                "id": f"fuzz:{i}",
                "nfa": nfa_to_json(nfa),
                "input_hex": data.hex(),
                "accepted": accepted,
                "match_len": match_len,
            }
        )

    # 3. DFA corpus (the memory-bound face's oracle).
    dfa_cases: list[dict[str, Any]] = []
    drng = random.Random(_FUZZ_SEED + 1)
    for i in range(12):
        dfa = _fuzz_dfa(drng, drng.randint(2, 6))
        payload = dfa_to_json(dfa)
        for j in range(3):
            data = bytes(ord(drng.choice(_ALPHABET)) for _ in range(drng.randint(0, 24)))
            accepted, match_len = simulate_dfa(dfa, data)
            dfa_cases.append(
                {
                    "id": f"dfa:{i}:{j}",
                    "dfa": payload,
                    "input_hex": data.hex(),
                    "accepted": accepted,
                    "match_len": match_len,
                }
            )

    return {
        "format_version": FORMAT_VERSION,
        "note": (
            "Oracle verdicts pinned before the 2026-08-16 refactor. Automata are "
            "serialized in full so the corpus survives changes to the generators."
        ),
        "nfa_cases": nfa_cases,
        "dfa_cases": dfa_cases,
    }


def main() -> int:
    corpus = build_corpus()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(corpus, separators=(",", ":")) + "\n")
    n_nfa = len(corpus["nfa_cases"])
    n_dfa = len(corpus["dfa_cases"])
    print(f"wrote {GOLDEN_PATH} ({n_nfa} NFA cases, {n_dfa} DFA cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
