"""CPU NFA simulator — the single source of truth for correctness.

Every GPU backend (Triton, CUDA) must reproduce :func:`simulate` exactly
(``accepted`` and ``match_len``) on every benchmark. Semantics are
*latch-first-match*: report as soon as any accepting state is active, returning
the length of the matched prefix (0 for a zero-length / start-state match).

Clarity over speed: this is the oracle, not a performance path.
"""

from __future__ import annotations

import numpy as np

from .nfa import ANY_SYMBOL, NFA


def epsilon_closure(states: set[int], nfa: NFA) -> set[int]:
    """Return the epsilon-closure of ``states`` (iterative DFS, no recursion)."""
    seen = set(states)
    stack = list(states)
    ep = nfa.eps_row_ptr
    et = nfa.eps_targets
    while stack:
        s = stack.pop()
        for k in range(int(ep[s]), int(ep[s + 1])):
            t = int(et[k])
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def simulate(nfa: NFA, input_bytes: bytes) -> tuple[bool, int]:
    """Simulate ``nfa`` over ``input_bytes`` with latch-first-match semantics.

    Returns ``(accepted, match_len)``.
    """
    sp = nfa.sym_row_ptr
    st = nfa.sym_targets
    ss = nfa.sym_symbols
    accept = nfa.accept

    active = epsilon_closure({nfa.start_state}, nfa)
    if any(accept[s] for s in active):
        return True, 0

    data = np.frombuffer(input_bytes, dtype=np.uint8)
    for i in range(data.size):
        b = int(data[i])
        nxt: set[int] = set()
        for s in active:
            for k in range(int(sp[s]), int(sp[s + 1])):
                sym = int(ss[k])
                if sym == b or sym == ANY_SYMBOL:
                    nxt.add(int(st[k]))
        if not nxt:
            break
        active = epsilon_closure(nxt, nfa)
        if any(accept[s] for s in active):
            return True, i + 1

    return False, 0
