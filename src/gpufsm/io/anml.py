"""ANML (Automata Network Markup Language) loading.

ANMLZoo/AutomataZoo automata are distributed as ANML, Micron's XML format for
*homogeneous* automata (the matching symbol-set lives on the state, not the
edge). Converting that to gpufsm's edge-labelled CSR :class:`~gpufsm.nfa.NFA`
requires parsing symbol-sets (character classes) and the
``activate-on-match`` graph.

A pure-Python parser is planned (tracked in the project plan, §11). Until then
this raises a clear error; tests and examples use :class:`~gpufsm.nfa.NFABuilder`
directly. The legacy C++ parser (``extern/anml``) remains available behind the
optional ``[bitgen]`` extra for large-suite reproduction.
"""

from __future__ import annotations

from pathlib import Path

from ..nfa import NFA


def load_anml(path: str | Path) -> NFA:  # pragma: no cover - not yet implemented
    raise NotImplementedError(
        "Pure-Python ANML loading is not implemented yet (see project plan §11). "
        "Build NFAs via gpufsm.NFABuilder, or use the optional C++ parser for the "
        "large ANMLZoo/AutomataZoo suite."
    )
