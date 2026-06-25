"""ANML loader tests — hand-built fixture + NFA->ANML->NFA round-trip (no downloads)."""

from __future__ import annotations

import random

import pytest

from gpufsm.io.anml import load_anml, parse_symbol_set, to_anml
from gpufsm.nfa import ANY_SYMBOL
from gpufsm.reference import simulate

# Homogeneous ANML for the pattern "ab": s0 (start, matches 'a') -> s1 (matches 'b', report).
_ANML_AB = """<?xml version="1.0"?>
<automata-network id="t">
  <state-transition-element id="s0" symbol-set="[0x61]" start="start-of-data">
    <activate-on-match element="s1"/>
  </state-transition-element>
  <state-transition-element id="s1" symbol-set="[0x62]">
    <report-on-match reportcode="1"/>
  </state-transition-element>
</automata-network>
"""


def test_parse_symbol_set_forms():
    assert parse_symbol_set("*") == {ANY_SYMBOL}
    assert parse_symbol_set("[0x61-0x63]") == {97, 98, 99}
    assert parse_symbol_set("[abc]") == {97, 98, 99}
    assert parse_symbol_set("[0x61]") == {97}
    # negation over 0..255
    neg = parse_symbol_set("[^0x61]")
    assert 97 not in neg and len(neg) == 255 and 98 in neg


def test_load_anml_fixture_semantics(tmp_path):
    p = tmp_path / "ab.anml"
    p.write_text(_ANML_AB)
    nfa = load_anml(p)
    assert simulate(nfa, b"ab") == (True, 2)
    assert simulate(nfa, b"a") == (False, 0)
    assert simulate(nfa, b"ax") == (False, 0)
    assert simulate(nfa, b"b") == (False, 0)
    assert simulate(nfa, b"") == (False, 0)


def test_anml_round_trip_preserves_verdicts(tmp_path):
    # load -> export -> reload must give identical verdicts (homogeneous-preserving).
    src = tmp_path / "ab.anml"
    src.write_text(_ANML_AB)
    nfa1 = load_anml(src)
    out = to_anml(nfa1, tmp_path / "rt.anml")
    nfa2 = load_anml(out)
    rng = random.Random(0)
    alphabet = b"abx"
    for _ in range(200):
        data = bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
        assert simulate(nfa1, data) == simulate(nfa2, data), data


def test_load_anml_empty_raises(tmp_path):
    p = tmp_path / "empty.anml"
    p.write_text('<?xml version="1.0"?><automata-network id="t"></automata-network>')
    with pytest.raises(ValueError):
        load_anml(p)
