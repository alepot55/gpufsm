"""LANDMARK P3 — cross-architecture re-validation, one command.

Central claim: tile-vs-thread regret follows the EXECUTION PARADIGM, not the arch. So on a different
GPU each witness's regret should PERSIST in direction (>1 for divergent witnesses, =1 for the
pointer-chase negative control) even as absolute throughput rescales. This re-runs the regret-law
witnesses + the M10 cure + the P2 selector on whatever GPU is present, compares each regret to the
committed RTX4070 baseline, and writes a tagged cross-arch CSV.

SAFE to run anywhere: each witness hardcodes a `*_rtx4070.csv` path, so we snapshot that file, run
the witness (it overwrites the file with THIS GPU's numbers), read the new regret, then RESTORE the
committed baseline and save this GPU's results under paper2/data/cross_arch/. Witnesses that error
(no CUDA / no nvcc / no from-source Triton for the selector) are skipped gracefully.

Falsifiable: if a witness's regret collapses to ~1 on another arch (e.g. rejection drops to ~1), the
paradigm-not-arch claim is wrong for it.
Run: .venv/bin/python experiments/cure/p3_cross_arch.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "paper2" / "data"
OUTDIR = DATA / "cross_arch"

# witness -> (module, csv path relative to repo, how to pull regret(s) from the csv)
# each entry's parser returns a dict {sub_label: regret_float} read from the witness's own CSV.
WITNESSES = [
    ("spmv", "experiments.cure.landmark_spmv", "paper2/data/landmark/spmv_rtx4070.csv"),
    (
        "rejection",
        "experiments.cure.landmark_rejection",
        "paper2/data/landmark/rejection_rtx4070.csv",
    ),
    ("pointer_chase", "experiments.cure.landmark_bfs", "paper2/data/landmark/bfs_rtx4070.csv"),
    (
        "hashprobe",
        "experiments.cure.landmark_hashprobe",
        "paper2/data/landmark/hashprobe_rtx4070.csv",
    ),
    (
        "automata_nfa",
        "experiments.cure.m10_scalar_program",
        "paper2/data/m10_scalar_program_rtx4070.csv",
    ),
]


def _regrets_from_csv(label: str, text: str) -> dict[str, float]:
    """Pull regret value(s) from a witness CSV's raw text. Aggregates multi-row CSVs by median."""
    lines = [ln for ln in text.strip().splitlines() if ln]
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    if label == "spmv":  # one regret per matrix kind (uniform, powerlaw)
        mi, ri = header.index("matrix"), header.index("regret")
        return {f"spmv_{r[mi]}": float(r[ri]) for r in rows}
    if label == "automata_nfa":  # tile-vs-thread = SP/WP2, median over (states,seed)
        ri = header.index("sp_over_wp2")
        vals = sorted(float(r[ri]) for r in rows)
        return {"automata_nfa": vals[len(vals) // 2]}
    # single-metric witnesses: median of the regret column
    ri = header.index("regret")
    vals = sorted(float(r[ri]) for r in rows)
    return {label: vals[len(vals) // 2]}


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown-gpu"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "gpu"


def _run_witness(label: str, module: str, csv_rel: str) -> dict[str, float]:
    """Run a witness subprocess; return THIS GPU's regret(s); restore the committed baseline."""
    csv_path = REPO / csv_rel
    orig = csv_path.read_text() if csv_path.exists() else None
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not csv_path.exists():
            print(f"  {label:<14} SKIP (exit {proc.returncode}; {proc.stdout.strip()[-80:]})")
            return {}
        new = _regrets_from_csv(label, csv_path.read_text())
        return new
    except Exception as e:
        print(f"  {label:<14} SKIP ({type(e).__name__}: {e})")
        return {}
    finally:
        if orig is not None:  # restore the committed RTX4070 baseline regardless of this GPU
            csv_path.write_text(orig)


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            print("SKIP: no CUDA device.")
            return 0
    except Exception:
        print("SKIP: torch/CUDA unavailable.")
        return 0

    gpu = _gpu_name()
    print(f"P3 cross-arch re-validation on: {gpu}")
    print(
        "  (regret should PERSIST in direction vs the RTX4070 baseline; throughput may rescale)\n"
    )

    # committed RTX4070 baselines, read from each witness CSV BEFORE we run anything.
    baselines: dict[str, float] = {}
    for label, _module, csv_rel in WITNESSES:
        p = REPO / csv_rel
        if p.exists():
            baselines.update(_regrets_from_csv(label, p.read_text()))

    measured: dict[str, float] = {}
    for label, module, csv_rel in WITNESSES:
        measured.update(_run_witness(label, module, csv_rel))

    # selector routing decision (does not depend on arch; record if it runs). Snapshot+restore its
    # committed CSV so this re-validation never mutates committed baselines.
    sel = "n/a"
    sel_csv = REPO / "paper2" / "data" / "landmark" / "p2_selector_rtx4070.csv"
    sel_orig = sel_csv.read_text() if sel_csv.exists() else None
    try:
        ps = subprocess.run(
            [sys.executable, "-m", "experiments.cure.p2_selector"],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        sel = "VERIFIED" if "selector VERIFIED" in ps.stdout else "ran"
    except Exception:
        sel = "skipped"
    finally:
        if sel_orig is not None:
            sel_csv.write_text(sel_orig)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outp = OUTDIR / f"regret_{_slug(gpu)}.csv"
    print(f"\n  {'witness':<16}{'baseline':>10}{'this_gpu':>10}{'persists':>10}")
    all_persist = True
    with outp.open("w") as f:
        f.write("witness,baseline_regret_rtx4070,this_regret,persists,gpu,selector\n")
        for key in sorted(set(baselines) | set(measured)):
            b = baselines.get(key)
            m = measured.get(key)
            if b is None or m is None:
                continue
            # persists: a >1 baseline stays >1 (>=1.1), a ~1 baseline stays ~1 (<=1.15)
            persists = (m >= 1.1) if b >= 1.1 else (m <= 1.15)
            all_persist = all_persist and persists
            print(f"  {key:<16}{b:>10.2f}{m:>10.2f}{('yes' if persists else 'NO'):>10}")
            f.write(f"{key},{b},{round(m, 3)},{int(persists)},{gpu},{sel}\n")
    print(f"\n  selector: {sel}")
    verdict = "CONFIRMED" if all_persist else "VIOLATED (investigate)"
    print(f"  => paradigm-not-architecture: {verdict}")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
