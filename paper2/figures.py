"""Regenerate paper-2 figures from the versioned CSVs ONLY (fully reproducible).

Usage:  .venv/bin/python paper2/figures.py
Outputs vector PDFs under paper2/figures/. No seaborn; matplotlib Agg backend.
Every figure reads paper2/data/*.csv so the paper's plots cannot drift from the measurements.

Scope: this generates the figure set for the live ASPLOS submission
(gpufsm_asplos.tex). The superseded variants in this directory (cgo, ppopp, taco,
gpufsm2) still include the older fig_*.png set, which is kept for them and is no longer
regenerated here.

Sizing contract: figures are emitted at their FINAL printed size (COL for one acmart
sigplan column, WIDE for a full-width figure*) and included at scale 1.0, so no label is
ever downscaled below the 8 pt floor the ASPLOS format rules impose. Do not include these
with a width= that differs from the figsize below.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# acmart[sigplan] on US letter: \textwidth 7.0in, \columnsep 0.25in.
COL = 3.33
WIDE = 7.0

# Every size at or above 8 pt: nothing here is rescaled at \includegraphics time.
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _read(name: str) -> list[dict]:
    """Read a versioned CSV, ignoring '#' provenance lines."""
    with (DATA / name).open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 (teaser)


def _ladder() -> tuple[list[str], list[float]]:
    m2e = [r for r in _read("m2e_worklist_packed_rtx4070.csv") if r["n_strings"] == "16384"]
    m3 = [r for r in _read("m3_lite_rtx4070.csv") if r["n_strings"] == "16384"]
    return (
        ["Triton\n(default)", "$+$ nw=1", "$+$ lane-pack", "CUDA"],
        [
            statistics.median(float(r["wt_gbps"]) for r in m2e),
            statistics.median(float(r["ws_gbps"]) for r in m2e),
            statistics.median(float(r["wp2_gbps"]) for r in m3),
            statistics.median(float(r["cu_gbps"]) for r in m2e),
        ],
    )


def _straggler() -> tuple[dict, dict]:
    """(dist -> E[warp-max], D) and (dist -> masked_us, cured_us, speedup, heldout?)."""
    shape = {}
    for src in ("landmark/cure_predictive_rtx4070.csv", "landmark/cure_heldout_rtx4070.csv"):
        for r in _read(src):
            shape[r["dist"]] = (float(r["warpmax_mean"]), float(r["D"]))
    perf = {}
    for r in _read("cross_arch/cure_wheel_rtx4070.csv"):
        if r["experiment"] in ("predictive_sweep", "heldout"):
            perf[r["dist_or_workload"]] = (
                float(r["masked_us"]),
                float(r["cured_us"]),
                float(r["speedup"]),
                r["experiment"] == "heldout",
            )
    return shape, perf


# Straggler law as refit on the pinned wheel recipe (see cure_wheel_rtx4070.csv footer).
FIT_A, FIT_B = 32.4, 1.091


def fig_anatomy_and_cure() -> None:
    """Fig. 1: the tax (staged decomposition) and its removal (flat cured floor)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL, 3.9))

    labels, vals = _ladder()
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]
    bars = ax1.bar(labels, vals, color=colors, width=0.62)
    top = max(vals)
    for b, v, g in zip(
        bars, vals, [None] + [vals[i] / vals[i - 1] for i in (1, 2, 3)], strict=True
    ):
        cx = b.get_x() + b.get_width() / 2
        ax1.text(cx, v + top * 0.02, f"{v:.0f}", ha="center", va="bottom")
        if g is not None:
            ax1.text(
                cx,
                v + top * 0.115,
                f"${g:.1f}\\times$",
                ha="center",
                va="bottom",
                color="#555",
                fontsize=8,
            )
    ax1.set_ylabel("Gbps")
    ax1.set_ylim(0, top * 1.30)
    ax1.tick_params(axis="x", length=0)
    ax1.annotate(
        "irreducible\nresidual",
        xy=(2.60, vals[2] + 90),
        xytext=(1.05, top * 0.99),
        fontsize=8,
        color="#333",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
    )
    ax1.text(0.012, 0.96, "(a)", transform=ax1.transAxes, fontsize=8, va="top")

    shape, perf = _straggler()
    xs_in = [shape[d][0] for d in perf if not perf[d][3]]
    ys_in = [perf[d][0] for d in perf if not perf[d][3]]
    xs_out = [shape[d][0] for d in perf if perf[d][3]]
    ys_out = [perf[d][0] for d in perf if perf[d][3]]
    cured = [perf[d][1] for d in perf]
    grid = [0, 270]
    ax2.plot(
        grid,
        [FIT_A + FIT_B * g for g in grid],
        "-",
        color="#c0392b",
        lw=1.0,
        zorder=1,
        label=f"${FIT_A}+{FIT_B}\\,x$  ($R^2{{=}}0.997$)",
    )
    ax2.plot(xs_in, ys_in, "o", color="#c0392b", zorder=3, label="lock-step tile")
    ax2.plot(xs_out, ys_out, "o", mfc="white", mec="#c0392b", mew=1.0, zorder=3, label="held out")
    ax2.plot(
        [shape[d][0] for d in perf],
        cured,
        "^",
        color="#27ae60",
        zorder=3,
        label="per-lane retirement",
    )
    ax2.set_xlabel(r"per-warp straggler $\mathbb{E}[\max\ \mathrm{trip}]$")
    ax2.set_ylabel(r"time ($\mu$s)")
    ax2.set_xlim(0, 285)
    ax2.set_ylim(0, 345)
    ax2.legend(loc="upper left", fontsize=7.6, labelspacing=0.25)
    ax2.text(0.985, 0.06, "(b)", transform=ax2.transAxes, fontsize=8, va="bottom", ha="right")

    fig.tight_layout(h_pad=0.9)
    _save(fig, "fig_anatomy_and_cure")


# ---------------------------------------------------------------- Fig. 2 (mechanism)


def fig_mechanism() -> None:
    """Fig. 2: the residual is a dependent-load stall, and the cure removes exactly that."""
    ns = {r["kernel"]: r for r in _read("m10_nsight_rtx4070.csv")}
    order = ["wp2_tile", "cuda_worklist", "sp_threads_cure"]
    names = ["Triton\ntile", "hand\nCUDA", "cured"]
    colors = ["#2980b9", "#27ae60", "#8e44ad"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(WIDE, 1.95))

    issue = [float(ns[k]["issue_active_pct"]) for k in order]
    for b, v in zip(ax1.bar(names, issue, color=colors, width=0.6), issue, strict=True):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8)
    ax1.set_ylabel("issue-slot activity (%)")
    ax1.set_ylim(0, max(issue) * 1.28)
    ax1.set_title("(a) the tile cannot issue")
    ax1.tick_params(axis="x", length=0)

    stall = [float(ns[k]["long_scoreboard_stall"]) / 1e3 for k in order]
    for b, v in zip(ax2.bar(names, stall, color=colors, width=0.6), stall, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:.0f}k", ha="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel("dependent-load stall (kcyc)")
    ax2.set_ylim(3, stall[0] * 6)
    ax2.set_title("(b) because it waits")
    ax2.tick_params(axis="x", length=0)

    rf = {r["kernel"]: r for r in _read("m5b_roofline_rtx4070.csv")}
    tile, cu = rf["wp2_perlane_triton"], rf["cuda_worklist"]
    x, w = [0, 1], 0.36
    ax3.bar(
        [i - w / 2 for i in x],
        [float(tile["pct_peak_issue"]), float(tile["pct_peak_dram"])],
        w,
        color="#2980b9",
        label="Triton tile",
    )
    ax3.bar(
        [i + w / 2 for i in x],
        [float(cu["pct_peak_issue"]), float(cu["pct_peak_dram"])],
        w,
        color="#27ae60",
        label="hand CUDA",
    )
    ax3.axhline(100, color="#c0392b", ls="--", lw=0.9)
    ax3.text(-0.44, 104, "hardware ceiling", fontsize=8, color="#c0392b", ha="left")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["issue rate", "DRAM bandwidth"])
    ax3.set_ylabel("% of peak")
    ax3.set_ylim(0, 152)
    ax3.set_title("(c) and neither is at a ceiling")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.tick_params(axis="x", length=0)

    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig_mechanism")


# ---------------------------------------------------------------- Fig. 3 (DFA regime)


def fig_dfa_crossover() -> None:
    """Fig. 3: the residual is regime-dependent; it closes once the table leaves L2."""
    rows = _read("m4_dfa_rtx4070.csv")
    tkb = [float(r["table_kb"]) for r in rows]
    ratio = [float(r["triton_packed_gbps"]) / float(r["cuda_gbps"]) for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ax.plot(tkb, ratio, "o-", color="#2980b9")
    ax.axhline(1.0, color="#27ae60", ls="--", lw=1.0)
    ax.text(
        tkb[0] * 1.15,
        1.035,
        "parity with CUDA",
        fontsize=8,
        color="#27ae60",
        va="bottom",
        ha="left",
    )
    ax.axvline(6144, color="gray", ls=":", lw=1.0)
    ax.text(6144 * 0.85, 0.88, "L2 ($\\sim$6 MB)", fontsize=8, color="gray", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DFA table size (KB)")
    ax.set_ylabel("lane-packed\nTriton / CUDA")
    ax.set_ylim(0.45, 1.22)
    fig.tight_layout()
    _save(fig, "fig_dfa_crossover")


# ---------------------------------------------------------------- Fig. 4 (the predictor)


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x, returning (a, b, R^2)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys, strict=True)) / sum(
        (a - mx) ** 2 for a in xs
    )
    a0 = my - b * mx
    pred = [a0 + b * v for v in xs]
    ss_res = sum((u - v) ** 2 for u, v in zip(ys, pred, strict=True))
    ss_tot = sum((u - my) ** 2 for u in ys)
    return a0, b, 1 - ss_res / ss_tot


def fig_straggler_law() -> None:
    """Fig. 4: the straggler predicts the lock-step cost; the divergence ratio does not."""
    shape, perf = _straggler()
    dists = list(perf)
    warpmax = [shape[d][0] for d in dists]
    dratio = [shape[d][1] for d in dists]
    masked = [perf[d][0] for d in dists]
    held = [perf[d][3] for d in dists]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    for ax, xv, lab in (
        (ax1, warpmax, r"straggler $\mathbb{E}[\max\ \mathrm{trip}]$"),
        (ax2, dratio, r"divergence ratio $D$"),
    ):
        a0, b, r2 = _ols(xv, masked)
        grid = [min(xv) * 0.0, max(xv) * 1.08]
        ax.plot(grid, [a0 + b * g for g in grid], "-", color="#c0392b", lw=1.0, zorder=1)
        for x, y, h in zip(xv, masked, held, strict=True):
            ax.plot(
                [x], [y], "o", color="#c0392b", mfc=("white" if h else "#c0392b"), mew=1.0, zorder=3
            )
        ax.set_xlabel(lab, fontsize=8)
        ax.set_xlim(0, max(xv) * 1.08)
        ax.text(
            0.5,
            0.96,
            f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    ax1.set_ylabel(r"lock-step time ($\mu$s)")
    ax1.set_ylim(0, 380)
    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig_straggler_law")


# ---------------------------------------------------------------- Fig. 5 (generality)


def fig_regret_law() -> None:
    """Fig. 5: regret across eight oracle-gated witnesses, coloured by dominant mechanism."""
    rows = {r["workload"]: r for r in _read("landmark/regret_law.csv")}
    order = [
        "pointer_chase",
        "spmv_uniform",
        "hashprobe",
        "automata_nfa",
        "spmv_powerlaw",
        "rejection",
        "moe_powerlaw",
        "attention_powerlaw",
    ]
    nice = {
        "pointer_chase": "graph\npointer-chase",
        "spmv_uniform": "SpMV\nuniform",
        "hashprobe": "hash\nprobe",
        "automata_nfa": "automata\n(NFA)",
        "spmv_powerlaw": "SpMV\npower-law",
        "rejection": "rejection\nsampling",
        "moe_powerlaw": "MoE routing\n(ML, scalar)",
        "attention_powerlaw": "attention\n(ML, dense)",
    }
    mech = {
        "negative_control_latency_equal": ("#27ae60", "no control divergence (control)"),
        "baseline_occupancy_50v94": ("#7f8c8d", "tile-lowering baseline"),
        "masked_waste_gather_diluted": ("#2980b9", "masked-lane waste (gather-diluted)"),
        "latency_starvation": ("#c0392b", "issue starvation"),
        "baseline_plus_divergence": ("#8e44ad", "baseline $+$ divergence"),
        "masked_waste_pure_compute": ("#e67e22", "masked-lane waste"),
        "scalar_control_ml_moe": ("#c0392b", "issue starvation"),
        "dense_vector_tile_wins": ("#16a085", "dense per-step work: tile wins"),
    }
    vals = [float(rows[w]["regret"]) for w in order]
    colors = [mech[rows[w]["dominant_mechanism"]][0] for w in order]

    fig, ax = plt.subplots(figsize=(WIDE, 2.05))
    for b, v in zip(
        ax.bar([nice[w] for w in order], vals, color=colors, width=0.62), vals, strict=True
    ):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}$\\times$", ha="center", fontsize=8
        )
    ax.axhline(1.0, color="#27ae60", ls="--", lw=1.0)
    ax.annotate(
        "the tile wins",
        xy=(7, vals[-1] + 0.14),
        xytext=(7, 1.62),
        ha="center",
        fontsize=8,
        color="#16a085",
        arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.9),
    )
    ax.set_ylabel(r"tile-vs-thread regret ($\times$)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="x", length=0)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    seen = set()
    handles = [Line2D([], [], color="#27ae60", ls="--", lw=1.0, label=r"no regret ($1\times$)")]
    for w in order:
        c, lab = mech[rows[w]["dominant_mechanism"]]
        if lab not in seen:
            seen.add(lab)
            handles.append(Patch(facecolor=c, label=lab))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig_regret_law")


def main() -> int:
    fig_anatomy_and_cure()
    fig_mechanism()
    fig_dfa_crossover()
    fig_straggler_law()
    fig_regret_law()
    print(f"wrote figures to {OUT}/:")
    for p in sorted(OUT.glob("*.pdf")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
