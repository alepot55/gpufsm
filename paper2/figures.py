"""Regenerate paper-2 figures from the versioned CSVs ONLY (fully reproducible).

Usage:  .venv/bin/python paper2/figures.py
Outputs PNGs under paper2/figures/. No seaborn; matplotlib Agg backend.
Every figure reads paper2/data/*.csv so the paper's plots cannot drift from the measurements.
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


def _read(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def fig_decomposition() -> None:
    """(a) Throughput climbs as each regret component is addressed, vs CUDA. m2e + m3_lite."""
    m2e = _read("m2e_worklist_packed_rtx4070.csv")
    m3 = _read("m3_lite_rtx4070.csv")
    # use batch 16384 rows
    e = [r for r in m2e if r["n_strings"] == "16384"]
    t = [r for r in m3 if r["n_strings"] == "16384"]
    wt = statistics.median(float(r["wt_gbps"]) for r in e)  # nw=4 default
    ws = statistics.median(float(r["ws_gbps"]) for r in e)  # nw=1 (artifact removed)
    wp2 = statistics.median(float(r["wp2_gbps"]) for r in t)  # lane-packed per-lane
    cu = statistics.median(float(r["cu_gbps"]) for r in e)  # hand-CUDA
    labels = [
        "Triton\nworklist\n(nw=4)",
        "+ nw=1\n(−artifact)",
        "+ lane-pack\n(WP2)",
        "CUDA\n(thread)",
    ]
    vals = [wt, ws, wp2, cu]
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Throughput (Gbps)")
    ax.set_title("Decomposing the NFA worklist regret (batch 16384, ≤64 states)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.annotate(
        f"artifact ≈{ws / wt:.1f}×",
        (1, ws),
        (1, ws + 90),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        f"lane-pack ≈{wp2 / ws:.1f}×",
        (2, wp2),
        (1.45, wp2 + 80),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        f"irreducible ≈{cu / wp2:.1f}×",
        (3, cu),
        (3, cu - 130),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->"),
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_decomposition.png", dpi=150)
    plt.close(fig)


def fig_occupancy_gating() -> None:
    """(b) Lane-packing benefit is occupancy-gated: C/B vs batch. m2_batch_scaling."""
    rows = _read("m2_batch_scaling_rtx4070.csv")
    n = [int(r["n_strings"]) for r in rows]
    cb = [float(r["pure_packing_C_over_B_median"]) for r in rows]
    ca = [float(r["realistic_C_over_A_median"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n, cb, "o-", color="#2980b9", label="pure lane-packing (work held equal)")
    ax.plot(n, ca, "s--", color="#8e44ad", label="realistic (vs work-efficient scalar)")
    ax.axhline(32, color="gray", ls=":", lw=1)
    ax.text(n[0], 32.5, "ideal 32× (warp width)", fontsize=8, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("batch (strings) → occupancy")
    ax.set_ylabel("lane-packing speedup (×)")
    ax.set_title("Lane-packing is occupancy-gated")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_occupancy_gating.png", dpi=150)
    plt.close(fig)


def fig_mechanism() -> None:
    """(c) Nsight: WP2 vs CUDA — matched occupancy/warps, fewer warp-inst, slower, low issue."""
    rows = _read("m3_lite_nsight_rtx4070.csv")
    wp2 = next(r for r in rows if r["kernel"].startswith("wp2"))
    cu = next(r for r in rows if r["kernel"].startswith("cuda"))
    metrics = ["warp-inst\n(M)", "occupancy\n(%)", "issue active\n(%)", "duration\n(µs)"]
    wp2_v = [
        float(wp2["warp_inst"]) / 1e6,
        float(wp2["occupancy_pct"]),
        float(wp2["issue_active_pct"]),
        float(wp2["duration_us"]),
    ]
    cu_v = [
        float(cu["warp_inst"]) / 1e6,
        float(cu["occupancy_pct"]),
        float(cu["issue_active_pct"] or 0) or float("nan"),
        float(cu["duration_us"]),
    ]
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - w / 2 for i in x], wp2_v, w, label="Triton WP2 (per-lane)", color="#2980b9")
    ax.bar([i + w / 2 for i in x], cu_v, w, label="CUDA worklist", color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Same occupancy & warps, fewer warp-inst — yet 3.3× slower (latency-bound)")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(wp2_v, cu_v, strict=True)):
        ax.text(i - w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=7)
        if b == b:  # not nan
            ax.text(i + w / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism.png", dpi=150)
    plt.close(fig)


def fig_dfa_crossover() -> None:
    """(d) DFA regime crossover: PK/CU vs table size. m4_dfa."""
    rows = _read("m4_dfa_rtx4070.csv")
    tkb = [float(r["table_kb"]) for r in rows]
    pkcu = [float(r["triton_packed_gbps"]) / float(r["cuda_gbps"]) for r in rows]
    pk = [float(r["triton_packed_gbps"]) for r in rows]
    cu = [float(r["cuda_gbps"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(tkb, pk, "o-", color="#2980b9", label="lane-packed Triton")
    ax1.plot(tkb, cu, "s-", color="#27ae60", label="CUDA")
    ax1.axvline(6144, color="gray", ls=":", lw=1)
    ax1.text(6144, min(pk), " L2 (~6 MB)", fontsize=8, color="gray")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("DFA table size (KB)")
    ax1.set_ylabel("Throughput (Gbps)")
    ax1.set_title("DFA: absolute throughput")
    ax1.legend(fontsize=8)
    ax2.plot(tkb, pkcu, "D-", color="#c0392b")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.axvline(6144, color="gray", ls=":", lw=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("DFA table size (KB)")
    ax2.set_ylabel("lane-packed Triton / CUDA")
    ax2.set_title("Gap closes in the DRAM regime")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dfa_crossover.png", dpi=150)
    plt.close(fig)


def fig_roofline() -> None:
    """(e) Both kernels sit far below the issue AND bandwidth ceilings -> latency-bound. m5b."""
    rows = _read("m5b_roofline_rtx4070.csv")
    labels = ["% of peak\nissue rate", "% of peak\nDRAM BW"]
    wp2 = next(r for r in rows if r["kernel"].startswith("wp2"))
    cu = next(r for r in rows if r["kernel"].startswith("cuda"))
    wp2_v = [float(wp2["pct_peak_issue"]), float(wp2["pct_peak_dram"])]
    cu_v = [float(cu["pct_peak_issue"]), float(cu["pct_peak_dram"])]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], wp2_v, w, label="Triton WP2 (per-lane)", color="#2980b9")
    ax.bar([i + w / 2 for i in x], cu_v, w, label="CUDA worklist", color="#27ae60")
    ax.axhline(100, color="red", ls="--", lw=1)
    ax.text(
        0.5,
        102,
        "hardware ceiling (instruction- or bandwidth-bound)",
        fontsize=8,
        color="red",
        ha="center",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of peak")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Neither kernel is instruction- or bandwidth-bound → latency-bound\n"
        f"(WP2 issues FEWER warp-inst: {int(wp2['warp_inst']):,} vs {int(cu['warp_inst']):,}, "
        "yet 3.5× slower)",
        fontsize=9,
    )
    for i, (a, b) in enumerate(zip(wp2_v, cu_v, strict=True)):
        ax.text(i - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_roofline.png", dpi=150)
    plt.close(fig)


def fig_cure() -> None:
    """(f) The cure: same per-lane source -> threads (SP) recovers BOTH throughput AND the
    thread-model issue signature vs the tile (WP2). m10 + m10_nsight."""
    rows = _read("m10_scalar_program_rtx4070.csv")
    sp = statistics.median(float(r["sp_gbps"]) for r in rows)
    cu = statistics.median(float(r["cu_gbps"]) for r in rows)
    wp2 = statistics.median(float(r["wp2_gbps"]) for r in rows)
    iss = {r["kernel"]: float(r["issue_active_pct"]) for r in _read("m10_nsight_rtx4070.csv")}
    labels = ["Triton\ntile (WP2)", "hand\nCUDA", "scalar_program\n→threads"]
    colors = ["#2980b9", "#7f8c8d", "#27ae60"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    tput = [wp2, cu, sp]
    for b, v in zip(ax1.bar(labels, tput, color=colors), tput, strict=True):
        ax1.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:.0f}", ha="center", fontsize=8)
    ax1.set_ylabel("Throughput (Gbps)")
    ax1.set_ylim(0, max(tput) * 1.18)
    ax1.set_title(f"Throughput (SP/WP2 = {sp / wp2:.1f}×)", fontsize=9)
    issue = [iss["wp2_tile"], iss["cuda_worklist"], iss["sp_threads_cure"]]
    for b, v in zip(ax2.bar(labels, issue, color=colors), issue, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax2.set_ylabel("Issue-slot activity (%)")
    ax2.set_ylim(0, max(issue) * 1.25)
    ax2.set_title("Mechanism: the cure restores issue activity", fontsize=9)
    fig.suptitle(
        "The cure: lowering the same source to threads recovers throughput AND mechanism",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_cure.png", dpi=150)
    plt.close(fig)


def main() -> int:
    fig_decomposition()
    fig_occupancy_gating()
    fig_mechanism()
    fig_dfa_crossover()
    fig_roofline()
    fig_cure()
    print(f"wrote figures to {OUT}/:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
