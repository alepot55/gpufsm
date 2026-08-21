"""Recompute every straggler-law number quoted in the paper directly from the CSVs.

The law and its out-of-sample check were originally fitted ad hoc, so the quoted
figures were not regenerable. This script is the canonical source: it prints the
fit, the floor, the correlations and the held-out errors, and reports each
per-run block separately so an appended rerun cannot silently move a number.

Predictions use the ROUNDED coefficients the paper publishes, because freezing those
is the method the paper declares. The unrounded refit is printed beside them, on its
own labelled line, so the small gap between the two is visible and explained instead
of looking like a number that fails to reproduce.

Usage: python experiments/cure/verify_straggler_law.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

PREDICTIVE = Path("paper2/data/landmark/cure_predictive_rtx4070.csv")
HELDOUT = Path("paper2/data/landmark/cure_heldout_rtx4070.csv")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def split_runs(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Split an append-only CSV into the successive full sweeps it contains."""
    runs: list[list[dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        tag = (row["dist"], row["mode"])
        if tag in seen:
            runs.append([])
            seen = set()
        seen.add(tag)
        if not runs:
            runs.append([])
        runs[-1].append(row)
    return runs


def by_dist(run: list[dict[str, str]], mode: str) -> dict[str, dict[str, float]]:
    out = {}
    for row in run:
        if row["mode"] != mode:
            continue
        out[row["dist"]] = {
            "warpmax": float(row["warpmax_mean"]),
            "trip": float(row["trip_mean"]),
            "D": float(row["D"]),
            "time": float(row["time_us"]),
            "oracle": row["oracle"],
        }
    return out


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares y = a + b*x, returning (a, b, R^2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1.0 - ss_res / ss_tot


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5


def report_run(label: str, pred: list[dict[str, str]], held: list[dict[str, str]]) -> None:
    off, retire = by_dist(pred, "off"), by_dist(pred, "retire")
    dists = sorted(off)

    warpmax = [off[d]["warpmax"] for d in dists]
    masked = [off[d]["time"] for d in dists]
    a, b, r2 = linfit(warpmax, masked)

    # The paper publishes the fit ROUNDED, t = 32.4 + 1.091 * E[warp-max], and its
    # out-of-sample check freezes exactly those published coefficients. Predicting with
    # the unrounded refit instead is a different method: it moves the worst held-out
    # error from 2.1% to 2.2%, which has already been read once as a paper number that
    # does not reproduce. It is not. Both are printed, the published pair first, so the
    # gap is visible and labelled rather than looking like a contradiction.
    a_pub, b_pub = round(a, 1), round(b, 3)

    floors = [retire[d]["time"] for d in dists]
    speedup = {d: off[d]["time"] / retire[d]["time"] for d in dists}

    print(f"\n=== {label} ===")
    bad = [d for d in dists if off[d]["oracle"] != "OK" or retire[d]["oracle"] != "OK"]
    print(f"oracle: {'all OK' if not bad else 'FAIL on ' + ','.join(bad)}")
    print(
        f"straggler law : t_masked = {a_pub} + {b_pub} * E[warp-max] us   R^2 = {r2:.3f}"
        "   <- published, and used for every prediction below"
    )
    print(f"                (unrounded refit, for reference: {a:.4f} + {b:.6f} * E[warp-max])")
    # pstdev (ddof=0) matches the numpy convention the paper's figure used.
    print(f"cured floor   : {statistics.mean(floors):.1f} +/- {statistics.pstdev(floors):.1f} us")

    dv = [off[d]["D"] for d in dists]
    _, _, r2_d = linfit(dv, masked)
    print(f"divergence-ratio fit of masked time: R^2 = {r2_d:.2f}")
    print(f"corr(speedup, straggler) = {pearson(warpmax, [speedup[d] for d in dists]):.2f}")
    print(f"corr(speedup, D)         = {pearson(dv, [speedup[d] for d in dists]):.2f}")

    print("  per-distribution speedup:")
    errs = []
    for d in dists:
        pred_t = a_pub + b_pub * off[d]["warpmax"]
        pred_s = pred_t / statistics.mean(floors)
        err = abs(pred_s - speedup[d]) / speedup[d] * 100
        errs.append(err)
        print(
            f"    {d:16} D={off[d]['D']:5.2f}  measured {speedup[d]:5.2f}x  "
            f"model {pred_s:5.2f}x  err {err:4.1f}%"
        )
    print(f"  model predicts every speedup to within {max(errs):.1f}%")

    if held:
        h_off, h_ret = by_dist(held, "off"), by_dist(held, "retire")
        print(f"  held-out (published coefficients {a_pub} + {b_pub}, frozen):")
        h_errs, h_errs_raw = [], []
        for d in sorted(h_off):
            pred_t = a_pub + b_pub * h_off[d]["warpmax"]
            err = abs(pred_t - h_off[d]["time"]) / h_off[d]["time"] * 100
            h_errs.append(err)
            h_errs_raw.append(
                abs(a + b * h_off[d]["warpmax"] - h_off[d]["time"]) / h_off[d]["time"] * 100
            )
            sp = h_off[d]["time"] / h_ret[d]["time"] if d in h_ret else float("nan")
            print(
                f"    {d:16} measured {h_off[d]['time']:6.1f}us  predicted {pred_t:7.2f}us  "
                f"err {err:5.3f}%   speedup {sp:5.2f}x"
            )
        print(
            f"  out-of-sample error: mean {statistics.mean(h_errs):.3f}% -> "
            f"{statistics.mean(h_errs):.1f}%   max {max(h_errs):.3f}% -> {max(h_errs):.1f}%"
            "   <- THE PAPER'S NUMBERS"
        )
        print(
            f"  unrounded refit, for reference: mean {statistics.mean(h_errs_raw):.3f}%  "
            f"max {max(h_errs_raw):.3f}%  (a different method, not a failure to reproduce)"
        )


def main() -> int:
    pred_runs = split_runs(load(PREDICTIVE))
    held_runs = split_runs(load(HELDOUT))
    print(f"predictive sweeps in CSV: {len(pred_runs)}   held-out sweeps: {len(held_runs)}")

    # The sweeps come from DIFFERENT builds and must never be pooled: sweep 1 is
    # the old local build, sweep 2 the pinned wheel recipe the paper quotes. The
    # two are reported side by side precisely to show the slope is build-stable.
    labels = ["sweep 1 (old local build)", "sweep 2 (pinned wheel recipe - PAPER)"]
    for i, pred in enumerate(pred_runs):
        held = held_runs[i] if i < len(held_runs) else []
        report_run(labels[i] if i < len(labels) else f"sweep {i + 1}", pred, held)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
