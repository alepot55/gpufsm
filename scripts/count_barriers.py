"""Count the CTA barriers Triton emits for a workload, per compiled kernel.

Used to decide, with numbers instead of opinions, whether a membar change removes any
real barrier: run the same driver commands before and after a compiler patch and diff the
counts. Each command gets a fresh Triton cache so nothing is inherited from a prior run.

    python count_barriers.py --label main --cmd "pytest -q python/tutorials/gluon/04-tma.py"

Output is CSV on stdout (`label,kernel,barriers,mbar_wait,mbar_arrive,ptx_lines`), so runs
from different builds can simply be concatenated and compared.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

# `bar.sync` is the classic form; `barrier.sync` is what ptxas emits for named barriers.
BARRIER = re.compile(r"\b(?:bar|barrier)\.sync\b")
MBAR_WAIT = re.compile(r"\bmbarrier\.(?:try_)?wait\b")
MBAR_ARRIVE = re.compile(r"\bmbarrier\.arrive\b")


def _kernel_name(ptx: str, fallback: str) -> str:
    m = re.search(r"\.visible \.entry (\w+)", ptx)
    return m.group(1) if m else fallback


def run(label: str, cmds: list[str], repo: str) -> int:
    rows: list[tuple] = []
    failed = 0
    for cmd in cmds:
        cache = tempfile.mkdtemp(prefix="tcache-")
        env = {**os.environ, "TRITON_CACHE_DIR": cache}
        r = subprocess.run(cmd, shell=True, cwd=repo, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            failed += 1
            print(f"# FAILED (rc={r.returncode}) {cmd}", file=sys.stderr)
            print(r.stdout[-3000:], file=sys.stderr)
            print(r.stderr[-3000:], file=sys.stderr)
        for path in sorted(pathlib.Path(cache).rglob("*.ptx")):
            ptx = path.read_text(errors="replace")
            rows.append(
                (
                    label,
                    _kernel_name(ptx, path.stem),
                    len(BARRIER.findall(ptx)),
                    len(MBAR_WAIT.findall(ptx)),
                    len(MBAR_ARRIVE.findall(ptx)),
                    ptx.count("\n"),
                )
            )

    print("label,kernel,barriers,mbar_wait,mbar_arrive,ptx_lines")
    for row in sorted(rows, key=lambda r: (r[1], r[2])):
        print(",".join(str(x) for x in row))
    total = sum(r[2] for r in rows)
    print(f"# {label}: {len(rows)} kernels, {total} barriers, {failed} failing command(s)")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="tag for this build (e.g. main / patched)")
    ap.add_argument("--cmd", action="append", required=True, help="driver command (repeatable)")
    ap.add_argument("--repo", default="/work/triton", help="Triton checkout to run from")
    args = ap.parse_args()
    return run(args.label, args.cmd, args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
