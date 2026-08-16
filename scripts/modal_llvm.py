"""Build MLIR from llvm-project on Modal and run mlir-opt / lit against it.

Same shape as scripts/modal_triton.py: the checkout lives in a Modal Volume so the
expensive first build is paid once, and later builds are incremental via ccache.

    python modal_llvm.py build                       # main, no patch
    python modal_llvm.py build --tree fix --patch p.patch
    python modal_llvm.py run --tree fix --cmd "mlir-opt /work/case.mlir --control-flow-sink"

CPU only: MLIR needs no GPU.
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import sys

VOLUME = "llvm-upstream"
WORK = "/work"
UPSTREAM = "https://github.com/llvm/llvm-project"
BUILD_CPUS = 32.0
BUILD_MEMORY_MB = 65536
BUILD_TIMEOUT = 7200

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]


def tree_paths(tree: str) -> tuple[str, str]:
    repo = f"{WORK}/llvm" if tree == "main" else f"{WORK}/llvm-{tree}"
    return repo, f"{repo}/build"


def _sh(cmd: str, cwd: str | None = None) -> str:
    import subprocess

    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout[-4000:], flush=True)
    if p.returncode != 0:
        print(p.stderr[-6000:], flush=True)
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p.stdout


if modal is not None:
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "git", "cmake", "ninja-build", "ccache", "clang", "lld", "zlib1g-dev", "curl"
        )
        .pip_install("lit")
    )
    app = modal.App("llvm-upstream")

    @app.function(
        image=image,
        volumes={WORK: volume},
        cpu=BUILD_CPUS,
        memory=BUILD_MEMORY_MB,
        timeout=BUILD_TIMEOUT,
    )
    def build(ref: str, patch_b64: str | None, patch_name: str, tree: str) -> dict:
        import os

        repo, build_dir = tree_paths(tree)
        if not os.path.exists(repo):
            # A shallow single-branch clone: the history is not what we are here for.
            _sh(f"git clone --depth 200 {UPSTREAM} {repo}")
        _sh("git fetch --depth 200 origin main", cwd=repo)
        _sh("git checkout -f " + (ref or "origin/main"), cwd=repo)
        _sh("git clean -fdx mlir/ llvm/ || true", cwd=repo)

        if patch_b64:
            p = f"{WORK}/{patch_name}"
            with open(p, "wb") as f:
                f.write(base64.b64decode(patch_b64))
            _sh(f"git apply -v {p}", cwd=repo)

        os.environ["CCACHE_DIR"] = f"{WORK}/ccache"
        _sh(
            "cmake -G Ninja -S llvm -B build "
            "-DLLVM_ENABLE_PROJECTS=mlir "
            "-DLLVM_TARGETS_TO_BUILD=host "
            "-DCMAKE_BUILD_TYPE=Release "
            "-DLLVM_ENABLE_ASSERTIONS=ON "
            "-DLLVM_USE_LINKER=lld "
            "-DLLVM_CCACHE_BUILD=ON "
            "-DLLVM_INCLUDE_BENCHMARKS=OFF "
            "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++",
            cwd=repo,
        )
        _sh("ninja -C build mlir-opt FileCheck not count", cwd=repo)
        volume.commit()
        head = _sh("git rev-parse HEAD", cwd=repo).strip()
        return {"tree": tree, "head": head, "bin": f"{build_dir}/bin/mlir-opt"}

    @app.function(
        image=image,
        volumes={WORK: volume},
        cpu=8.0,
        memory=16384,
        timeout=3600,
    )
    def run(cmd: str, tree: str) -> str:
        # Returns the output rather than printing it: remote stdout does not reach
        # the local terminal reliably here, and a silent run reads like a hang.
        import os
        import subprocess

        repo, build_dir = tree_paths(tree)
        env = dict(os.environ)
        env["PATH"] = f"{build_dir}/bin:" + env["PATH"]
        p = subprocess.run(cmd, shell=True, cwd=repo, env=env, capture_output=True, text=True)
        out = [p.stdout[-20000:]]
        if p.stderr:
            out.append("--- stderr ---\n" + p.stderr[-20000:])
        out.append(f"(rc={p.returncode})")
        return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", default="")
    b.add_argument("--patch", default=None)
    b.add_argument("--tree", default="main")
    r = sub.add_parser("run")
    r.add_argument("--cmd", required=True)
    r.add_argument("--tree", default="main")
    a = ap.parse_args()

    if modal is None:
        print("modal non installato: pip install modal", file=sys.stderr)
        return 2

    if a.cmd == "build":
        pb, pn = None, ""
        if a.patch:
            raw = pathlib.Path(a.patch).read_bytes()
            pb, pn = base64.b64encode(raw).decode(), pathlib.Path(a.patch).name
        with app.run():
            out = build.remote(a.ref, pb, pn, a.tree)
        print(out)
    else:
        with app.run():
            print(run.remote(a.cmd, a.tree))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
