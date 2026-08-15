"""Build Triton from source on Modal and run things against it on a rented GPU.

Companion to `modal_gpu.py`, for the upstream work rather than for gpufsm: `modal_gpu.py`
mounts this repo into a container with a pip-installed Triton, which is useless when the
question *is* what the Triton compiler does. Here the checkout lives in a Modal Volume, so
the expensive part (LLVM download + C++ build) is paid once and every later run is
incremental.

    python scripts/modal_triton.py build                      # main, no patch
    python scripts/modal_triton.py build --ref <sha> --patch p.patch
    python scripts/modal_triton.py run --cmd "pytest -q python/test/unit/language/test_core.py -k histogram"
    python scripts/modal_triton.py opt --file case.mlir --args "--test-print-membar"

The build runs on CPU only (a GPU is not needed to compile, and CPU minutes are much
cheaper); `run` is the one that rents the GPU. Both share the same Volume.
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import sys

VOLUME = "triton-upstream"
WORK = "/work"
REPO = f"{WORK}/triton"
VENV = f"{WORK}/venv"
PY = f"{VENV}/bin/python"
UPSTREAM = "https://github.com/triton-lang/triton"

# Ada/Hopper/Blackwell: what we can actually rent. Keeping the list short keeps the build
# short — Triton compiles a backend per listed target.
GPU_DEFAULT = "H100"
BUILD_CPUS = 16.0
BUILD_MEMORY_MB = 32768
BUILD_TIMEOUT = 5400
RUN_TIMEOUT = 3600

try:
    import modal
except ImportError:  # the CLI still prints a useful message without modal installed
    modal = None  # type: ignore[assignment]


def _sh(cmd: str, cwd: str | None = None, env: dict[str, str] | None = None) -> str:
    """Run a shell command, streaming it to the log, and return stdout."""
    import os
    import subprocess

    print(f"$ {cmd}", flush=True)
    r = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    if r.stdout:
        print(r.stdout[-8000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-8000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"command failed (rc={r.returncode}): {cmd}")
    return r.stdout


if modal is not None:
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
        )
        .apt_install(
            "git", "cmake", "ninja-build", "ccache", "clang", "lld", "zlib1g-dev", "curl"
        )
        .pip_install("wheel", "setuptools", "pybind11", "ninja", "cmake")
    )
    app = modal.App("triton-upstream")

    @app.function(
        image=image,
        volumes={WORK: volume},
        cpu=BUILD_CPUS,
        memory=BUILD_MEMORY_MB,
        timeout=BUILD_TIMEOUT,
    )
    def build(ref: str, patch_b64: str | None, patch_name: str) -> dict:
        """Check out `ref`, optionally apply a patch, and (re)build Triton in the Volume."""
        import os

        env = {
            "CCACHE_DIR": f"{WORK}/ccache",
            "TRITON_BUILD_WITH_CCACHE": "true",
            "MAX_JOBS": str(int(BUILD_CPUS)),
        }
        os.makedirs(f"{WORK}/ccache", exist_ok=True)

        if not os.path.exists(f"{REPO}/.git"):
            _sh(f"git clone {UPSTREAM} {REPO}")
        _sh("git fetch origin --tags --force", cwd=REPO)
        _sh("git reset --hard && git clean -fd", cwd=REPO)
        _sh(f"git checkout --detach {ref}", cwd=REPO)
        head = _sh("git rev-parse HEAD", cwd=REPO).strip()

        applied = None
        if patch_b64:
            path = f"{WORK}/{patch_name}"
            pathlib.Path(path).write_bytes(base64.b64decode(patch_b64))
            # `git apply` keeps the tree at `ref` + patch, with no commit noise.
            _sh(f"git apply --3way --verbose {path}", cwd=REPO)
            applied = patch_name

        if not os.path.exists(PY):
            _sh(f"python3 -m venv {VENV}")
            _sh(f"{PY} -m pip install -q --upgrade pip wheel setuptools")
            _sh(f"{PY} -m pip install -q torch numpy pytest")

        _sh(f"{PY} -m pip install -e . --no-build-isolation -v", cwd=REPO, env=env)
        version = _sh(
            f"{PY} -c \"import triton;print(triton.__version__, triton.__file__)\""
        ).strip()
        volume.commit()
        return {"head": head, "patch": applied, "triton": version}

    @app.function(
        image=image,
        volumes={WORK: volume},
        gpu=GPU_DEFAULT,
        timeout=RUN_TIMEOUT,
        memory=BUILD_MEMORY_MB,
    )
    def run(cmds: list[str], upload: dict[str, str] | None = None) -> list[dict]:
        """Run commands against the built Triton on a GPU. Files in `upload` land in /work."""
        import subprocess

        for name, b64 in (upload or {}).items():
            dest = pathlib.Path(WORK) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(b64))

        out = []
        for cmd in cmds:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=REPO,
                capture_output=True,
                text=True,
                env={
                    "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/root",
                    "TRITON_CACHE_DIR": f"{WORK}/triton-cache",
                },
            )
            out.append(
                {
                    "cmd": cmd,
                    "rc": r.returncode,
                    "stdout": r.stdout[-30000:],
                    "stderr": r.stderr[-8000:],
                }
            )
        return out


def _b64(path: str | None) -> tuple[str | None, str]:
    if not path:
        return None, ""
    p = pathlib.Path(path)
    return base64.b64encode(p.read_bytes()).decode(), p.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)

    b = sub.add_parser("build", help="build Triton in the Modal Volume")
    b.add_argument("--ref", default="main", help="commit/branch to check out")
    b.add_argument("--patch", help="patch file applied on top of --ref")

    r = sub.add_parser("run", help="run commands on the GPU against the built Triton")
    r.add_argument("--cmd", action="append", required=True)
    r.add_argument("--upload", action="append", default=[], help="file to copy to /work")

    o = sub.add_parser("opt", help="shortcut: run triton-opt on one MLIR file")
    o.add_argument("--file", required=True)
    o.add_argument("--args", default="--test-print-membar")

    args = ap.parse_args()
    if modal is None:
        print("modal is not installed: pip install modal", file=sys.stderr)
        return 2

    with app.run():
        if args.action == "build":
            patch_b64, patch_name = _b64(args.patch)
            res = build.remote(args.ref, patch_b64, patch_name)
            print(f"built {res['triton']}\n  head={res['head']}\n  patch={res['patch']}")
            return 0

        if args.action == "opt":
            payload, name = _b64(args.file)
            assert payload
            cmds = [f"{REPO}/build/*/bin/triton-opt {args.args} {WORK}/{name}"]
            sections = run.remote(cmds, {name: payload})
        else:
            uploads = {}
            for path in args.upload:
                payload, name = _b64(path)
                assert payload
                uploads[name] = payload
            sections = run.remote(args.cmd, uploads)

        failed = 0
        for s in sections:
            print(f"\n===== {s['cmd']} (rc={s['rc']}) =====")
            print(s["stdout"])
            if s["rc"] != 0:
                failed += 1
                print(f"--- stderr ---\n{s['stderr']}")
        print(f"== DONE ({failed} failing command(s)) ==")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
