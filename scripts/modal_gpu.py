"""Run any repo command on a rented Modal GPU, from a laptop or a Claude cloud session.

Generalizes the one-off `modal_a100.py` / `modal_cure_a100.py` jobs: pick a GPU, pass the
commands to run, list the files to bring back. The container stops itself when done, so the
only cost is the seconds actually spent on the GPU (Modal's free tier is $30/month).

    python scripts/modal_gpu.py --preflight
    python scripts/modal_gpu.py --gpu H100 \
        --cmd "python experiments/cure/p3_cross_arch.py" \
        --fetch "paper2/data/cross_arch/*"

Requires `api.modal.com` to be reachable. Inside a Claude cloud session the default
**Trusted** network policy blocks it (403 on CONNECT); see `docs/MODAL_FROM_CLOUD.md` for the
environment settings that open it. `--preflight` reports exactly what is missing.

Implementation note: `modal run` needs the GPU type at import time, so the CLI re-executes
this file under `modal run` with the choices passed through the environment.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib

LOCAL_REPO = pathlib.Path(__file__).resolve().parents[1]
REPO = "/root/gpufsm"
MAX_FETCH_BYTES = 8 * 1024 * 1024

GPU = os.environ.get("GPUFSM_MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("GPUFSM_MODAL_TIMEOUT", "1800"))
PIP = os.environ.get("GPUFSM_MODAL_PIP", "torch numpy triton").split()
CMDS: list[str] = json.loads(
    os.environ.get(
        "GPUFSM_MODAL_CMDS",
        '["python -c \\"import torch;print(torch.cuda.get_device_name(0))\\""]',
    )
)
FETCH: list[str] = json.loads(os.environ.get("GPUFSM_MODAL_FETCH", "[]"))

IGNORE = [
    ".git",
    ".venv",
    "build",
    "**/__pycache__",
    "*.pdf",
    ".claude",
    "paper*/*.aux",
    "paper*/*.log",
]


def _run_remote() -> dict:
    """Body of the Modal function: run the commands, collect the requested files."""
    import glob
    import subprocess
    import sys

    os.chdir(REPO)
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}"}
    try:
        import torch

        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception as exc:  # no torch, or no visible device
        gpu = f"UNKNOWN({exc})"

    sections = []
    for cmd in CMDS:
        r = subprocess.run(
            cmd, shell=True, env=env, capture_output=True, text=True, timeout=TIMEOUT
        )
        sections.append(
            {
                "cmd": cmd,
                "rc": r.returncode,
                "stdout": r.stdout[-20000:],
                "stderr": r.stderr[-4000:],
            }
        )
        print(f"===== {cmd} (rc={r.returncode}) =====", file=sys.stderr)

    files: dict[str, str] = {}
    skipped: list[str] = []
    for pattern in FETCH:
        for path in glob.glob(pattern, recursive=True):
            p = pathlib.Path(path)
            if not p.is_file():
                continue
            rel = os.path.relpath(p.resolve(), REPO)
            if rel.startswith(".."):
                skipped.append(f"{rel} (outside repo)")
                continue
            if p.stat().st_size > MAX_FETCH_BYTES:
                skipped.append(f"{rel} ({p.stat().st_size} bytes)")
                continue
            files[rel] = base64.b64encode(p.read_bytes()).decode()
    return {"gpu": gpu, "sections": sections, "files": files, "skipped": skipped}


# --------------------------------------------------------------------------------------
# Modal app. Imported both by `modal run` and by the CLI below; keep it side-effect free.
# --------------------------------------------------------------------------------------
try:
    import modal
except ImportError:  # --preflight must still work without modal installed
    modal = None  # type: ignore[assignment]

if modal is not None:
    app = modal.App("gpufsm-modal-gpu")
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
        .pip_install(*PIP)
        .add_local_dir(str(LOCAL_REPO), remote_path=REPO, ignore=IGNORE)
    )

    remote_run = app.function(image=image, gpu=GPU, timeout=TIMEOUT + 300)(_run_remote)

    @app.local_entrypoint()
    def main() -> None:
        res = remote_run.remote()
        print(f"== GPU: {res['gpu']} ==")
        failed = 0
        for s in res["sections"]:
            print(f"\n===== {s['cmd']} (rc={s['rc']}) =====")
            print(s["stdout"])
            if s["rc"] != 0:
                failed += 1
                print(f"--- stderr ---\n{s['stderr']}")
        for rel, b64 in res["files"].items():
            dest = LOCAL_REPO / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(b64))
            print(f"saved {rel}")
        for s in res["skipped"]:
            print(f"NOT fetched: {s}")
        print(f"== DONE ({failed} failing command(s)); container stopped. ==")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _proxy_ca_missing_from_certifi() -> str | None:
    """Return the proxy CA path when it is absent from certifi's bundle, else None.

    Modal's gRPC channel and its blob client both build their SSL context from
    `certifi.where()`, so a TLS-terminating egress proxy stays untrusted no matter what
    SSL_CERT_FILE says: the CA has to be appended to that bundle.
    """
    candidates = [pathlib.Path("/root/.ccr/agent-proxy-ca.crt")]
    ssl_env = os.environ.get("SSL_CERT_FILE")
    if ssl_env:
        candidates.insert(0, pathlib.Path(ssl_env).with_name("agent-proxy-ca.crt"))
    try:
        import certifi

        bundle = pathlib.Path(certifi.where()).read_text()
    except Exception:
        return None
    for ca in candidates:
        if ca.exists() and ca.read_text().strip() not in bundle:
            return str(ca)
    return None


def _trust_proxy_ca_in_certifi() -> str | None:
    """Append the session's proxy CA to certifi's bundle when missing. Returns what it fixed.

    Done here rather than in the environment's setup script because `/root/.ccr/` does not
    exist yet at that point: the proxy is materialized after setup runs. Idempotent, gated on
    HTTPS_PROXY being set and on the CA file existing, so it never touches a normal machine.
    """
    if not os.environ.get("HTTPS_PROXY"):
        return None
    missing = _proxy_ca_missing_from_certifi()
    if missing is None:
        return None
    try:
        import certifi

        with pathlib.Path(certifi.where()).open("a") as fh:
            fh.write("\n" + pathlib.Path(missing).read_text().strip() + "\n")
    except Exception:
        return None  # best effort; the preflight check re-tests and reports the manual fix
    return missing


def preflight() -> int:
    """Report whether this machine can drive Modal. Returns a process exit code."""
    import urllib.error
    import urllib.request

    checks: list[tuple[str, bool, str]] = []

    checks.append(("modal installed", modal is not None, "pip install 'modal[api-proxy-support]'"))

    reachable, detail = False, ""
    try:
        urllib.request.urlopen("https://api.modal.com/", timeout=15)
        reachable = True
    except urllib.error.HTTPError as exc:  # the host answered: egress is open
        reachable, detail = True, f"HTTP {exc.code}"
    except Exception as exc:
        detail = str(exc)
    checks.append(
        (
            "api.modal.com reachable",
            reachable,
            f"blocked ({detail}) -> set the cloud environment's network access to Full, or "
            "Custom with *.modal.com; see docs/MODAL_FROM_CLOUD.md",
        )
    )

    has_tokens = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    has_config = (pathlib.Path.home() / ".modal.toml").exists()
    checks.append(
        (
            "modal credentials",
            has_tokens or has_config,
            "`modal token set --token-id ak-... --token-secret as-... --no-verify`, or set "
            "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (modal.com/settings/tokens)",
        )
    )

    if os.environ.get("HTTPS_PROXY") and modal is not None:
        try:
            import importlib.metadata as md

            md.version("aiohttp-socks")
            proxy_ok = True
        except Exception:
            proxy_ok = False
        checks.append(
            (
                "proxy support (HTTPS_PROXY is set)",
                proxy_ok,
                "pip install 'modal[api-proxy-support]' so the client honours HTTPS_PROXY",
            )
        )

        appended = _trust_proxy_ca_in_certifi()
        missing_ca = _proxy_ca_missing_from_certifi()
        checks.append(
            (
                "proxy CA trusted by certifi" + (" (appended just now)" if appended else ""),
                missing_ca is None,
                f"cat {missing_ca} >> \"$(python -c 'import certifi;print(certifi.where())')\"",
            )
        )

    failed = 0
    for name, ok, fix in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
            print(f"       fix: {fix}")
    print("preflight OK" if not failed else f"{failed} check(s) failed")
    return 0 if not failed else 1


def cli() -> int:
    import argparse
    import subprocess
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preflight", action="store_true", help="check connectivity and credentials")
    ap.add_argument("--gpu", default="A100", help="Modal GPU: A100, A100-80GB, H100, L40S, T4 ...")
    ap.add_argument("--cmd", action="append", default=[], help="command to run (repeatable)")
    ap.add_argument("--fetch", action="append", default=[], help="glob of files to bring back")
    ap.add_argument("--pip", default="torch numpy triton", help="packages for the container image")
    ap.add_argument("--timeout", type=int, default=1800, help="per-command timeout, seconds")
    ap.add_argument("--dry-run", action="store_true", help="print the modal command and exit")
    args = ap.parse_args()

    if args.preflight:
        return preflight()

    env = {
        **os.environ,
        "GPUFSM_MODAL_CHILD": "1",
        "GPUFSM_MODAL_GPU": args.gpu,
        "GPUFSM_MODAL_TIMEOUT": str(args.timeout),
        "GPUFSM_MODAL_PIP": args.pip,
        "GPUFSM_MODAL_FETCH": json.dumps(args.fetch),
    }
    if args.cmd:
        env["GPUFSM_MODAL_CMDS"] = json.dumps(args.cmd)
    cmd = [sys.executable, "-m", "modal", "run", str(pathlib.Path(__file__).resolve())]
    if args.dry_run:
        shown = {k: v for k, v in env.items() if k.startswith("GPUFSM_MODAL_")}
        print(json.dumps(shown, indent=2))
        print(" ".join(cmd))
        return 0
    if appended := _trust_proxy_ca_in_certifi():
        print(f"trusted {appended} in certifi's bundle")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__" and not os.environ.get("GPUFSM_MODAL_CHILD"):
    raise SystemExit(cli())
