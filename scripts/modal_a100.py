"""A100 cross-arch validation on Modal (free-credits friendly).

Runs PHASE 1 of the datacenter validation: the regret-law witnesses
(p3_cross_arch.py) on an A100 with stock Triton + nvcc. This is the number
that closes the "single consumer GPU" gap for the TACO paper's core thesis
(the regret follows the execution paradigm, not the arch). No custom Triton
build/upload -- robust and cheap (~$0.3 of Modal's free $30/mo credits).

The container stops itself when done (no pod to shut down). Results are
returned in-band and written to paper2/data/cross_arch/ locally.

Run:  .venv/bin/python -m modal run scripts/modal_a100.py
"""

import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parents[1]
REPO = "/root/gpufsm"

app = modal.App("gpufsm-a100-validate")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch", "numpy", "scipy", "triton")
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=REPO,
        ignore=[
            ".git",
            ".venv",
            "build",
            "**/__pycache__",
            "*.pdf",
            ".claude",
            "paper2/*.aux",
            "paper2/*.log",
        ],
    )
)


@app.function(image=image, gpu="A100", timeout=2400)
def validate() -> dict:
    import os
    import pathlib
    import subprocess
    import sys

    os.chdir(REPO)
    out: dict = {}
    gpu = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch;print(torch.cuda.get_device_name(0).replace(' ','_'))",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    out["gpu"] = gpu

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    r1 = subprocess.run(
        [sys.executable, "experiments/cure/p3_cross_arch.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=2000,
    )
    out["phase1_log"] = (r1.stdout + r1.stderr)[-16000:]

    files = {}
    d = pathlib.Path("paper2/data/cross_arch")
    if d.exists():
        for f in d.iterdir():
            if f.is_file() and "rtx_4070" not in f.name.lower():
                files[f.name] = f.read_text(errors="replace")
    out["files"] = files
    return out


@app.local_entrypoint()
def main():
    res = validate.remote()
    print(f"== GPU: {res['gpu']} ==")
    print("== PHASE 1 (regret law) log ==")
    print(res["phase1_log"])
    outdir = LOCAL_REPO / "paper2/data/cross_arch"
    outdir.mkdir(parents=True, exist_ok=True)
    gpu = res["gpu"]
    for name, content in res["files"].items():
        (outdir / name).write_text(content)
        print(f"saved paper2/data/cross_arch/{name}")
    (outdir / f"phase1_{gpu}.log").write_text(res["phase1_log"])
    print("== DONE — results saved locally; container stops automatically. ==")
