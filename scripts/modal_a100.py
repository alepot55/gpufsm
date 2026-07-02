"""A100 cross-arch validation on Modal (free-credits friendly).

Runs the same two phases as scripts/a100_validate.sh on a Modal A100:
  PHASE 1 (stock Triton + nvcc): the regret-law witnesses (p3_cross_arch.py).
  PHASE 2 (prebuilt Triton with the per-lane retirement pass, shipped from
           the local build -- no 50-min build on the pod): the built-cure
           headline (bench_perlane_retire.py, oracle-exact, PTX evidence).

The local Triton tree is staged with symlinks dereferenced (the nvidia
backend is a symlink out of python/) and mounted read-only; results are
returned and written into paper2/data/cross_arch/ locally.

Run:  .venv/bin/modal run scripts/modal_a100.py
      (CHEAP only: .venv/bin/modal run scripts/modal_a100.py --no-cure)
"""

import os
import pathlib
import subprocess

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parents[1]
LOCAL_TRITON_PY = pathlib.Path.home() / "m3full_build/triton-src/python"
STAGING = pathlib.Path.home() / ".cache/gpufsm_modal_triton"

REPO = "/root/gpufsm"
TRITON_PY = "/root/tsrc/python"

app = modal.App("gpufsm-a100-validate")


def _stage_triton() -> pathlib.Path:
    """Copy the local Triton python tree with symlinks dereferenced."""
    marker = STAGING / "triton/_C/libtriton.so"
    src_so = LOCAL_TRITON_PY / "triton/_C/libtriton.so"
    if not marker.exists() or marker.stat().st_mtime < src_so.stat().st_mtime:
        STAGING.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-aL", "--delete", f"{LOCAL_TRITON_PY}/", str(STAGING)],
            check=True,
        )
    return STAGING


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install("torch", "numpy", "scipy")
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=REPO,
        ignore=[".git", ".venv", "build", "**/__pycache__", "*.pdf", ".claude"],
    )
    .add_local_dir(str(_stage_triton()), remote_path=TRITON_PY)
)


@app.function(image=image, gpu="A100", timeout=3600)
def validate(cure: bool = True) -> dict:
    import os
    import pathlib
    import subprocess
    import sys

    os.chdir(REPO)
    out: dict = {}
    gpu = subprocess.run(
        [sys.executable, "-c",
         "import torch;print(torch.cuda.get_device_name(0).replace(' ','_'))"],
        capture_output=True, text=True).stdout.strip()
    out["gpu"] = gpu

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    r1 = subprocess.run(
        [sys.executable, "experiments/cure/p3_cross_arch.py"],
        env=env, capture_output=True, text=True, timeout=2400)
    out["phase1_log"] = (r1.stdout + r1.stderr)[-12000:]

    if cure:
        env2 = os.environ.copy()
        env2["PYTHONPATH"] = f"{TRITON_PY}:{REPO}"
        r2 = subprocess.run(
            [sys.executable, "experiments/cure/bench_perlane_retire.py"],
            env=env2, capture_output=True, text=True, timeout=1200)
        out["phase2_log"] = (r2.stdout + r2.stderr)[-8000:]

    files = {}
    d = pathlib.Path("paper2/data/cross_arch")
    if d.exists():
        for f in d.iterdir():
            if f.is_file() and "rtx_4070" not in f.name.lower():
                files[f.name] = f.read_text(errors="replace")
    out["files"] = files
    return out


@app.local_entrypoint()
def main(cure: bool = True):
    res = validate.remote(cure=cure)
    print(f"== GPU: {res['gpu']} ==")
    print("== PHASE 1 (regret law) tail ==")
    print(res["phase1_log"][-3000:])
    if "phase2_log" in res:
        print("== PHASE 2 (built cure) ==")
        print(res["phase2_log"][-2000:])
    outdir = LOCAL_REPO / "paper2/data/cross_arch"
    outdir.mkdir(parents=True, exist_ok=True)
    gpu = res["gpu"]
    for name, content in res["files"].items():
        (outdir / name).write_text(content)
        print(f"saved paper2/data/cross_arch/{name}")
    (outdir / f"phase1_{gpu}.log").write_text(res["phase1_log"])
    if "phase2_log" in res:
        (outdir / f"cure_{gpu}.log").write_text(res["phase2_log"])
    print("== DONE — results saved locally; container stops automatically (no pod to shut down). ==")
