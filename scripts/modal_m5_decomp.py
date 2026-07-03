"""M5: run the FLAGSHIP decomposition on an A100 (2nd GPU) for the TACO paper.

Addresses the reviewer's M5: every flagship quantitative result (the NFA
worklist decomposition, Component A/B, the throughput ladder / Table 1) is
RTX-4070-only. This builds gpufsm with its CUDA extension on the A100 (the
CMakeLists targets 80-real, so sm_80 is compiled) and re-runs the four
decomposition scripts. Each writes a *_rtx4070.csv; we snapshot+restore the
committed baseline and return this GPU's output tagged separately.

Run:  .venv/bin/python -m modal run scripts/modal_m5_decomp.py
"""

import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parents[1]
REPO = "/root/gpufsm"

app = modal.App("gpufsm-m5-decomp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "g++", "gcc", "clang")
    .pip_install("torch", "numpy", "scipy", "triton", "cmake", "ninja",
                 "scikit-build-core", "pybind11")
    .add_local_dir(
        str(LOCAL_REPO), remote_path=REPO,
        ignore=[".git", ".venv", "build", "**/__pycache__", "*.pdf", ".claude",
                "paper2/*.aux", "paper2/*.log", "*.so"],
    )
)

SCRIPTS = [
    "experiments.cure.m0_anchor",
    "experiments.cure.m2f_numwarps",
    "experiments.cure.m2_lane_packed",
    "experiments.cure.m2e_worklist_packed",
]
CSVS = [
    "paper2/data/m0_anchor_rtx4070.csv",
    "paper2/data/m2f_numwarps_rtx4070.csv",
    "paper2/data/m2_lane_packed_rtx4070.csv",
    "paper2/data/m2e_worklist_packed_rtx4070.csv",
]


@app.function(image=image, gpu="A100", timeout=3600)
def run() -> dict:
    import os
    import subprocess
    import sys

    os.chdir(REPO)
    out: dict = {}
    gpu = subprocess.run(
        [sys.executable, "-c", "import torch;print(torch.cuda.get_device_name(0))"],
        capture_output=True, text=True).stdout.strip()
    out["gpu"] = gpu

    # Build gpufsm + its CUDA extension for the present arch (sm_80).
    build_env = {**os.environ, "CC": "gcc", "CXX": "g++"}
    b = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[triton]",
         "--config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON", "-q"],
        capture_output=True, text=True, timeout=1800, env=build_env)
    out["build_tail"] = (b.stdout + b.stderr)[-3000:]
    out["build_ok"] = b.returncode == 0

    # snapshot committed baselines, run each script, capture this GPU's CSV, restore
    snap = {c: (pathlib.Path(REPO) / c).read_text() for c in CSVS
            if (pathlib.Path(REPO) / c).exists()}
    logs = {}
    files = {}
    env = {**os.environ, "PYTHONPATH": REPO}
    for mod, csv in zip(SCRIPTS, CSVS):
        p = pathlib.Path(REPO) / csv
        before = p.read_text() if p.exists() else None
        r = subprocess.run([sys.executable, "-m", mod], env=env,
                           capture_output=True, text=True, timeout=1800)
        logs[mod] = (r.stdout + r.stderr)[-2500:]
        # Only capture if the script SUCCEEDED and actually rewrote the file,
        # otherwise we would echo the committed baseline (fake cross-arch data).
        after = p.read_text() if p.exists() else None
        if r.returncode == 0 and after is not None and after != before:
            files[csv] = after
        else:
            logs[mod] += f"\n[skip-capture rc={r.returncode} changed={after != before}]"
        if before is not None:
            p.write_text(before)  # restore committed baseline
    out["logs"] = logs
    out["files"] = files
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    print(f"== GPU: {res['gpu']}  build_ok={res['build_ok']} ==")
    if not res["build_ok"]:
        print("BUILD FAILED:\n", res["build_tail"][-1500:])
    for mod, log in res["logs"].items():
        print(f"\n--- {mod} ---\n{log[-1200:]}")
    outdir = LOCAL_REPO / "paper2/data/cross_arch"
    outdir.mkdir(parents=True, exist_ok=True)
    gpu_slug = res["gpu"].replace(" ", "_").replace("-", "_").lower()
    for csv, content in res["files"].items():
        name = pathlib.Path(csv).name.replace("_rtx4070", f"_{gpu_slug}")
        (outdir / name).write_text(content)
        print(f"saved paper2/data/cross_arch/{name}")
    print("== DONE — container self-stops. ==")
