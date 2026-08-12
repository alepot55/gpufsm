"""Cure-on-A100 (closes the open half of M5): build Triton + the per-lane-loop-retirement
pass as a wheel on a CPU worker, then run the four cure experiments on an A100.

Two stages to keep GPU-time minimal:
  1. build_wheel (cpu=16): clone triton @ the pinned upstream base, apply the versioned
     plain-diff patch (experiments/cure/triton_thread_region_pass/perlane_retire_full.patch),
     `pip wheel .` -> persisted to a Modal Volume.
  2. run_cure (gpu=A100): install the wheel over stock triton, run
     bench_perlane_retire + cure_generalize + cure_spmv + cure_moe (masked vs cured).

Run:  .venv/bin/python -m modal run scripts/modal_cure_a100.py
Outputs: paper2/data/cross_arch/cure_<gpu>.log (raw stdout, one section per experiment).
"""

import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "81a46fa0c04526e5df55a018ecfab72ff922f592"  # upstream parent of the pass commit
PATCH = "experiments/cure/triton_thread_region_pass/perlane_retire_full.patch"

app = modal.App("gpufsm-cure-a100")
vol = modal.Volume.from_name("gpufsm-triton-wheel", create_if_missing=True)

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "g++", "gcc", "cmake", "ninja-build",
                 "zlib1g-dev", "curl", "ca-certificates")
    .pip_install("cmake<4", "ninja", "wheel", "setuptools", "pybind11")
    .add_local_file(str(LOCAL_REPO / PATCH), "/root/perlane.patch")
)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch", "numpy")
    .add_local_dir(str(LOCAL_REPO / "experiments"), "/root/repo/experiments",
                   ignore=["**/__pycache__"])
)


@app.function(image=build_image, cpu=16, memory=65536, timeout=7200, volumes={"/vol": vol})
def build_wheel() -> str:
    import glob
    import os
    import subprocess

    existing = glob.glob("/vol/wheels/triton-*.whl")
    if existing:
        return f"cached: {existing[0]}"

    src = "/root/triton"
    os.makedirs(src, exist_ok=True)

    def run(cmd):
        subprocess.run(cmd, shell=True, check=True, cwd=src)

    run("git init -q .")
    run("git remote add origin https://github.com/triton-lang/triton.git")
    run(f"git fetch -q --depth 1 origin {BASE_COMMIT}")
    run(f"git checkout -q {BASE_COMMIT}")
    run("git apply /root/perlane.patch")
    env = {**os.environ, "MAX_JOBS": "16", "TRITON_PARALLEL_LINK_JOBS": "2",
           "CC": "gcc", "CXX": "g++"}
    r = subprocess.run("pip wheel . -w /vol/wheels --no-deps -v", shell=True, cwd=src,
                       env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-3000:]
    if r.returncode != 0:
        raise RuntimeError(f"wheel build failed:\n{tail}")
    vol.commit()
    built = glob.glob("/vol/wheels/triton-*.whl")
    return f"built: {built}\n...{tail[-500:]}"


@app.function(image=gpu_image, gpu="A100", timeout=2400, volumes={"/vol": vol})
def run_cure() -> dict:
    import glob
    import os
    import subprocess

    wheels = glob.glob("/vol/wheels/triton-*.whl")
    assert wheels, "no wheel in volume"
    subprocess.run(f"pip install -q {wheels[0]} --force-reinstall --no-deps",
                   shell=True, check=True)
    import torch
    gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "NOCUDA"

    base_env = {**os.environ, "TRITON_ALWAYS_COMPILE": "1", "PYTHONPATH": "/root/repo"}
    cure_env = {**base_env, "TRITON_ENABLE_PERLANE_LOOP_RETIREMENT": "1",
                "GPUFSM_THREAD_REGION": "retire"}
    exp = "/root/repo/experiments/cure"
    jobs = [
        ("bench_perlane_retire", f"python {exp}/bench_perlane_retire.py", base_env),
        ("cure_generalize.masked", f"python {exp}/cure_generalize.py", base_env),
        ("cure_generalize.cured", f"python {exp}/cure_generalize.py", cure_env),
        ("cure_spmv.masked", f"python {exp}/cure_spmv.py", base_env),
        ("cure_spmv.cured", f"python {exp}/cure_spmv.py", cure_env),
        ("cure_moe.masked", f"python {exp}/cure_moe.py", base_env),
        ("cure_moe.cured", f"python {exp}/cure_moe.py", cure_env),
        ("cure_rejection.masked", f"python {exp}/cure_rejection.py", base_env),
        ("cure_rejection.cured", f"python {exp}/cure_rejection.py", cure_env),
    ]
    out = {"gpu": gpu, "sections": {}}
    for name, cmd, env in jobs:
        r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True,
                           timeout=900)
        out["sections"][name] = {
            "rc": r.returncode,
            "stdout": r.stdout[-4000:],
            "stderr": r.stderr[-1500:] if r.returncode != 0 else "",
        }
    return out


@app.local_entrypoint()
def main():
    print(build_wheel.remote())
    res = run_cure.remote()
    gpu = res["gpu"].lower()
    dest = LOCAL_REPO / "paper2" / "data" / "cross_arch" / f"cure_{gpu}.log"
    with open(dest, "w") as f:
        f.write(f"# cure-on-A100 raw outputs  gpu={res['gpu']}\n")
        for name, s in res["sections"].items():
            f.write(f"\n===== {name} (rc={s['rc']}) =====\n{s['stdout']}\n")
            if s["stderr"]:
                f.write(f"--- stderr ---\n{s['stderr']}\n")
    print(f"wrote {dest}")
    for name, s in res["sections"].items():
        status = "OK" if s["rc"] == 0 else f"RC={s['rc']}"
        print(f"{name:28} {status}")
