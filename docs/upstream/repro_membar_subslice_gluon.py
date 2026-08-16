import sys
sys.path.insert(0, sys.argv[1] + "/python/test/gluon")
from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

SH = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
BL = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[8, 4], warps_per_cta=[4, 1], order=[1, 0])

@gluon.jit
def kernel(SH: ttgl.constexpr, BL: ttgl.constexpr):
    buf = ttgl.allocate_shared_memory(ttgl.float16, [32, 16], SH)
    lo = buf.slice(16, 16, dim=0)                       # righe 16..31 -> byte [512,1024)
    lo.store(ttgl.full([16, 16], 1, ttgl.float16, layout=BL))
    same = buf.reinterpret(shape=[16, 32]).slice(8, 8, dim=0)   # stessi byte [512,1024)
    v = same.load(layout=BL)                           # noqa: F841

mod = run_parser(kernel, (SH, BL), {"num_warps": 4}, target=GPUTarget("cuda", 90, 32))
print(mod.str_nodebug())
