// gpufsm "abstraction regret" research pass — tritongpu-thread-region.
//   default / GPUFSM_THREAD_REGION=1 : DETECT + mark the lock-step signature (an scf.while over
//     #blocked tile iter-args whose scf.condition derives from a tt.reduce of a per-lane predicate).
//   GPUFSM_THREAD_REGION=hoist        : additionally REWRITE the matched while (reduce-hoist): the
//     per-iteration tt.reduce(cmpi slt %j, %trip) gate is replaced by a scalar counter bounded by a
//     once-hoisted reduce_max(%trip). Provably equivalent (j is a uniform splat; the body stays masked
//     by j<trip), removes the per-iteration cross-lane reduce, preserves per-warp termination. (~1.4x;
//     see experiments/cure/f3_reduce_cost.py). It does NOT give per-lane retirement — that needs a
//     below-TritonGPU lowering (the structural wall).
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <cstring>

#define DEBUG_TYPE "tritongpu-thread-region"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTHREADREGION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isBlockedTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa_and_nonnull<BlockedEncodingAttr>(tensor.getEncoding());
}

// Backward walk within the before-region; return the first op of type T feeding `root`.
template <typename T>
static T findInBefore(Value root, scf::WhileOp w) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  SmallVector<Value, 16> wl{root};
  while (!wl.empty()) {
    Value v = wl.pop_back_val();
    Operation *d = v.getDefiningOp();
    if (!d || !seen.insert(d).second)
      continue;
    if (d->getParentRegion() != &w.getBefore())
      continue;
    if (auto t = dyn_cast<T>(d))
      return t;
    for (Value o : d->getOperands())
      wl.push_back(o);
  }
  return nullptr;
}

// The reduce-hoist rewrite. Returns success if it transformed the loop.
static LogicalResult hoistRewrite(scf::WhileOp w) {
  Block &before = w.getBefore().front();
  auto cond = cast<scf::ConditionOp>(before.getTerminator());
  triton::ReduceOp red = findInBefore<triton::ReduceOp>(cond.getCondition(), w);
  if (!red)
    return failure();
  arith::CmpIOp cmp = findInBefore<arith::CmpIOp>(red->getOperand(0), w);
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return failure();

  // Identify %j (a before-block arg, the per-lane counter) and %trip (loop-invariant).
  auto isBeforeArg = [&](Value v) {
    auto ba = dyn_cast<BlockArgument>(v);
    return ba && ba.getOwner() == &before;
  };
  Value jv = cmp.getLhs(), tripv = cmp.getRhs();
  if (!isBeforeArg(jv)) {
    std::swap(jv, tripv);
    if (!isBeforeArg(jv))
      return failure();
  }
  // %trip must be defined outside the while (loop-invariant).
  if (Operation *td = tripv.getDefiningOp())
    if (w->isAncestor(td))
      return failure();
  auto jArg = cast<BlockArgument>(jv);
  unsigned jIdx = jArg.getArgNumber(); // same index in inits / results / after-args

  OpBuilder b(w);
  Location loc = w.getLoc();
  // Hoist %mt = reduce_max(%trip): clone the matched reduce, feeding it %trip directly.
  IRMapping hm;
  hm.map(red->getOperand(0), tripv);
  Operation *mtOp = b.clone(*red.getOperation(), hm);
  Value mt = mtOp->getResult(0); // scalar i32 = max(trip)
  Value zero = arith::ConstantOp::create(b, loc, b.getIntegerAttr(mt.getType(), 0));

  // Rebuild the while with one extra i32 iter-arg (the scalar counter js).
  SmallVector<Value> inits(w.getInits());
  inits.push_back(zero);
  SmallVector<Type> resTypes(w.getResultTypes());
  resTypes.push_back(mt.getType());
  auto nw = scf::WhileOp::create(b, loc, resTypes, inits);

  // --- before region ---
  Block *nb = b.createBlock(&nw.getBefore());
  IRMapping bm;
  for (BlockArgument a : before.getArguments())
    bm.map(a, nb->addArgument(a.getType(), a.getLoc()));
  Value jsBefore = nb->addArgument(mt.getType(), loc);
  b.setInsertionPointToStart(nb);
  for (Operation &op : before.without_terminator())
    b.clone(op, bm);
  // scalar condition: js < mt ; forward the original condition args + js
  Value scond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, jsBefore, mt);
  SmallVector<Value> fwd;
  for (Value v : cond.getArgs())
    fwd.push_back(bm.lookupOrDefault(v));
  fwd.push_back(jsBefore);
  scf::ConditionOp::create(b, loc, scond, fwd);

  // --- after region ---
  Block &after = w.getAfter().front();
  auto yield = cast<scf::YieldOp>(after.getTerminator());
  Block *na = b.createBlock(&nw.getAfter());
  IRMapping am;
  for (BlockArgument a : after.getArguments())
    am.map(a, na->addArgument(a.getType(), a.getLoc()));
  Value jsAfter = na->addArgument(mt.getType(), loc);
  b.setInsertionPointToStart(na);
  for (Operation &op : after.without_terminator())
    b.clone(op, am);
  Value one = arith::ConstantOp::create(b, loc, b.getIntegerAttr(mt.getType(), 1));
  Value jsNext = arith::AddIOp::create(b, loc, jsAfter, one);
  SmallVector<Value> yvals;
  for (Value v : yield.getOperands())
    yvals.push_back(am.lookupOrDefault(v));
  yvals.push_back(jsNext);
  scf::YieldOp::create(b, loc, yvals);

  // Wire results (drop the extra scalar result) and erase the old while.
  w.getOperation()->replaceAllUsesWith(nw.getResults().take_front(w.getNumResults()));
  w.erase();
  (void)jIdx;
  return success();
}

struct ThreadRegionPass : public impl::TritonGPUThreadRegionBase<ThreadRegionPass> {
  void runOnOperation() override {
    const char *mode = std::getenv("GPUFSM_THREAD_REGION");
    bool doHoist = mode && std::strcmp(mode, "hoist") == 0;
    ModuleOp mod = getOperation();
    SmallVector<scf::WhileOp> matched;
    mod.walk([&](scf::WhileOp w) {
      bool carriesTile = false;
      for (Value init : w.getInits())
        if (isBlockedTile(init.getType())) {
          carriesTile = true;
          break;
        }
      if (!carriesTile)
        return;
      auto cond = cast<scf::ConditionOp>(w.getBefore().front().getTerminator());
      if (!findInBefore<triton::ReduceOp>(cond.getCondition(), w))
        return;
      w->setAttr("ttg.thread_region_candidate", UnitAttr::get(&getContext()));
      w->emitRemark("thread_region candidate: lock-step tile while-loop gated by tt.reduce");
      matched.push_back(w);
    });
    if (doHoist)
      for (scf::WhileOp w : matched)
        (void)hoistRewrite(w);
  }
};

} // namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
