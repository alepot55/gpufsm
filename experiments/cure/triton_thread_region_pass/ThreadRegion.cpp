// gpufsm "abstraction regret" research pass — DETECTION half of the tile->thread (thread_region)
// lowering. Finds the lock-step signature of a data-dependent irregular region and marks it.
//
// Signature (see docs/P2_PASS_DESIGN.md, experiments/cure/p2_ttgir_probe.py):
//   scf.while (iter args : tensor<...x..., #blocked>)   // the whole tile is carried
//     before: %r = tt.reduce(<per-lane predicate>) : -> scalar
//             scf.condition(<derived from %r>)          // tile loops to the busiest lane
// That tt.reduce-gated condition over a #blocked tile is the masked-lane waste / issue deficit
// made syntactic — exactly the region the thread_region transform must de-vectorize to recover
// per-lane intra-warp latency hiding. This pass only DETECTS + marks (attribute + remark); the
// lowering is the next step.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-thread-region"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTHREADREGION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// True if `type` is a ranked tensor carrying a TritonGPU #blocked encoding (a tile).
static bool isBlockedTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return false;
  return isa_and_nonnull<BlockedEncodingAttr>(tensor.getEncoding());
}

// True if a tt.reduce feeds `root` through a backward def-use walk bounded to `whileOp`'s
// before-region (the condition computation). This is the "reduce-to-scalar gate".
static bool conditionDerivesFromReduce(Value root, scf::WhileOp whileOp) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  SmallVector<Value, 16> worklist{root};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    // Stay within the before-region of this while loop.
    if (def->getParentRegion() != &whileOp.getBefore())
      continue;
    if (isa<triton::ReduceOp>(def))
      return true;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

struct ThreadRegionPass
    : public impl::TritonGPUThreadRegionBase<ThreadRegionPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int matches = 0;
    mod.walk([&](scf::WhileOp whileOp) {
      // (1) at least one iter-arg is a #blocked tile tensor.
      bool carriesTile = false;
      for (Value init : whileOp.getInits())
        if (isBlockedTile(init.getType())) {
          carriesTile = true;
          break;
        }
      if (!carriesTile)
        return;
      // (2) the loop condition is gated by a tt.reduce to a scalar.
      auto cond = cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
      if (!conditionDerivesFromReduce(cond.getCondition(), whileOp))
        return;
      // Match: mark + remark. (Lowering is the next step.)
      whileOp->setAttr("ttg.thread_region_candidate", UnitAttr::get(&getContext()));
      whileOp->emitRemark("thread_region candidate: lock-step tile while-loop gated by tt.reduce "
                          "(forfeits intra-warp latency hiding; see docs/P2_PASS_DESIGN.md)");
      ++matches;
    });
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] matched " << matches
                            << " thread_region candidate(s)\n");
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
