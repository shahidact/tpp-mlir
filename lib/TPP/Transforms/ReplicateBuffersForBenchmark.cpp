//===------------ ReplicateBuffersForBenchmark.cpp ---------------*- C++-*-===//
//
// This pass replicates buffers for cold-cache benchmarking.
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REPLICATEBUFFERSFORBENCHMARK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

/// Calculate the total size in bytes of a memref type.
static int64_t getMemRefSizeBytes(MemRefType memrefType) {
  if (!memrefType.hasStaticShape())
    return -1;

  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape())
    numElements *= dim;

  unsigned elementBits = memrefType.getElementTypeBitWidth();
  return numElements * (elementBits / 8);
}

// MemRef-based replication using SEPARATE ALLOCATIONS per layer.
struct ReplicateBuffersForBenchmark
    : public tpp::impl::ReplicateBuffersForBenchmarkBase<
          ReplicateBuffersForBenchmark> {
  using ReplicateBuffersForBenchmarkBase::ReplicateBuffersForBenchmarkBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the wrapper function created by TppRunnerWrapper.
    // It uses __wrapper_* global memrefs and calls a renamed kernel (_name).
    func::FuncOp mainFunc = nullptr;
    for (auto func : module.getOps<func::FuncOp>()) {
      // The wrapper function uses global memrefs with __wrapper_ prefix.
      bool hasWrapperGlobals = false;
      func.walk([&](memref::GetGlobalOp getGlobal) {
        if (getGlobal.getName().starts_with("__wrapper_")) {
          hasWrapperGlobals = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      
      if (hasWrapperGlobals) {
        mainFunc = func;
        break;
      }
    }

    if (!mainFunc) {
      // No wrapper function found - pass is a no-op.
      emitError(module.getLoc(),
                "No wrapper function with __wrapper_ globals found");
      return signalPassFailure();
    }

    // Collect global memrefs used in main (TppRunnerWrapper creates these).
    SmallVector<memref::GetGlobalOp> globalGetOps;
    SmallVector<memref::GlobalOp> globalOps;
    DenseMap<StringRef, memref::GlobalOp> globalMap;

    // Build map of global memrefs.
    for (auto globalOp : module.getOps<memref::GlobalOp>()) {
      globalMap[globalOp.getName()] = globalOp;
    }

    // Find all memref.get_global ops in main.
    mainFunc.walk([&](memref::GetGlobalOp getGlobal) {
      auto it = globalMap.find(getGlobal.getName());
      if (it != globalMap.end()) {
        globalGetOps.push_back(getGlobal);
        globalOps.push_back(it->second);
      }
    });

    if (globalGetOps.empty()) {
      // No global memrefs to replicate.
      emitError(module.getLoc(), "No global memrefs found in wrapper function");
      return signalPassFailure();
    }

    // Calculate total buffer size.
    int64_t totalBytesPerLayer = 0;
    int64_t totalBytes = 0;
    for (auto globalOp : globalOps) {
      auto memrefType = cast<MemRefType>(globalOp.getType());
      int64_t size = getMemRefSizeBytes(memrefType);
      if (size < 0) {
        emitError(globalOp.getLoc(),
                  "Cannot compute size for dynamic memref, skipping");
        return signalPassFailure();
      }
      totalBytesPerLayer += size;
    }

    if (totalBytesPerLayer == 0) {
      emitError(module.getLoc(),
                "Total buffer size is zero, nothing to replicate");
      return signalPassFailure();
    }

    // Calculate effective number of layers.
    int64_t effectiveNumLayers = numLayers;
    if (numLayers == -1) {
      int64_t targetBytes =
          static_cast<int64_t>(targetWorkingSetGB * 1024 * 1024 * 1024);
      while (totalBytes < targetBytes) {
        effectiveNumLayers++;
        totalBytes = totalBytesPerLayer * effectiveNumLayers;
      }
    }

    if (effectiveNumLayers <= 1) {
      emitError(
          module.getLoc(),
          "Effective number of layers is 1 or less, no replication needed");
      return signalPassFailure();
    }

    // Find the benchmark perf.bench operation (the one with most iterations).
    // TppRunnerWrapper creates warmup (small iter count) and benchmark (large iter count).
    perf::BenchOp benchmarkOp = nullptr;
    int64_t maxIters = 0;
    mainFunc.walk([&](perf::BenchOp benchOp) {
      // Try to get the iteration count as a constant.
      if (auto numIters = benchOp.getNumIters()) {
        if (auto constOp = numIters.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            int64_t iters = intAttr.getInt();
            if (iters > maxIters) {
              maxIters = iters;
              benchmarkOp = benchOp;
            }
          }
        }
      }
    });

    // Find the kernel call - either inside benchmark perf.bench or standalone.
    func::CallOp kernelCall = nullptr;
    Operation *searchRegion = benchmarkOp ? benchmarkOp.getOperation() : mainFunc.getOperation();
    searchRegion->walk([&](func::CallOp call) {
      // The kernel call is typically not "main"/"entry" and not a built-in.
      StringRef callee = call.getCallee();
      if (callee != "main" && callee != "_mlir_ciface_main" &&
          callee != "entry" && callee.starts_with("_")) {
        // Renamed kernel functions start with "_" (e.g., "_entry").
        if (!kernelCall) {
          kernelCall = call;
        }
      }
    });

    if (!kernelCall) {
      emitWarning(mainFunc.getLoc(), "No kernel call found in main");
      return signalPassFailure();
    }

    // Map from original GetGlobalOp results to their GlobalOp.
    DenseMap<Value, memref::GlobalOp> valueToGlobal;
    for (size_t i = 0; i < globalGetOps.size(); ++i) {
      Value getGlobalResult = globalGetOps[i].getResult();
      valueToGlobal[getGlobalResult] = globalOps[i];
    }

    // Find which kernel arguments come from global memrefs.
    SmallVector<std::pair<unsigned, memref::GlobalOp>> argsFromGlobals;
    for (unsigned i = 0; i < kernelCall.getNumOperands(); ++i) {
      Value arg = kernelCall.getOperand(i);
      auto it = valueToGlobal.find(arg);
      if (it != valueToGlobal.end()) {
        argsFromGlobals.push_back({i, it->second});
      }
    }

    if (argsFromGlobals.empty()) {
      emitWarning(kernelCall.getLoc(),
                  "No kernel arguments sourced from global memrefs");
      return signalPassFailure();
    }

    // Now transform the IR.
    OpBuilder builder(module.getContext());
    Location loc = kernelCall.getLoc();

    // Insert allocations at the beginning.
    builder.setInsertionPointToStart(&mainFunc.getBody().front());

    // Track allocations for cleanup.
    SmallVector<Value> allocsToDealloc;

    // Map from globalOp to vector of allocated memrefs (one per layer).
    DenseMap<memref::GlobalOp, SmallVector<Value>> globalToLayerMemRefs;

    // First pass: create separate allocations for each layer (unrolled).
    Operation *insertBeforeBench =
        benchmarkOp ? benchmarkOp.getOperation() : kernelCall.getOperation();
    builder.setInsertionPoint(insertBeforeBench);

    for (auto &[argIdx, globalOp] : argsFromGlobals) {
      if (globalToLayerMemRefs.count(globalOp))
        continue;

      auto origMemRefType = cast<MemRefType>(globalOp.getType());

      // Allocate each layer separately.
      SmallVector<Value> layerMemRefs;
      for (int64_t layer = 0; layer < effectiveNumLayers; ++layer) {
        auto layerAlloc = memref::AllocOp::create(builder, loc, origMemRefType);
        layerMemRefs.push_back(layerAlloc);
        allocsToDealloc.push_back(layerAlloc);
      }
      globalToLayerMemRefs[globalOp] = std::move(layerMemRefs);
    }

    // Insert unrolled kernel calls INSIDE the benchmark region.
    builder.setInsertionPoint(kernelCall);

    // Unrolled layer calls.
    for (int64_t layer = 0; layer < effectiveNumLayers; ++layer) {
      // Build new operands using the layer's memrefs.
      SmallVector<Value> newOperands;
      for (unsigned i = 0; i < kernelCall.getNumOperands(); ++i) {
        Value origArg = kernelCall.getOperand(i);

        auto globalIt = valueToGlobal.find(origArg);
        if (globalIt == valueToGlobal.end()) {
          newOperands.push_back(origArg);
          continue;
        }
        auto layerMemRefsIt = globalToLayerMemRefs.find(globalIt->second);
        if (layerMemRefsIt == globalToLayerMemRefs.end()) {
          newOperands.push_back(origArg);
          continue;
        }

        // Use this layer's memref directly.
        newOperands.push_back(layerMemRefsIt->second[layer]);
      }

      // Kernel call for this layer.
      func::CallOp::create(builder, loc, kernelCall.getCallee(),
                           kernelCall.getResultTypes(), newOperands);
    }

    // Erase the original kernel call.
    kernelCall.erase();

    // Adjust timing computation: divide by (iterations * numLayers) instead
    // of just iterations.We need to multiply the divisor by numLayers.
    if (benchmarkOp && effectiveNumLayers > 1) {
      Value benchResult = benchmarkOp.getResult(0);

      // Find the divf that uses the bench result.
      for (Operation *user : benchResult.getUsers()) {
        if (auto divOp = dyn_cast<arith::DivFOp>(user)) {
          // Get the divisor (iterations converted to f64).
          Value divisor = divOp.getRhs();

          // Insert multiplication: divisor * numLayers
          builder.setInsertionPoint(divOp);
          auto f64Type = builder.getF64Type();
          auto numLayersF64 = arith::ConstantFloatOp::create(
              builder, loc, f64Type,
              APFloat(static_cast<double>(effectiveNumLayers)));
          auto newDivisor =
              arith::MulFOp::create(builder, loc, divisor, numLayersF64);

          // Update the divf to use the new divisor.
          divOp.setOperand(1, newDivisor);
          break;
        }
      }
    }

    // Add deallocations for memref-based allocations before the return.
    if (!allocsToDealloc.empty()) {
      func::ReturnOp returnOp = nullptr;
      mainFunc.walk([&](func::ReturnOp op) { returnOp = op; });
      if (returnOp) {
        builder.setInsertionPoint(returnOp);
        for (Value alloc : allocsToDealloc) {
          memref::DeallocOp::create(builder, loc, alloc);
        }
      }
    }
  }
};

} // namespace