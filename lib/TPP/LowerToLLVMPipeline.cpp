//===- LowerToLLVMPipeline.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LowerToLLVMPipeline pass bundle, which performs
// partial lowering and LLVM dialect conversion. This pass is designed to run
// after bufferization (e.g., after DefaultTppPasses), allowing wrapper passes
// to be inserted between tensor-to-memref conversion and final LLVM lowering.
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"
#include "TPP/PassUtils.h"
#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LOWERTOLLVMPIPELINE
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// The lowering pipeline from memref-level IR to LLVM dialect.
struct LowerToLLVMPipeline
    : public tpp::impl::LowerToLLVMPipelineBase<LowerToLLVMPipeline>,
      PassBundle<ModuleOp> {
  using LowerToLLVMPipelineBase::LowerToLLVMPipelineBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all core MLIR dialects as this pipeline may touch many.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    // Lower TPP-specific dialects (perf, check, xsmm) before partial lowering.
    // This is needed when wrapper passes (e.g., TppRunnerWrapper) are run
    // after DefaultTppPasses but create perf ops that need lowering.
    pm.addPass(createLocalDialectsLowering());

    // Partial Lowering
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createConvertTensorToLinalgPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    if (enableParallel)
      pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createConvertVectorToSCFPass());
    mlir::arith::ArithExpandOpsPassOptions arithExpandOpsOptions;
    arithExpandOpsOptions.includeF8E8M0 = true;
    pm.addPass(arith::createArithExpandOpsPass(arithExpandOpsOptions));
    pm.addPass(createLowerAffinePass());

    // Lower to LLVM
    ConvertVectorToLLVMPassOptions options;
#if defined(__x86_64__)
    options.x86 = true;
#endif
    pm.addPass(createConvertVectorToLLVMPass(options));
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createSCFToControlFlowPass());

    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
    pm.addPass(createGpuToLLVMConversionPass());
    GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
    gpuModuleToBinaryPassOptions.compilationTarget = "fatbin";
    pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass());

    pm.addPass(createConvertFuncToLLVMPass());

    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    if (enableParallel)
      pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createUBToLLVMConversionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    // Anything useful has been lowered by now.
    // Cleanup IR by removing any dead symbols.
    // This step aims to avoid errors caused by frontend leftovers.
    // See issue: #704
    pm.addPass(createSymbolDCEPass());
  }
};

} // namespace
