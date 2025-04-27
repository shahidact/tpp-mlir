//===- VectorToKernel.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "TPP/PassBundles.h"
#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORTOKERNEL
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "convert-vector-to-kernels"

// Apply collection of vector-level passes that map vector patterns to
// specialized micro-kernels akin to libxsmm kernels.
struct VectorToKernel : public tpp::impl::VectorToKernelBase<VectorToKernel>,
                    PassBundle<ModuleOp> {
  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module))) {
      return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    // TODO: Pass ordering based on target architecture starting from AMX ->
    // avx512 -> avx2 to subset needs to be improved by moving out some logic of
    // Bf16DotProduct related to iterarg creation and let hoistvectorTransfer
    // pass address it.
    pm.addNestedPass<func::FuncOp>(createBF16DotProduct());
    pm.addNestedPass<func::FuncOp>(createHoistVectorTransfers());
    if (vnni::utils::hasAMX())
      pm.addNestedPass<func::FuncOp>(createVectorContractToAMX());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createVectorContractToFMA());
  }
};
