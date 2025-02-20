//===--------------- VectorContractToAMX.cpp ------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction to vector amx.
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "vector-contract-to-amx"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORCONTRACTTOAMX
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {
/// Returns true if the \p map is transposed.
static bool isTransposed(AffineMap map) {
  auto results = map.getResults();
  // Assert if the map does not have 3 or 4 inputs ([] m, n, k).
  assert((map.getNumInputs() == 4 || map.getNumInputs() == 5) &&
         "4 or 5 input dim expected");
  // Assert if the result is not 2D.
  assert(map.getNumResults() == 2 && "Only 2 output dim expected");

  // Check the last two dimensions for transposition.
  auto dimExpr0 = dyn_cast<AffineDimExpr>(results[0]);
  auto dimExpr1 = dyn_cast<AffineDimExpr>(results[1]);
  assert((dimExpr0 && dimExpr1) && "Unexpected dim expression");

  // Exclude output map result.
  bool isOutputResultMap =
      dimExpr0 ==
          mlir::getAffineDimExpr(map.getNumInputs() - 3, map.getContext()) &&
      dimExpr1 ==
          mlir::getAffineDimExpr(map.getNumInputs() - 2, map.getContext());
  assert(!isOutputResultMap && "Output result map not expected");

  // It's transposed if result found as (k, m) or (n, k), else not transposed.
  if ((dimExpr0 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 1, map.getContext()) &&
       dimExpr1 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 3, map.getContext())) ||
      (dimExpr0 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 2, map.getContext()) &&
       dimExpr1 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 1, map.getContext())))
    return true;
  return false;
}
} // namespace

namespace mlir {
namespace tpp {

// Structure to hold transformation context
struct TransformationContext {
  scf::ForOp innerForOp;
  scf::ForOp outerForOp;
  scf::ForOp outermostLoop;
};

enum class MatMulType { Standard, Batch, BatchReduce };

struct VectorContractToAMX
    : public tpp::impl::VectorContractToAMXBase<VectorContractToAMX> {

  using VectorContractToAMXBase::VectorContractToAMXBase;

  void runOnOperation() override;

private:
  TransformationContext ctx;
};

namespace {
// Static function to verify the accumulator and loop structure.
static LogicalResult verifyAccumulatorAndLoopStructure(
    vector::ContractionOp op, mlir::tpp::TransformationContext &ctx, Value &acc,
    vector::TransferReadOp &accDefiningOp) {
  // Verify that the accumulator is coming through a chain of iterargs of
  // nested loop and it is define by 'TransferReadOp'.
  ctx.innerForOp = op->getParentOfType<scf::ForOp>();
  if (!ctx.innerForOp) {
    llvm::errs() << "Inner loop not found\n";
    return failure();
  }

  ctx.outerForOp = ctx.innerForOp->getParentOfType<scf::ForOp>();
  if (ctx.outerForOp) {
    ctx.outermostLoop = ctx.outerForOp->getParentOfType<scf::ForOp>();
    if (!ctx.outermostLoop) {
      llvm::errs() << "Outermost loop not found\n";
      return failure();
    }
  }

  // Verify original inner loop has only one iterarg.
  auto origIterArgs = ctx.innerForOp.getRegionIterArgs();
  if (origIterArgs.size() != 1) {
    llvm::errs() << "Original inner loop does not have exactly one iterarg\n";
    return failure();
  }

  // Verify chain, accumulator must be inner loop's iterarg.
  auto bbArg = dyn_cast<BlockArgument>(acc);
  if (!bbArg) {
    llvm::errs() << "Accumulator is not a BlockArgument\n";
    return failure();
  }

  // This block arg must be init arg, not induction variable.
  if (bbArg.getOwner() != ctx.innerForOp.getBody() ||
      bbArg.getArgNumber() == 0) {
    llvm::errs()
        << "Block argument is not an init arg or is an induction variable\n";
    return failure();
  }

  // This iterarg must be intialized by outer loop's iterarg.
  auto innerInitValue = ctx.innerForOp.getInitArgs()[bbArg.getArgNumber() - 1];
  auto outerBBArg = dyn_cast<BlockArgument>(innerInitValue);
  if (ctx.outerForOp && !outerBBArg) {
    llvm::errs() << "Inner init value is not a BlockArgument\n";
    return failure();
  }

  // This block arg must be init arg, not induction variable.
  if (ctx.outerForOp && (outerBBArg.getOwner() != ctx.outerForOp.getBody() ||
                         outerBBArg.getArgNumber() == 0)) {
    llvm::errs() << "Outer block argument is not an init arg or is an "
                    "induction variable\n";
    return failure();
  }

  // Outer loop's iterarg initializer must be a TransferReadOp.
  acc = ctx.outerForOp
            ? ctx.outerForOp.getInitArgs()[outerBBArg.getArgNumber() - 1]
            : innerInitValue;

  //  This must be defined by vector.transfer_read
  if (!acc.getDefiningOp<vector::TransferReadOp>()) {
    llvm::errs() << "Outer loop's iterarg initializer is not defined by "
                    "TransferReadOp\n";
    return failure();
  }

  accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
  if (!accDefiningOp) {
    llvm::errs() << "Accumulator defining op not found\n";
    return failure();
  }

  // Only 2-D output expected.
  auto accType = cast<ShapedType>(accDefiningOp.getType());
  if (accType.getRank() != 2) {
    llvm::errs() << "Accumulator type is not 2-D\n";
    return failure();
  }

  return success();
}
} // namespace

struct VectorContractToAMXPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  VectorContractToAMXPattern(MLIRContext *context, TransformationContext &ctx)
      : OpRewritePattern<vector::ContractionOp>(context), ctx(ctx) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "Attempting to match and rewrite vector::ContractionOp\n";

    if (op.getKind() != vector::CombiningKind::ADD) {
      llvm::errs()
          << "Unsupported combining kind, only supports ADD at the moment\n";
      return rewriter.notifyMatchFailure(
          op, "Unsupported combining kind, only supports ADD at the moment)");
    }

    auto maskableOp = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskableOp.isMasked()) {
      llvm::errs() << "Masked contractOp not supported\n";
      return rewriter.notifyMatchFailure(op, "Masked contractOp not supported");
    }

    SmallVector<AffineMap, 3> maps = op.getIndexingMapsArray();
    if (llvm::any_of(maps, [](AffineMap map) {
          return !map.isProjectedPermutation();
        })) {
      llvm::errs() << "Unexpected map\n";
      return rewriter.notifyMatchFailure(op, "Unexpected map");
    }

    // Check for the variant of matrix multiply.
    auto iteratorTypes = op.getIteratorTypesArray();
    MatMulType matmulType;
    unsigned outerDimIndex = 0;
    if (iteratorTypes.size() > 3) {
      outerDimIndex = iteratorTypes.size() - 4;
      matmulType =
          iteratorTypes[outerDimIndex] == vector::IteratorType::parallel
              ? MatMulType::Batch
              : MatMulType::BatchReduce;
      outerDimIndex++;
    } else if (iteratorTypes.size() == 3) {
      matmulType = MatMulType::Standard;
    } else {
      llvm::errs() << "Not a gemm\n";
      return rewriter.notifyMatchFailure(op, "Not a gemm");
    }

    if (matmulType == MatMulType::Batch) {
      llvm::errs() << "Batch matmul not supported\n";
      return rewriter.notifyMatchFailure(op, "Batch matmul not supported");
    }
    if (iteratorTypes[outerDimIndex] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 1] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 2] != vector::IteratorType::reduction) {
      llvm::errs() << "Not a gemm\n";
      return rewriter.notifyMatchFailure(op, "Not a gemm");
    }

    SmallVector<Value, 4> results;

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto acc = op.getAcc();
    auto lhsDefiningOp = lhs.getDefiningOp<vector::TransferReadOp>();
    auto rhsDefiningOp = rhs.getDefiningOp<vector::TransferReadOp>();
    auto accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
    if (!lhsDefiningOp || !rhsDefiningOp) {
      llvm::errs() << "LHS or RHS not defined by TransferReadOp\n";
      return failure();
    }

    if (accDefiningOp) {
      llvm::errs() << "Accumulator defined by TransferReadOp\n";
      return failure();
    }

    if (!llvm::all_of(lhsDefiningOp.getIndices(), isZeroIndex) ||
        !llvm::all_of(rhsDefiningOp.getIndices(), isZeroIndex)) {
      llvm::errs() << "Inputs are not whole tensor or subview\n";
      return failure();
    }

    auto lhsType = cast<ShapedType>(lhsDefiningOp.getType());
    auto rhsType = cast<ShapedType>(rhsDefiningOp.getType());
    auto expectedRank = matmulType == MatMulType::BatchReduce ? 4 : 3;
    if (!vnni::utils::isInVnniLayout(expectedRank, lhsType) ||
        !vnni::utils::isInVnniLayout(expectedRank, rhsType)) {
      llvm::errs() << "Expects VNNI layout \n";
      return rewriter.notifyMatchFailure(op, "Expects VNNI layout");
    }

    if (matmulType == MatMulType::BatchReduce &&
        (lhsType.getRank() != 4 || rhsType.getRank() != 4)) {
      llvm::errs() << "BatchReduce matmul type in VNNI with incorrect rank\n";
      return failure();
    }

    if (matmulType == MatMulType::Standard &&
        (lhsType.getRank() != 3 || rhsType.getRank() != 3)) {
      llvm::errs() << "Standard matmul type in VNNI with incorrect rank\n";
      return failure();
    }

    // Check for non-transposed matrices.
    auto mapLHS = maps[0];
    auto mapRHS = maps[1];
    if (matmulType == MatMulType::BatchReduce) {
      mapLHS = mapLHS.dropResult(0);
      mapLHS = mapLHS.dropResult(1);
      mapRHS = mapRHS.dropResult(0);
      mapRHS = mapRHS.dropResult(1);
    }
    llvm::errs() << "mapLHS: " << mapLHS << "\n";
    llvm::errs() << "mapRHS: " << mapRHS << "\n";
    if (isTransposed(mapLHS) || isTransposed(mapRHS)) {
      llvm::errs() << "Transposed matrices are not expected\n";
      return rewriter.notifyMatchFailure(
          op, "Transposed matrices are not expected");
    }

    if (failed(
            verifyAccumulatorAndLoopStructure(op, ctx, acc, accDefiningOp))) {
      llvm::errs() << "Failed to verify accumulator and loop structure\n";
      return failure();
    }

    auto accType = cast<ShapedType>(accDefiningOp.getType());
    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 2);

    auto accSubview = accDefiningOp.getSource();
    Location loc = op.getLoc();

    // Create M different <1xN> subviews.
    SmallVector<OpFoldResult> mixedSizes = {rewriter.getIndexAttr(K),
                                            rewriter.getIndexAttr(N)};
    SmallVector<OpFoldResult> mixedStrides = {rewriter.getIndexAttr(1),
                                              rewriter.getIndexAttr(1)};
    if (ctx.outerForOp) {
      rewriter.setInsertionPoint(
          ctx.outermostLoop.getBody(),
          std::prev(ctx.outermostLoop.getBody()->end(), 1));
    } else {
      rewriter.setInsertionPoint(
          ctx.innerForOp->getBlock(),
          (ctx.innerForOp->getNextNode()->getIterator()));
    }
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Intialize each accumulator with a vector of size N
    SmallVector<Value, 4> initAccs;
    auto amxTile16x32xBf16Ty =
        mlir::amx::TileType::get({16, 32}, rewriter.getBF16Type());
    auto amxTile16x16xF32Ty =
        mlir::amx::TileType::get({16, 16}, rewriter.getF32Type());
    for (auto mIndices = 0; mIndices < M; mIndices += 16) {
      for (auto nIndices = 0; nIndices < N; nIndices += 16) {
        auto acc = rewriter.create<amx::TileLoadOp>(
            loc, amxTile16x16xF32Ty, accSubview,
            ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, mIndices),
                       rewriter.create<arith::ConstantIndexOp>(loc, nIndices)});
        initAccs.push_back(acc);
      }
    }

    // Lamda to create inner loop body when the loop will be nested into other
    // outer SCF::ForOp loop.
    auto innerLoopBodyForNestedLoop = [&](OpBuilder &innerBuilder,
                                             Location loc, Value iv,
                                             Value innerIv,
                                             ValueRange innerIterArgs) {
      llvm::errs() << "Entering innerLoopBodyForNestedLoop\n";
      IRMapping mapping;
      mapping.map(lhsDefiningOp.getSource().getDefiningOp()->getOperand(1), iv);
      llvm::errs() << "Entering innerLoopBodyForNestedLoop 1\n";
      mapping.map(lhsDefiningOp.getSource().getDefiningOp()->getOperand(3),
                  innerIv);
      llvm::errs() << "Entering innerLoopBodyForNestedLoop 2\n";

      auto lhsClone = innerBuilder.clone(
          *lhsDefiningOp.getSource().getDefiningOp(), mapping);
      llvm::errs() << "Cloned lhsClone: " << *lhsClone << "\n";
      // Load and broadcast individual elements
      SmallVector<Value, 4> aLoadTiles;
      for (int i = 0; i < M; i += 16) {
        llvm::errs() << "Loading and broadcasting LHS element at index " << i
                     << "\n";
        auto elem = innerBuilder.create<amx::TileLoadOp>(
            loc, amxTile16x32xBf16Ty, lhsClone->getResult(0),
            ValueRange{c0, innerBuilder.create<arith::ConstantIndexOp>(loc, i),
                       c0, c0});
        aLoadTiles.push_back(elem);
      }

      IRMapping rhsMapping;
      rhsMapping.map(rhsDefiningOp.getSource().getDefiningOp()->getOperand(1),
                     iv);
      rhsMapping.map(rhsDefiningOp.getSource().getDefiningOp()->getOperand(2),
                     innerIv);

      auto rhsClone = innerBuilder.clone(
          *rhsDefiningOp.getSource().getDefiningOp(), rhsMapping);

      SmallVector<Value, 4> bLoadTiles;
      for (int i = 0; i < N; i += 16) {
        llvm::errs() << "Loading and broadcasting RHS element at index " << i
                     << "\n";
        auto loadTile = innerBuilder.create<amx::TileLoadOp>(
            loc, amxTile16x32xBf16Ty, rhsClone->getResult(0),
            ValueRange{c0, c0,
                       innerBuilder.create<arith::ConstantIndexOp>(loc, i),
                       c0});
        bLoadTiles.push_back(loadTile);
      }

      // Create MxN/16x16 different AMXs TileMulFOp.
      int numIterArgs = 0;
      for (int i = 0; i < M / 16; i++) {
        for (int j = 0; j < N / 16; j++) {
          llvm::errs() << "Creating AMX TileMulFOp for indices (" << i << ", "
                       << j << ")\n";
          auto amx = innerBuilder.create<amx::TileMulFOp>(
              loc, amxTile16x16xF32Ty, aLoadTiles[i], bLoadTiles[j],
              innerIterArgs[numIterArgs++]);
          results.push_back(amx);
        }
      }

      // Yield all results
      llvm::errs() << "Yielding results from innerLoopBodyForNestedLoop\n";
      innerBuilder.create<scf::YieldOp>(loc, results);
    };

    // Lamda to create inner loop body when the loop will not be nested into
    // other outer SCF::ForOp loop.
    auto innerLoopBodyForNonNestedLoop = [&](OpBuilder &innerBuilder,
                                             Location loc, Value iv,
                                             Value innerIv,
                                             ValueRange innerIterArgs) {
      llvm::errs() << "Entering innerLoopBodyForNonNestedLoop\n";
      auto temp = lhsDefiningOp.getSource().getDefiningOp();
      llvm::errs() << "lhsDefiningOp.getSource().getDefiningOp(): " << *temp
                   << "\n";
      IRMapping mapping;
      mapping.map(lhsDefiningOp.getSource().getDefiningOp()->getOperand(1),
                  innerIv);
      llvm::errs() << "Entering innerLoopBodyForNonNestedLoop 1\n";

      llvm::errs() << "Entering innerLoopBodyForNonNestedLoop 2\n";

      auto lhsClone = innerBuilder.clone(
          *lhsDefiningOp.getSource().getDefiningOp(), mapping);
      llvm::errs() << "Cloned lhsClone: " << *lhsClone << "\n";
      // Load and broadcast individual elements
      SmallVector<Value, 4> aLoadTiles;
      for (int i = 0; i < M; i += 16) {
        llvm::errs() << "Loading and broadcasting LHS element at index " << i
                     << "\n";
        auto elem = innerBuilder.create<amx::TileLoadOp>(
            loc, amxTile16x32xBf16Ty, lhsClone->getResult(0),
            ValueRange{c0, innerBuilder.create<arith::ConstantIndexOp>(loc, i),
                       c0, c0});
        aLoadTiles.push_back(elem);
      }

      IRMapping rhsMapping;
      rhsMapping.map(rhsDefiningOp.getSource().getDefiningOp()->getOperand(1),
                     innerIv);
      auto rhsClone = innerBuilder.clone(
          *rhsDefiningOp.getSource().getDefiningOp(), rhsMapping);

      SmallVector<Value, 4> bLoadTiles;
      for (int i = 0; i < N; i += 16) {
        llvm::errs() << "Loading and broadcasting RHS element at index " << i
                     << "\n";
        auto loadTile = innerBuilder.create<amx::TileLoadOp>(
            loc, amxTile16x32xBf16Ty, rhsClone->getResult(0),
            ValueRange{c0, c0,
                       innerBuilder.create<arith::ConstantIndexOp>(loc, i),
                       c0});
        bLoadTiles.push_back(loadTile);
      }

      // Create M different AMXs using broadcasts and current accumulator
      // values
      int numIterArgs = 0;
      for (int i = 0; i < M / 16; i++) {
        for (int j = 0; j < N / 16; j++) {
          llvm::errs() << "Creating AMX TileMulFOp for indices (" << i << ", "
                       << j << ")\n";
          auto amx = innerBuilder.create<amx::TileMulFOp>(
              loc, amxTile16x16xF32Ty, aLoadTiles[i], bLoadTiles[j],
              innerIterArgs[numIterArgs++]);
          results.push_back(amx);
        }
      }

      // Yield all M results
      llvm::errs() << "Yielding results from innerLoopLambda\n";
      innerBuilder.create<scf::YieldOp>(loc, results);
    };

    auto createInnerLoop = [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                               ValueRange iterArgs) {
      return nestedBuilder.create<scf::ForOp>(
          loc, ctx.innerForOp.getLowerBound(), ctx.innerForOp.getUpperBound(),
          ctx.innerForOp.getStep(), iterArgs,
          [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
              ValueRange innerIterArgs) {
            innerLoopBodyForNestedLoop(innerBuilder, loc, iv, innerIv, innerIterArgs);
          });
    };

    // Create new outer loop with M different accumulators.
    scf::ForOp newOuterForOp;
    if (ctx.outerForOp) {
      newOuterForOp = rewriter.create<scf::ForOp>(
          loc, ctx.outerForOp.getLowerBound(), ctx.outerForOp.getUpperBound(),
          ctx.outerForOp.getStep(), initAccs,
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            // Use the lambda in the ForOp creation

            auto newInnerForOp =
                createInnerLoop(nestedBuilder, loc, iv, iterArgs);

            // Yield results from inner loop to outer loop
            nestedBuilder.create<scf::YieldOp>(loc, newInnerForOp.getResults());
          });
    } else {
      newOuterForOp = rewriter.create<scf::ForOp>(
          loc, ctx.innerForOp.getLowerBound(), ctx.innerForOp.getUpperBound(),
          ctx.innerForOp.getStep(), initAccs,
          [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
              ValueRange innerIterArgs) {
            innerLoopBodyForNonNestedLoop(innerBuilder, loc, c0, innerIv,
                                          innerIterArgs);
          });
    }

    Value matResult = ctx.outerForOp ? ctx.outerForOp.getResult(0)
                                     : ctx.innerForOp.getResult(0);
    Operation *writeOp;
    for (auto user : matResult.getUsers()) {
      writeOp = dyn_cast<vector::TransferWriteOp>(user);
      if (writeOp)
        break;
    }

    // Store final results back to original locations.
    if (writeOp) {
      int numIterArgs = 0;
      for (auto mIndices = 0; mIndices < M; mIndices += 16) {
        for (auto nIndices = 0; nIndices < N; nIndices += 16) {
          rewriter.create<amx::TileStoreOp>(
              loc, accSubview,
              ValueRange{
                  rewriter.create<arith::ConstantIndexOp>(loc, mIndices),
                  rewriter.create<arith::ConstantIndexOp>(loc, nIndices)},
              newOuterForOp.getResult(numIterArgs++));
        }
      }
    }

    // Erase original write.
    if (writeOp)
      rewriter.eraseOp(writeOp);

    llvm::errs() << "Successfully matched and rewrote vector::ContractionOp\n";
    return success();
  }

private:
  TransformationContext &ctx;
};

void VectorContractToAMX::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<VectorContractToAMXPattern>(context, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace tpp
} // namespace mlir

std::unique_ptr<Pass> createVectorContractToAMX() {
  return std::make_unique<VectorContractToAMX>();
}