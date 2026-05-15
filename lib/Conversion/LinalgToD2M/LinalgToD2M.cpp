// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/LinalgToD2M/LinalgToD2M.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

namespace mlir::tt {

namespace {

class LinalgToD2MRewriterCommon {
protected:
  LinalgToD2MRewriterCommon(ttcore::MemorySpace defaultInputMemSpace,
                            ttcore::MemorySpace defaultOutputMemSpace)
      : memorySpaces{defaultInputMemSpace, defaultOutputMemSpace} {}

  static bool hasOnlyParallelIterators(linalg::GenericOp op) {
    return llvm::all_of(op.getIteratorTypesArray(),
                        [](utils::IteratorType iteratorType) {
                          return iteratorType == utils::IteratorType::parallel;
                        });
  }

  static bool hasTensorSemantics(linalg::GenericOp op) {
    if (op.getNumDpsInits() != op->getNumResults()) {
      return false;
    }

    return llvm::all_of(op->getOperands(), [](Value value) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(value.getType());
      return tensorType && tensorType.getRank() >= 2;
    });
  }

  static bool hasSupportedIndexingMaps(linalg::GenericOp op) {
    unsigned expectedMaps = op.getNumDpsInputs() + op.getNumDpsInits();
    if (op.getIndexingMapsArray().size() != expectedMaps) {
      return false;
    }

    unsigned iteratorRank = op.getNumLoops();
    return llvm::all_of(op.getIndexingMapsArray(), [&](AffineMap map) {
      return map.getNumDims() == iteratorRank && map.getNumSymbols() == 0;
    });
  }

  static bool isTileableScalarType(Type type) {
    return mlir::isa<ttcore::TileType, FloatType, IntegerType>(type);
  }

  static bool isSupportedBodyOperation(Operation *op) {
    bool supportedOp =
        mlir::isa<arith::AddFOp, arith::CmpFOp, arith::ConstantOp,
                  arith::DivFOp, arith::MaximumFOp, arith::MinimumFOp,
                  arith::MulFOp, arith::NegFOp, arith::SubFOp, math::AbsFOp,
                  math::CosOp, math::ErfOp, math::ExpOp, math::LogOp,
                  math::Log1pOp, math::RsqrtOp, math::SinOp, math::SqrtOp,
                  math::TanhOp>(op);
    if (!supportedOp) {
      return false;
    }

    return llvm::all_of(op->getResults(), [](Value result) {
      return isTileableScalarType(result.getType());
    });
  }

  static bool hasSupportedPayload(linalg::GenericOp op) {
    Block *body = op.getBody();
    if (!body) {
      return false;
    }

    bool foundYield = false;
    for (Operation &bodyOp : body->getOperations()) {
      if (mlir::isa<linalg::YieldOp>(bodyOp)) {
        foundYield = true;
        continue;
      }
      if (!isSupportedBodyOperation(&bodyOp)) {
        return false;
      }
    }

    return foundYield;
  }

  static Type getTileType(Type type) {
    if (mlir::isa<ttcore::TileType>(type)) {
      return type;
    }

    if (!isTileableScalarType(type)) {
      return {};
    }

    return ttcore::TileType::get(type, ttcore::TileType::getDefaultShape());
  }

  static FailureOr<RankedTensorType> getShardType(Type type) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
    if (!tensorType) {
      return failure();
    }

    auto layout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
            tensorType.getEncoding());
    if (!layout) {
      return failure();
    }

    return RankedTensorType::get(layout.getShardShape(tensorType),
                                 tensorType.getElementType());
  }

  static SmallVector<int64_t> getUnitGrid(std::size_t rank) {
    return SmallVector<int64_t>(rank, 1);
  }

  static ttcore::MetalLayoutAttr
  getUncollapsedMetalLayout(MLIRContext *ctx, ArrayRef<int64_t> logicalShape,
                            ttcore::MemorySpace memSpace) {
    auto emptyIntervalType =
        RankedTensorType::get({0, 2}, IntegerType::get(ctx, 64));
    auto emptyCollapseIntervals =
        DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});
    return ttcore::MetalLayoutAttr::get(
        ctx, logicalShape, memSpace, ttcore::TensorMemoryLayout::Sharded,
        emptyCollapseIntervals);
  }

  Value createTiledLayoutOp(Value value, ttcore::MemorySpace memSpace,
                            ConversionPatternRewriter &rewriter) const {
    auto tensorType = mlir::cast<RankedTensorType>(value.getType());
    if (mlir::isa_and_nonnull<ttcore::MetalLayoutAttr>(
            tensorType.getEncoding())) {
      return value;
    }

    Type elementType = ttcore::TileType::get(
        tensorType.getElementType(), ttcore::TileType::getDefaultShape());
    ttcore::MetalLayoutAttr layout = getUncollapsedMetalLayout(
        rewriter.getContext(), tensorType.getShape(), memSpace);
    constexpr std::array<int64_t, 2> defaultTileShape =
        ttcore::TileType::getDefaultShape();
    SmallVector<int64_t> tileShape(defaultTileShape.begin(),
                                   defaultTileShape.end());
    SmallVector<int64_t> physicalShape = layout.getPhysicalShape(tileShape);
    SmallVector<int64_t> unitGrid = getUnitGrid(physicalShape.size());
    SmallVector<int64_t> deviceShape =
        layout.getDeviceShape(unitGrid, tileShape);

    auto emptyOp = rewriter.create<d2m::EmptyOp>(
        value.getLoc(), deviceShape, elementType, layout);
    if (tensorType.getRank() > 2) {
      auto [forwardMap, inverseMap] =
          ttmlir::d2m::utils::grids::createCoreVirtMaps(
              rewriter.getContext(), unitGrid, {1, 1});
      emptyOp.setVirtualGridInverseMappingAttr(AffineMapAttr::get(inverseMap));
      emptyOp.setVirtualGridForwardMappingAttr(AffineMapAttr::get(forwardMap));
    }

    return rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp)
        .getResult(0);
  }

  Value createHostLayoutOp(Value value, RankedTensorType resultType,
                           ConversionPatternRewriter &rewriter) const {
    auto emptyOp = rewriter.create<d2m::EmptyOp>(
        value.getLoc(), resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());
    return rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp)
        .getResult(0);
  }

  std::array<SmallVector<Value>, 2>
  toTiledLayouts(linalg::GenericOp op,
                 linalg::GenericOp::Adaptor adaptor,
                 ConversionPatternRewriter &rewriter) const {
    std::array<SmallVector<Value>, 2> result;
    result[0].reserve(op.getNumDpsInputs());
    result[1].reserve(op.getNumDpsInits());

    for (Value input : adaptor.getInputs()) {
      result[0].push_back(
          createTiledLayoutOp(input, memorySpaces[0], rewriter));
    }

    for (Value output : adaptor.getOutputs()) {
      result[1].push_back(
          createTiledLayoutOp(output, memorySpaces[1], rewriter));
    }

    return result;
  }

  static SmallVector<Attribute>
  getD2MIteratorTypes(OpBuilder &builder, unsigned rank) {
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    return SmallVector<Attribute>(rank, parallel);
  }

  static SmallVector<Value> createRemoteLoadsAndOutputShards(
      Location loc, d2m::GenericOp generic, TypeRange inputs,
      TypeRange outputs, ConversionPatternRewriter &rewriter) {
    SmallVector<Value> blockValues;
    blockValues.reserve(inputs.size() + outputs.size());

    for (auto [index, inputType] : llvm::enumerate(inputs)) {
      FailureOr<RankedTensorType> shardType = getShardType(inputType);
      TT_assert(succeeded(shardType));
      Value buffer = rewriter.create<tensor::EmptyOp>(
          loc, shardType->getShape(), shardType->getElementType());
      SmallVector<Value> indices =
          d2m::utils::buildGridIndices(rewriter, loc,
                                       generic.getIndexingMap(index));
      Value loaded = rewriter
                         .create<d2m::RemoteLoadOp>(
                             loc, *shardType, buffer,
                             generic->getOperand(index), indices)
                         .getResult();
      blockValues.push_back(loaded);
    }

    for (Type outputType : outputs) {
      FailureOr<RankedTensorType> shardType = getShardType(outputType);
      TT_assert(succeeded(shardType));
      blockValues.push_back(rewriter.create<tensor::EmptyOp>(
          loc, shardType->getShape(), shardType->getElementType()));
    }

    return blockValues;
  }

  static FailureOr<Value> lookupMappedValue(Value value, IRMapping &mapping) {
    if (Value mapped = mapping.lookupOrNull(value)) {
      return mapped;
    }
    return failure();
  }

  template <typename TileOp>
  static LogicalResult convertUnaryOp(Operation *op, IRMapping &mapping,
                                      ConversionPatternRewriter &rewriter) {
    FailureOr<Value> operand = lookupMappedValue(op->getOperand(0), mapping);
    if (failed(operand)) {
      return failure();
    }

    Type resultType = getTileType(op->getResult(0).getType());
    if (!resultType) {
      return failure();
    }

    Value result =
        rewriter.create<TileOp>(op->getLoc(), resultType, *operand).getResult();
    mapping.map(op->getResult(0), result);
    return success();
  }

  template <typename TileOp>
  static LogicalResult convertBinaryOp(Operation *op, IRMapping &mapping,
                                       ConversionPatternRewriter &rewriter) {
    FailureOr<Value> lhs = lookupMappedValue(op->getOperand(0), mapping);
    FailureOr<Value> rhs = lookupMappedValue(op->getOperand(1), mapping);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }

    Type resultType = getTileType(op->getResult(0).getType());
    if (!resultType) {
      return failure();
    }

    Value result =
        rewriter.create<TileOp>(op->getLoc(), resultType, *lhs, *rhs)
            .getResult();
    mapping.map(op->getResult(0), result);
    return success();
  }

  static LogicalResult convertConstantOp(arith::ConstantOp op,
                                         IRMapping &mapping,
                                         ConversionPatternRewriter &rewriter) {
    Type tileType = getTileType(op.getType());
    if (!tileType) {
      return failure();
    }

    Value scalar = rewriter.create<arith::ConstantOp>(
        op.getLoc(), op.getType(), op.getValue());
    Value tile =
        rewriter.create<d2m::TileFillOp>(op.getLoc(), tileType, scalar)
            .getResult();
    mapping.map(op.getResult(), tile);
    return success();
  }

  static LogicalResult convertCmpFOp(arith::CmpFOp op, IRMapping &mapping,
                                     ConversionPatternRewriter &rewriter) {
    FailureOr<Value> lhs = lookupMappedValue(op.getLhs(), mapping);
    FailureOr<Value> rhs = lookupMappedValue(op.getRhs(), mapping);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }

    auto lhsTileType = mlir::cast<ttcore::TileType>((*lhs).getType());
    Type boolTileType = ttcore::TileType::get(op.getContext(),
                                              lhsTileType.getShape(),
                                              ttcore::DataType::Bool);

    Value diff =
        rewriter.create<d2m::TileSubOp>(op.getLoc(), (*lhs).getType(), *lhs,
                                        *rhs)
            .getResult();

    Value result;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      result =
          rewriter.create<d2m::TileEqzOp>(op.getLoc(), boolTileType, diff);
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      result =
          rewriter.create<d2m::TileNezOp>(op.getLoc(), boolTileType, diff);
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      result =
          rewriter.create<d2m::TileGtzOp>(op.getLoc(), boolTileType, diff);
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      result =
          rewriter.create<d2m::TileGezOp>(op.getLoc(), boolTileType, diff);
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      result =
          rewriter.create<d2m::TileLtzOp>(op.getLoc(), boolTileType, diff);
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      result =
          rewriter.create<d2m::TileLezOp>(op.getLoc(), boolTileType, diff);
      break;
    default:
      return failure();
    }

    mapping.map(op.getResult(), result);
    return success();
  }

  static LogicalResult convertBodyOp(Operation *op, IRMapping &mapping,
                                     ConversionPatternRewriter &rewriter) {
    TT_assert(isSupportedBodyOperation(op));
    if (auto constantOp = mlir::dyn_cast<arith::ConstantOp>(op)) {
      return convertConstantOp(constantOp, mapping, rewriter);
    }
    if (auto cmpOp = mlir::dyn_cast<arith::CmpFOp>(op)) {
      return convertCmpFOp(cmpOp, mapping, rewriter);
    }
    if (mlir::isa<arith::AddFOp>(op)) {
      return convertBinaryOp<d2m::TileAddOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::SubFOp>(op)) {
      return convertBinaryOp<d2m::TileSubOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::MulFOp>(op)) {
      return convertBinaryOp<d2m::TileMulOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::DivFOp>(op)) {
      return convertBinaryOp<d2m::TileDivOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::MaximumFOp>(op)) {
      return convertBinaryOp<d2m::TileMaximumOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::MinimumFOp>(op)) {
      return convertBinaryOp<d2m::TileMinimumOp>(op, mapping, rewriter);
    }
    if (mlir::isa<arith::NegFOp>(op)) {
      return convertUnaryOp<d2m::TileNegativeOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::AbsFOp>(op)) {
      return convertUnaryOp<d2m::TileAbsOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::CosOp>(op)) {
      return convertUnaryOp<d2m::TileCosOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::ErfOp>(op)) {
      return convertUnaryOp<d2m::TileErfOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::ExpOp>(op)) {
      return convertUnaryOp<d2m::TileExpOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::LogOp>(op)) {
      return convertUnaryOp<d2m::TileLogOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::Log1pOp>(op)) {
      return convertUnaryOp<d2m::TileLog1pOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::RsqrtOp>(op)) {
      return convertUnaryOp<d2m::TileRsqrtOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::SinOp>(op)) {
      return convertUnaryOp<d2m::TileSinOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::SqrtOp>(op)) {
      return convertUnaryOp<d2m::TileSqrtOp>(op, mapping, rewriter);
    }
    if (mlir::isa<math::TanhOp>(op)) {
      return convertUnaryOp<d2m::TileTanhOp>(op, mapping, rewriter);
    }

    return failure();
  }

  static LogicalResult
  convertLinalgPayload(linalg::GenericOp src, linalg::GenericOp dst,
                       ConversionPatternRewriter &rewriter) {
    Block *srcBlock = src.getBody();
    Region &dstRegion = dst.getRegion();
    Block *dstBlock = rewriter.createBlock(&dstRegion);

    for (Value input : dst.getInputs()) {
      auto tensorType = mlir::cast<RankedTensorType>(input.getType());
      dstBlock->addArgument(tensorType.getElementType(), src.getLoc());
    }
    for (Value output : dst.getOutputs()) {
      auto tensorType = mlir::cast<RankedTensorType>(output.getType());
      dstBlock->addArgument(tensorType.getElementType(), src.getLoc());
    }

    IRMapping mapping;
    for (auto [oldArg, newArg] :
         llvm::zip(srcBlock->getArguments(), dstBlock->getArguments())) {
      mapping.map(oldArg, newArg);
    }

    rewriter.setInsertionPointToStart(dstBlock);
    for (Operation &bodyOp : srcBlock->getOperations()) {
      if (auto yieldOp = mlir::dyn_cast<linalg::YieldOp>(bodyOp)) {
        SmallVector<Value> yieldValues;
        yieldValues.reserve(yieldOp->getNumOperands());
        for (Value operand : yieldOp.getValues()) {
          FailureOr<Value> mapped = lookupMappedValue(operand, mapping);
          if (failed(mapped)) {
            return rewriter.notifyMatchFailure(
                src, "linalg.yield operand was not produced by a supported op");
          }
          yieldValues.push_back(*mapped);
        }
        rewriter.create<linalg::YieldOp>(yieldOp.getLoc(), yieldValues);
        return success();
      }

      if (failed(convertBodyOp(&bodyOp, mapping, rewriter))) {
        return rewriter.notifyMatchFailure(
            src, "unsupported linalg.generic body operation");
      }
    }

    return rewriter.notifyMatchFailure(src, "missing linalg.yield terminator");
  }

  std::array<ttcore::MemorySpace, 2> memorySpaces;
};

class LinalgGenericToD2MRewriter final
    : public OpConversionPattern<linalg::GenericOp>,
      LinalgToD2MRewriterCommon {
public:
  LinalgGenericToD2MRewriter(const TypeConverter &typeConverter,
                             MLIRContext *ctx,
                             ttcore::MemorySpace defaultInputMemSpace,
                             ttcore::MemorySpace defaultOutputMemSpace)
      : OpConversionPattern<linalg::GenericOp>(typeConverter, ctx),
        LinalgToD2MRewriterCommon(defaultInputMemSpace,
                                  defaultOutputMemSpace) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op->getParentOfType<d2m::GenericOp>()) {
      return rewriter.notifyMatchFailure(op,
                                         "already nested inside d2m.generic");
    }

    if (!hasTensorSemantics(op)) {
      return rewriter.notifyMatchFailure(
          op, "only tensor linalg.generic ops are supported");
    }

    if (!hasOnlyParallelIterators(op)) {
      return rewriter.notifyMatchFailure(
          op, "only all-parallel linalg.generic ops are supported");
    }

    if (!hasSupportedIndexingMaps(op)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.generic indexing maps");
    }

    if (!hasSupportedPayload(op)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.generic body operation");
    }

    Location loc = op.getLoc();
    auto [inputs, outputs] = toTiledLayouts(op, adaptor, rewriter);

    unsigned iteratorRank = op.getNumLoops();
    SmallVector<Attribute> iteratorTypes =
        getD2MIteratorTypes(rewriter, iteratorRank);
    auto d2mGeneric = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(op.getIndexingMapsArray()),
        rewriter.getArrayAttr(iteratorTypes));

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(d2mGeneric);
    {
      Region &region = d2mGeneric->getRegions().front();
      Block *block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);

      SmallVector<Value> blockValues = createRemoteLoadsAndOutputShards(
          loc, d2mGeneric, TypeRange(inputs), TypeRange(outputs), rewriter);
      auto numInputs = inputs.size();
      auto numOutputs = outputs.size();

      auto innerGeneric = rewriter.create<linalg::GenericOp>(
          loc,
          llvm::to_vector(
              ValueRange(blockValues).take_back(numOutputs).getTypes()),
          ValueRange(blockValues).take_front(numInputs),
          ValueRange(blockValues).take_back(numOutputs),
          op.getIndexingMapsArray(), op.getIteratorTypesArray());

      LogicalResult payloadResult =
          convertLinalgPayload(op, innerGeneric, rewriter);
      TT_assert(succeeded(payloadResult));
      (void)payloadResult;

      rewriter.setInsertionPointAfter(innerGeneric);
      SmallVector<Value> storeResults;
      storeResults.reserve(numOutputs);
      for (std::size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
        std::size_t operandIdx = numInputs + outputIdx;
        SmallVector<Value> indices =
            d2m::utils::buildGridIndices(rewriter, loc,
                                         d2mGeneric.getIndexingMap(operandIdx));
        Value genericOperand = d2mGeneric->getOperand(operandIdx);
        Value result = innerGeneric->getResult(outputIdx);
        storeResults.push_back(
            rewriter
                .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                            genericOperand, indices, result)
                .getResult());
      }
      rewriter.create<d2m::YieldOp>(loc, storeResults);
    }
    rewriter.finalizeOpModification(d2mGeneric);
    rewriter.restoreInsertionPoint(insertPoint);

    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    for (auto [result, originalType] :
         llvm::zip(d2mGeneric->getResults(), op->getResultTypes())) {
      replacements.push_back(createHostLayoutOp(
          result, mlir::cast<RankedTensorType>(originalType), rewriter));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void populateLinalgToD2MPatterns(MLIRContext *ctx,
                                 RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 ttcore::MemorySpace defaultInputMemSpace,
                                 ttcore::MemorySpace defaultOutputMemSpace) {
  patterns.add<LinalgGenericToD2MRewriter>(
      typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace);
}

#define GEN_PASS_DEF_CONVERTLINALGTOD2M
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

class ConvertLinalgToD2MPass final
    : public impl::ConvertLinalgToD2MBase<ConvertLinalgToD2MPass> {
public:
  using Base = impl::ConvertLinalgToD2MBase<ConvertLinalgToD2MPass>;

  ConvertLinalgToD2MPass() = default;

  ConvertLinalgToD2MPass(const ConvertLinalgToD2MOptions &options) : Base() {
    this->defaultInputMemSpace = options.defaultInputMemSpace;
    this->defaultOutputMemSpace = options.defaultOutputMemSpace;
  }

  ConvertLinalgToD2MPass(const ConvertLinalgToD2MPass &rhs) : Base(rhs) {
    this->defaultInputMemSpace = rhs.defaultInputMemSpace;
    this->defaultOutputMemSpace = rhs.defaultOutputMemSpace;
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(ctx);
    populateLinalgToD2MPatterns(ctx, patterns, typeConverter,
                                defaultInputMemSpace, defaultOutputMemSpace);

    ConversionTarget target(*ctx);
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<d2m::D2MDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) {
          return op->getParentOfType<d2m::GenericOp>() != nullptr;
        });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToD2MPass() {
  return std::make_unique<ConvertLinalgToD2MPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgToD2MPass(const ConvertLinalgToD2MOptions &options) {
  return std::make_unique<ConvertLinalgToD2MPass>(options);
}

} // namespace mlir::tt
