// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_LINALGTOD2M_LINALGTOD2M_H
#define TTMLIR_CONVERSION_LINALGTOD2M_LINALGTOD2M_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#define GEN_PASS_DECL_CONVERTLINALGTOD2M
#include "ttmlir/Conversion/Passes.h.inc"

void populateLinalgToD2MPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttcore::MemorySpace defaultInputMemSpace,
    ttcore::MemorySpace defaultOutputMemSpace);

std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToD2MPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgToD2MPass(const ConvertLinalgToD2MOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_LINALGTOD2M_LINALGTOD2M_H
