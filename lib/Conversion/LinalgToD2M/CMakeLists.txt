add_mlir_conversion_library(TTMLIRLinalgToD2M
  LinalgToD2M.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRD2MDialect
  MLIRD2MUtils
  MLIRTTCoreDialect
)
