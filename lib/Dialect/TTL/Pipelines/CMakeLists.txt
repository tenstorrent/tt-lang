add_mlir_dialect_library(TTLangTTLPipelines
  TTLPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIREmitCTransforms
  MLIRMemRefTransforms
  TTLangTTLTransforms
  TTMLIRTTKernelToEmitC
)
