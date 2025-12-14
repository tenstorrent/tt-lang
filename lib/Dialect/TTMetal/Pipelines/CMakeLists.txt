set(TTLANG_TTMETAL_PIPELINES_LIBS
  MLIRD2MTransforms
  MLIRTTTransforms
  MLIRTTIRPipelines
  MLIRTTIRTransforms
  MLIRTTKernelTransforms
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIREmitCTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

add_mlir_dialect_library(TTLangTTMetalPipelines
  TTMetalPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS
  PRIVATE
  ${TTLANG_TTMETAL_PIPELINES_LIBS}
)
