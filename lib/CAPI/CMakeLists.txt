add_mlir_library(TTLangCAPI
  Dialects.cpp
  TTLAttrs.cpp

  EXCLUDE_FROM_LIBMLIR
  ENABLE_AGGREGATION

  DEPENDS
  MLIRTTLOpsAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRTTLDialect
)
