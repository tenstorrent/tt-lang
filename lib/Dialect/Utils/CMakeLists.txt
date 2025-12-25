add_mlir_dialect_library(TTLangUtils
  LayoutUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTTNNDialect
)
