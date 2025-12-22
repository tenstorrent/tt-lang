# Shared utility code for tt-lang dialects

add_mlir_library(TTLangDialectUtils
  LayoutUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTTNNDialect
)
