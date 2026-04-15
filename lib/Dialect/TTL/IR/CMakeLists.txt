add_mlir_dialect_library(MLIRTTLDialect
    TTLDialect.cpp
    TTLOps.cpp
    TTLOpsEnums.cpp
    TTLOpsUtils.cpp
    TTLOpsVerifyUtils.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttlang

    DEPENDS
    MLIRTTLOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIndexingMapOpInterface
    MLIRTTCoreDialect
    MLIRTilingInterface
)
