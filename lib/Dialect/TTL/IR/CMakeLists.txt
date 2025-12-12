add_mlir_dialect_library(MLIRTTLDialect
    TTLDialect.cpp
    TTLOps.cpp
    TTLOpsTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttlang

    DEPENDS
    MLIRTTLOpsIncGen

    LINK_LIBS PUBLIC
    MLIRTTCoreDialect
    )
