# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Custom LLK header generation that uses correct paths for overlay builds
# The upstream CMakeLists uses CMAKE_SOURCE_DIR which doesn't work in our setup.

include(GenerateRawStringHeader)

# Use ttmlir_SOURCE_DIR instead of CMAKE_SOURCE_DIR
set(LLK_HEADERS
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_tilize_llks.h
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_untilize_llks.h
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_invoke_sfpi_llks.h
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_dataflow_api.h
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_matmul_llks.h
    ${ttmlir_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_coord_translation.h
)

# Set the output directory for generated headers (use the binary dir path that matches include structure)
set(GENERATED_HEADERS_DIR ${ttmlir_BINARY_DIR}/include/ttmlir/Target/TTKernel/LLKs)
file(MAKE_DIRECTORY ${GENERATED_HEADERS_DIR})

# Generate hex header files
set(GENERATED_LLK_HEADERS)
foreach(llk_header ${LLK_HEADERS})
    get_filename_component(header_name ${llk_header} NAME_WE)
    set(output_file ${GENERATED_HEADERS_DIR}/${header_name}_generated.h)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${llk_header}
            -DOUTPUT_FILE=${output_file}
            -DVARIABLE_NAME=${header_name}_generated
            -P ${ttmlir_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake
        DEPENDS ${llk_header}
        COMMENT "Generating header ${header_name}_generated.h"
        VERBATIM
    )
    list(APPEND GENERATED_LLK_HEADERS ${output_file})
endforeach()

# Create a target for the generated headers
add_custom_target(TTKernelGeneratedLLKHeaders DEPENDS ${GENERATED_LLK_HEADERS})
