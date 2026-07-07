# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# The ttcore/ttkernel dialects, the TTKernelToEmitC conversion, and the
# TTKernelToCpp translation are compiled under lib/Dialect, lib/Conversion, and
# lib/Target. This directory only builds the CAPI layer for the Python bindings.
add_subdirectory(CAPI)
