# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# REQUIRES: ttnn, tt-device, multi-device
# RUN: cd %S/../.. && env TT_METAL_ALLOCATOR_MODE_HYBRID=1 TTLANG_COMPILER_OPTIONS=--ttl-sram-allocation-mode=per-core %python -m pytest %S/fabric/test_ccl.py::test_compiler_l1_point_to_point -xq
# Verify independent receiver addresses and tensor backing across mesh devices.
