# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Centralized list of tests marked as xfail.

Tests are specified using fully qualified pytest names:
- For class-based tests: module_path::ClassName::test_method_name
- For parametrized tests: module_path::ClassName::test_method_name[param_id]

Module path uses dots (e.g., test.me2e.ops.test_binary).
These tests are expected to fail and will be marked with xfail during test collection.
"""

# Dictionary of fully qualified test names to their xfail reasons. Can be empty.
# Format: "module_path::ClassName::test_method_name": "reason" or
#         "module_path::ClassName::test_method_name[param_id]": "reason"
# Example:
#   "test.me2e.ops.test_binary::TestAddFloat32::test_validate_golden": "f32 produces incorrect results (#254)"
XFAIL_TESTS = {
    # SFPU binary add/sub/mul f32: ULP ~2^21 despite fp32_dest_acc_en (#245)
    "test.me2e.ops.test_binary::TestAddFloat32::test_validate_golden": "SFPU binary add f32 precision degraded (#245)",
    "test.me2e.ops.test_binary::TestSubFloat32::test_validate_golden": "SFPU binary sub f32 precision degraded (#245)",
    "test.me2e.ops.test_binary::TestMulFloat32::test_validate_golden": "SFPU binary mul f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-1x1_buf2_interleaved-add]": "SFPU binary add f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-2x2_buf2_interleaved-add]": "SFPU binary add f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-1x1_buf2_interleaved-sub]": "SFPU binary sub f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-2x2_buf2_interleaved-sub]": "SFPU binary sub f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-1x1_buf2_interleaved-mul]": "SFPU binary mul f32 precision degraded (#245)",
    "test.me2e.test_compute_ops::test_compute[float32-2x2_buf2_interleaved-mul]": "SFPU binary mul f32 precision degraded (#245)",
}
