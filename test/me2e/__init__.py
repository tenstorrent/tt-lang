# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ME2E test framework for TTL compilation and execution.

Provides class-based test infrastructure for testing TTL dialect operations
through the full pipeline: MLIR generation, compilation, kernel translation,
hardware execution, and golden validation.
"""
