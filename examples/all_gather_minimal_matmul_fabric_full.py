# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler

"""Compatibility wrapper for the full context all-gather matmul example."""

from all_gather_minimal_matmul.full_context import main


if __name__ == "__main__":
    main()
