# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compute the complete output on every device using replicated weights."""

from ..__main__ import main

if __name__ == "__main__":
    main(variant="replicated")
