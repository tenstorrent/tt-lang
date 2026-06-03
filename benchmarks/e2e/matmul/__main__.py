# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``python -m benchmarks.e2e.matmul [--filter ...] [--plot] [--csv ...]``."""

from benchmarks.common import cli
from benchmarks.e2e.matmul.bench import SPEC

if __name__ == "__main__":
    cli(SPEC)
