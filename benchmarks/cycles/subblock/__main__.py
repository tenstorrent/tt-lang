# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core subblock cycle benchmark: sweep every subblock shape.

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.subblock
"""

from .sweep import main

if __name__ == "__main__":
    main()
