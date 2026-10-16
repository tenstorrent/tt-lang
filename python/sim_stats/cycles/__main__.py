# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable entry point for the cycle estimator: ``python -m sim_stats.cycles``.

Backs the bundled ``tt-lang-sim-cycles`` command.
"""

from . import main

if __name__ == "__main__":
    main()
