# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

from ttl.version import __version__

# `ttl._sim_only_marker` is shipped by the tt-lang-sim wheel and absent from
# the tt-lang wheel. Detection is marker-based, not try/except, so a broken
# hardware install still raises ImportError instead of silently degrading.
try:
    import ttl._sim_only_marker  # type: ignore[reportMissingImports] # noqa: F401

    _SIM_ONLY_INSTALL = True
except ImportError:
    _SIM_ONLY_INSTALL = False

if _SIM_ONLY_INSTALL:
    _elementwise_all: list[str] = []
    __all__ = ["__version__"]
else:
    from ttl.ttl import (
        operation,
        compute,
        datamovement,
        Program,
        make_dataflow_buffer_like,
        copy,
        node,
        grid_size,
        math,
    )

    from ttl._generated_elementwise import *  # noqa: F401,F403
    from ttl._generated_elementwise import __all__ as _elementwise_all

    from ttl.operators import signpost
    from ttl.compiler_options import CompilerOptions
    from ttl.ttl_api import (
        CircularBuffer,  # Deprecated, superseded by DataflowBuffer
        DataflowBuffer,
        CopyTransferHandler,
        TensorBlock,
    )
    from ttl.pipe import Pipe, PipeNet

    __all__ = [
        "operation",
        "compute",
        "datamovement",
        "Program",
        "DataflowBuffer",
        "CircularBuffer",
        "CompilerOptions",
        "TensorBlock",
        "CopyTransferHandler",
        "Pipe",
        "PipeNet",
        "make_dataflow_buffer_like",
        "copy",
        "node",
        "grid_size",
        "math",
        "signpost",
        *_elementwise_all,
    ]
