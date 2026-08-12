# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

from ttl.version import __version__, build_info

# `ttl._sim_only_marker` is shipped by the tt-lang-sim wheel and absent from
# the tt-lang wheel. Detection is marker-based, not try/except, so a broken
# hardware install still raises ImportError instead of silently degrading.
# `TTLANG_SIM_ONLY=1` is the source-tree equivalent for environments that run
# the simulator without installing the sim wheel (e.g. the test-sim CI job).
# The env var is only read when the marker is absent; it has no effect on an
# installed sim wheel.
try:
    import ttl._sim_only_marker  # type: ignore[reportMissingImports] # noqa: F401

    _SIM_ONLY_INSTALL = True
except ImportError:
    import os as _os

    _SIM_ONLY_INSTALL = _os.environ.get("TTLANG_SIM_ONLY", "0") == "1"

if _SIM_ONLY_INSTALL:
    _elementwise_all: list[str] = []
    __all__ = ["__version__", "build_info"]
else:
    from ttl.ttl import (
        operation,
        DFB,
        Kernel,
        KernelKind,
        CoreRuntimeArgs,
        KernelDefine,
        KernelRuntimeResources,
        ProgramRuntimeResources,
        DispatchCondition,
        ScalarType,
        compute,
        datamovement,
        make_dataflow_buffer_like,
        make_dfb,
        make_tensor_backed_dfb,
        copy,
        node,
        grid_size,
        math,
        DFBEffect,
        call_extern_func,
        dfb_descriptor,
        get_dfb_id,
        raw_addr,
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
        "__version__",
        "build_info",
        "operation",
        "DFB",
        "Kernel",
        "KernelKind",
        "CoreRuntimeArgs",
        "KernelDefine",
        "KernelRuntimeResources",
        "ProgramRuntimeResources",
        "DispatchCondition",
        "ScalarType",
        "compute",
        "datamovement",
        "DataflowBuffer",
        "CircularBuffer",
        "CompilerOptions",
        "TensorBlock",
        "CopyTransferHandler",
        "Pipe",
        "PipeNet",
        "make_dataflow_buffer_like",
        "make_dfb",
        "make_tensor_backed_dfb",
        "copy",
        "node",
        "grid_size",
        "math",
        "signpost",
        "DFBEffect",
        "call_extern_func",
        "dfb_descriptor",
        "get_dfb_id",
        "raw_addr",
        *_elementwise_all,
    ]
