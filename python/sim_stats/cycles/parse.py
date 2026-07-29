# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace parsing and per-kernel work extraction for cycle estimation."""

from __future__ import annotations

from pathlib import Path

from .types import KernelWork, OpWork, TraceEvent
from ..utils import as_int, iter_events

# Trace events this reader consumes. Defined by the producer in sim/trace.py;
# pinned against its registry by test/sim/test_trace_contract.py.
CONSUMED_EVENTS: frozenset[str] = frozenset({"compute_op", "copy_end"})


def parse_trace(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for obj in iter_events(path):
        tick = as_int(obj.get("tick", 0))
        event = str(obj.get("event", ""))
        kernel_obj = obj.get("kernel")
        kernel = str(kernel_obj) if kernel_obj is not None else None
        data = {
            k: v for k, v in obj.items() if k not in {"tick", "event", "kernel", "node"}
        }

        if "node" in obj:
            data["node"] = obj["node"]

        events.append(TraceEvent(tick=tick, event=event, kernel=kernel, data=data))

    return events


def extract_kernel_work(events: list[TraceEvent]) -> dict[str, KernelWork]:
    """Build per-kernel work records from trace events.

    Reads two event kinds:
      - ``copy_end``   -> movement OpWork, one per non-zero locality tile count.
      - ``compute_op`` -> compute OpWork (op_type, dtype, tiles).

    ``compute_op`` events are emitted by the simulator once per math op. When a
    trace lacks them (e.g. generated before the instrumentation), the compute path
    is simply empty and the estimate is movement-only.
    """
    work: dict[str, KernelWork] = {}

    for ev in events:
        kernel = ev.kernel
        if not kernel:
            continue

        kw = work.get(kernel)
        if kw is None:
            kw = KernelWork(kernel=kernel)
            work[kernel] = kw

        if ev.event == "compute_op":
            tiles = as_int(ev.data.get("tiles", 0))
            if tiles > 0:
                kw.ops.append(
                    OpWork(
                        kind="compute",
                        op_type=str(ev.data.get("op_type", "")),
                        dtype=str(ev.data.get("dtype", "")),
                        tiles=tiles,
                    )
                )
        elif ev.event == "copy_end":
            # One movement op per locality with a non-zero tile count.
            direction = str(ev.data.get("direction", ""))
            for locality in ("local_l1", "remote_l1", "dram"):
                tiles = as_int(ev.data.get(locality, 0))
                if tiles > 0:
                    kw.ops.append(
                        OpWork(
                            kind="movement",
                            op_type="copy",
                            tiles=tiles,
                            locality=locality,
                            direction=direction,
                        )
                    )

    return work


def build_pipeline(path: Path) -> list[KernelWork]:
    """Trace file -> per-kernel work records (parse + extract)."""
    return list(extract_kernel_work(parse_trace(path)).values())
