# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Declarative sweep runner for the microbenchmarks.

A `MicroBenchmark` subclass is class-level constants plus a single `build()` hook.
The base owns argparse, the sweep product, I/O tensor materialization, warmup +
dispatch, the result row, the per-config print line, and CSV writing — none of
which a subclass touches.

Declared per benchmark (uppercase class constants):
  NAME, ZONE, CSV_COLUMNS, DEFAULT_CSV, PARAMS, INPUTS, OUTPUTS, DFBS, STRATEGIES,
  PER_UNIT (cfg key to normalize times by), CSV_TAG, WARMUP.
Implemented per benchmark:
  build(ctx) -> (kernels, ref)   # ctx.tensors are materialized; ctx.torch holds
                                 # the input torch tensors for the reference.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable

import torch
import ttnn

from benchmarks.common import create_benchmark_arg_parser, measure_pcc
from benchmarks.microbench import harness
from benchmarks.microbench.harness import TILE, DTYPES


@dataclass(frozen=True)
class Param:
    name: str
    default: object
    sweep: bool = False        # True -> comma-list, product-swept
    choices: tuple = None
    help: str = ""


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: Callable            # cfg -> (rows, cols) in elements
    init: str = "randn"        # randn | zeros | ones | empty (outputs use empty)
    scale: float = 0.1


@dataclass(frozen=True)
class DFB:
    index: int
    pages: Callable            # cfg -> tile count


class Ctx:
    """Per-(config, strategy) materialized inputs handed to build()."""

    def __init__(self, cfg, strategy, core, grid, tensors, torch_inputs):
        self.cfg = cfg
        self.strategy = strategy
        self.core = core
        self.grid = grid
        self.tensors = tensors          # name -> device tensor
        self.torch = torch_inputs       # name -> input torch tensor


class MicroBenchmark:
    NAME = ""
    ZONE = ""
    CSV_COLUMNS = ()
    DEFAULT_CSV = ""
    PARAMS = ()
    INPUTS = ()
    OUTPUTS = ()
    DFBS = ()
    STRATEGIES = ("",)
    PER_UNIT = None
    CSV_TAG = ()
    WARMUP = 1
    SEED = 2026

    def build(self, ctx):
        """Return (kernels, ref) for this config/strategy."""
        raise NotImplementedError

    def legal(self, cfg, strategy):
        """Whether (cfg, strategy) is runnable (e.g. output fits DST). Default: yes."""
        return True

    def _materialize(self, cfg, device):
        ttnn_dtype, torch_dtype, _ = DTYPES[cfg["dtype"]]
        torch.manual_seed(self.SEED)
        torch_inputs, dev_tensors = {}, {}
        for spec in self.INPUTS:
            shape = spec.shape(cfg)
            if spec.init == "randn":
                tensor = torch.randn(*shape, dtype=torch_dtype) * spec.scale
            elif spec.init == "zeros":
                tensor = torch.zeros(*shape, dtype=torch_dtype)
            elif spec.init == "ones":
                tensor = torch.ones(*shape, dtype=torch_dtype)
            else:
                raise ValueError(f"input {spec.name} has init {spec.init!r}")
            torch_inputs[spec.name] = tensor
            dev_tensors[spec.name] = ttnn.from_torch(
                tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for spec in self.OUTPUTS:
            dev_tensors[spec.name] = ttnn.empty(
                spec.shape(cfg), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return torch_inputs, dev_tensors

    def _row(self, cfg, strategy, zone, pcc):
        row = dict(cfg)
        row["strategy"] = strategy
        zone_summary = harness.zone_fields(zone, pcc)
        row.update(zone_summary)
        if self.PER_UNIT:
            unit = cfg[self.PER_UNIT]
            for key in ("trisc_max_us", "unpack_us", "math_us", "pack_us"):
                value = zone_summary[key]
                row[f"{key}_per_{self.PER_UNIT}"] = None if value is None else value / unit
        return row

    def summary(self, cfg, by_strategy):
        head = " ".join(f"{key}={value}" for key, value in cfg.items())
        present = [strategy for strategy in self.STRATEGIES if strategy in by_strategy]
        if len(present) > 1:
            times = {name: by_strategy[name]["trisc_max_us"] for name in present}
            faster = sorted(present, key=lambda name: (times[name] is None, times[name]))[0]
            shown = " ".join(f"{name}={times[name]}" for name in present)
            pccs = " ".join(f"{name}_pcc={by_strategy[name]['pcc']}" for name in present)
            return f"{head} | {shown} us | faster={faster} | {pccs}"
        row = by_strategy[present[0]]
        return f"{head} | trisc_max={row['trisc_max_us']} us | pcc={row['pcc']}"

    def _run(self, device, cfg, strategy):
        ttnn_dtype, _, dtype_bytes = DTYPES[cfg["dtype"]]
        page_size = dtype_bytes * TILE * TILE
        core, grid = harness.single_core()
        torch_inputs, dev_tensors = self._materialize(cfg, device)
        dfbs = [
            harness.dfb(spec.index, ttnn_dtype, page_size, grid, spec.pages(cfg))
            for spec in self.DFBS
        ]
        kernels, ref = self.build(Ctx(cfg, strategy, core, grid, dev_tensors, torch_inputs))
        io_tensors = [dev_tensors[spec.name] for spec in self.INPUTS]
        io_tensors += [dev_tensors[spec.name] for spec in self.OUTPUTS]
        output, zone = harness.dispatch(device, io_tensors, kernels, dfbs, self.ZONE, self.WARMUP)
        pcc = measure_pcc(ref.float(), ttnn.to_torch(output).float())
        for tensor in io_tensors:
            ttnn.deallocate(tensor)
        return self._row(cfg, strategy, zone, pcc)

    def main(self):
        parser = create_benchmark_arg_parser(self.NAME, default_csv=self.DEFAULT_CSV)
        for param in self.PARAMS:
            flag = f"--{param.name.replace('_', '-')}"
            if isinstance(param.default, bool):
                parser.add_argument(flag, action="store_true")
            elif param.choices:
                parser.add_argument(
                    flag, default=param.default, choices=list(param.choices), help=param.help
                )
            else:
                parser.add_argument(flag, default=param.default, help=param.help)
        args = parser.parse_args()
        if args.compile_only:
            print("compile-only: nothing to execute without a device.")
            return

        def coerce(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return value

        sweep, scalar = {}, {}
        for param in self.PARAMS:
            value = getattr(args, param.name)
            if param.sweep:
                sweep[param.name] = [coerce(item) for item in str(value).split(",")]
            elif isinstance(value, bool):
                scalar[param.name] = value
            else:
                scalar[param.name] = coerce(value)
        sweep_names = list(sweep)

        device = ttnn.open_device(device_id=args.device_id)
        rows = []
        try:
            for combo in product(*sweep.values()):
                cfg = {**scalar, **dict(zip(sweep_names, combo))}
                by_strategy = {
                    strategy: self._run(device, cfg, strategy)
                    for strategy in self.STRATEGIES
                    if self.legal(cfg, strategy)
                }
                if not by_strategy:
                    continue
                rows.extend(by_strategy.values())
                print(self.summary(cfg, by_strategy), flush=True)
        finally:
            ttnn.close_device(device)

        if not args.no_csv and rows:
            arch = rows[0].get("arch", "dev")
            tag = [scalar[key] for key in self.CSV_TAG]
            out_csv = harness.write_csv(rows, args.csv, self.CSV_COLUMNS, arch, *tag)
            print(f"wrote {len(rows)} rows to {out_csv}", flush=True)
