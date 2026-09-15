// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===========================================================================//
//                                                                           //
//                       GENERATED FILE -- DO NOT EDIT                       //
//                                                                           //
//===========================================================================//
//
// Emitted by llk-perf/gen_cost_table.py in tt-lang-ops-and-models. Regenerate
// from a checkout of that repository, which needs a device and the tt-llk
// harness:
//   llk-perf/run_sweep.sh      --tt-lang <path to this tt-lang>
//   llk-perf/gen_cost_table.py --tt-lang <path to this tt-lang>
//
// Generated 2026-08-20 23:55:43 UTC, from 11054 CSV rows over 7 benchmarks,
// measured on blackhole; costs are not transferable across architectures. The
// CSVs are in neither tree, so the revisions they were produced against are:
//
//   tt-lang-ops-and-models  0aff7803   the sweeps and this generator
//   tt-lang                 b1725d6c3  the dialect the keys are checked against
//   tt-lang's tt-metal      ea042c4ad  tt-llk and the perf harness
//
// Included into a scope that defines `MeasuredCost`, `EngineCost` and `OpCost`;
// this file is data only. See lib/Analysis/OpCost.cpp. What the table
// answers and what it does not is reported by llk-perf/coverage_report.py.
//
// Each entry is one engine of one benchmark's loop body, attributed to the
// ttkernel operation owning that engine, and excludes the cross-thread credit
// handshakes the scheduler derives instead.
//
// Every number is measured. An engine an operation occupies but nothing timed
// carries an empty span, and a configuration no row matches has no answer, so
// a caller that gets a cost knows a measurement stands behind it. `op` "lump:"
// marks an engine whose loop body ran more than one operation, recorded rather
// than attributed and recoverable by subtracting a separately measured
// component.

// Measured rows, grouped so each (op, engine) run is contiguous. Reached
// only through the engine slot that slices it.
static const MeasuredCost kMeasured[] = {
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.30, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.81, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 42.30, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 42.81, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 42.30, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 42.61, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.73, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 42.73, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 42.70, 0.00},  // abs_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 41.17, 0.00},  // abs_tile/math, n=2, min=41.16, max=41.19
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 41.17, 0.00},  // abs_tile/math, n=2, min=41.16, max=41.19
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 41.22, 0.00},  // abs_tile/math, n=2, min=41.06, max=41.38
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 42.20, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // abs_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // abs_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // abs_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // abs_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // abs_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 57.00, 0.00},  // abs_tile_init/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.48, 0.00},  // abs_tile_int32/math, n=2, min=41.44, max=41.53
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 586.36, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 490.95, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 586.36, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 490.95, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 586.36, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 490.88, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 490.91, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 490.91, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 490.88, 0.00},  // acos_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 489.69, 0.00},  // acos_tile/math, n=2, min=489.62, max=489.75
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 489.69, 0.00},  // acos_tile/math, n=2, min=489.62, max=489.75
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 489.77, 0.00},  // acos_tile/math, n=2, min=489.53, max=490.00
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 586.20, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // acos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // acos_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // acos_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // acos_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // acos_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // acos_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.66, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.62, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // add_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // add_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // add_binary_tile/math, n=2, min=180.50, max=180.66
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // add_binary_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.89, 0.00},  // add_binary_tile/math, n=2, min=370.88, max=370.91
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 47.50, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // add_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // add_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // add_binary_tile_init/math, n=4, min=47.00, max=55.00, median=47.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, min=48.00, max=53.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.41, 0.00},  // add_int_tile/math, n=2, min=41.28, max=41.53
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 61.00, 0.00},  // add_int_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.09, 0.00},  // add_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.09, 0.00},  // add_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.09, 0.00},  // add_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 18.25, 0.00},  // add_tiles/math, n=2, min=18.09, max=18.41
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // add_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.19
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // add_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.19
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 18.94, 0.00},  // add_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 18.19, 0.00},  // add_tiles/math, n=2, min=18.09, max=18.28
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // add_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.19
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 19.03, 0.00},  // add_tiles/math, n=2, min=18.97, max=19.09
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // add_tiles/math, n=2, min=18.12, max=18.19
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 18.94, 0.00},  // add_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 18.19, 0.00},  // add_tiles/math, n=2, min=18.09, max=18.28
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.20, 0.00},  // add_tiles/unpack, n=2, min=37.19, max=37.22
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.20, 0.00},  // add_tiles/unpack, n=2, min=37.19, max=37.22
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.19, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.19, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.16
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.16
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.11, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.11, 0.00},  // add_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.08, 0.00},  // add_tiles/unpack, n=2, min=40.06, max=40.09
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.81, 0.00},  // add_tiles/unpack, n=2, min=37.75, max=37.88
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.08, 0.00},  // add_tiles/unpack, n=2, min=40.06, max=40.09
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.81, 0.00},  // add_tiles/unpack, n=2, min=37.75, max=37.88
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.08, 0.00},  // add_tiles/unpack, n=2, min=40.06, max=40.09
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.75, 0.00},  // add_tiles/unpack, n=2, min=37.69, max=37.81
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 578.45, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.94, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 578.45, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 466.94, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 578.45, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 466.78, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 578.27, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.84, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 578.27, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 466.84, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 578.27, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 466.80, 0.00},  // asin_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 465.61, 0.00},  // asin_tile/math, n=2, min=465.53, max=465.69
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 578.27, 0.00},  // asin_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 465.66, 0.00},  // asin_tile/math, n=2, min=465.53, max=465.78
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 578.27, 0.00},  // asin_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 465.47, 0.00},  // asin_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 578.36, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // asin_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.98, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1873.41, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 1264.98, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1873.41, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 1264.98, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1873.50, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.84, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1873.38, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 1264.97, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1873.38, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 1264.84, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1873.28, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1874.69, 0.00},  // atan2_binary_tile/math, n=2, min=1874.56, max=1874.81
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.94, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1874.69, 0.00},  // atan2_binary_tile/math, n=2, min=1874.56, max=1874.81
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 1264.84, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1874.56, 0.00},  // atan2_binary_tile/math, n=2, min=1874.44, max=1874.69
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 1264.84, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // atan2_binary_tile_init/math, n=2, min=49.00, max=56.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.50, 0.00},  // atan2_binary_tile_init/math, n=4, min=49.00, max=51.00, median=49.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.50, 0.00},  // atan2_binary_tile_init/math, n=4, min=49.00, max=51.00, median=49.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 50.75, 0.00},  // atan2_binary_tile_init/math, n=4, min=49.00, max=56.00, median=49.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 338.69, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 338.69, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 338.41, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 338.67, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 338.67, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 338.44, 0.00},  // atan_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 337.78, 0.00},  // atan_tile/math, n=2, min=337.62, max=337.94
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 337.78, 0.00},  // atan_tile/math, n=2, min=337.62, max=337.94
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 337.59, 0.00},  // atan_tile/math, n=2, min=337.50, max=337.69
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 426.47, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // atan_tile_init/math, n=2, min=65.00, max=66.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // atan_tile_init/math, n=2, min=65.00, max=66.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // atan_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.50, 0.00},  // atan_tile_init/math, n=2, min=63.00, max=66.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // atan_tile_init/math, n=4, min=62.00, max=66.00, median=64.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 47.50, 0.00},  // binary_bitwise_tile_init/math, n=6, min=47.00, max=48.00, median=47.50
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 338.94, 0.00},  // binary_left_shift_tile/math, n=2, min=338.88, max=339.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.86, 0.00},  // binary_logical_right_shift_tile/math, n=2, min=370.84, max=370.88
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 238.48, 0.00},  // binary_max_int32_tile/math, n=2, min=238.41, max=238.56
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 73.50, 0.00},  // binary_max_int32_tile_init/math, n=2, min=71.00, max=76.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.00, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 124.14, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 124.08, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.19, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 124.19, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 124.00, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_max_tile/math, n=2, min=139.19, max=139.31
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_max_tile/math, n=2, min=139.19, max=139.31
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 139.31, 0.00},  // binary_max_tile/math, n=2, min=139.25, max=139.38
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=67.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=67.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2, min=65.00, max=67.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=4, min=65.00, max=67.00, median=66.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=4, min=65.00, max=67.00, median=66.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.25, 0.00},  // binary_max_tile_init/math, n=4, min=65.00, max=67.00, median=66.50
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 238.48, 0.00},  // binary_min_int32_tile/math, n=2, min=238.41, max=238.56
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 73.50, 0.00},  // binary_min_int32_tile_init/math, n=2, min=71.00, max=76.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.14, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 124.14, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 124.08, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.19, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 124.19, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 124.00, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_min_tile/math, n=2, min=139.19, max=139.31
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.25, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_min_tile/math, n=2, min=139.19, max=139.31
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.20, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 139.31, 0.00},  // binary_min_tile/math, n=2, min=139.25, max=139.38
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.17, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=67.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=67.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=67.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=4, min=65.00, max=67.00, median=66.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=4, min=65.00, max=67.00, median=66.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2, min=65.00, max=67.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 65.25, 0.00},  // binary_min_tile_init/math, n=4, min=63.00, max=67.00, median=65.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 91.25, 0.00},  // binary_op_init_common/math, n=4, min=90.00, max=95.00, median=90.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 93.25, 0.00},  // binary_op_init_common/math, n=4, min=91.00, max=94.00, median=94.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 91.25, 0.00},  // binary_op_init_common/math, n=4, min=90.00, max=95.00, median=90.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 93.25, 0.00},  // binary_op_init_common/math, n=4, min=91.00, max=94.00, median=94.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 91.25, 0.00},  // binary_op_init_common/math, n=4, min=90.00, max=95.00, median=90.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 95.00, 0.00},  // binary_op_init_common/math, n=4
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 91.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 94.83, 0.00},  // binary_op_init_common/math, n=6, min=94.00, max=95.00, median=95.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 91.00, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 94.83, 0.00},  // binary_op_init_common/math, n=6, min=94.00, max=95.00, median=95.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 91.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 93.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=94.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 91.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 94.83, 0.00},  // binary_op_init_common/math, n=6, min=94.00, max=95.00, median=95.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 91.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 94.83, 0.00},  // binary_op_init_common/math, n=6, min=94.00, max=95.00, median=95.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 91.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=90.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 93.17, 0.00},  // binary_op_init_common/math, n=6, min=89.00, max=94.00, median=94.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 202.50, 0.00},  // binary_op_init_common/pack, n=8, min=200.00, max=205.00, median=202.50
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 204.50, 0.00},  // binary_op_init_common/pack, n=8, min=198.00, max=209.00, median=204.50
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 202.75, 0.00},  // binary_op_init_common/pack, n=8, min=200.00, max=205.00, median=202.50
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 204.50, 0.00},  // binary_op_init_common/pack, n=8, min=198.00, max=209.00, median=204.50
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 202.62, 0.00},  // binary_op_init_common/pack, n=8, min=201.00, max=205.00, median=202.50
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 201.75, 0.00},  // binary_op_init_common/pack, n=8, min=197.00, max=208.00, median=202.50
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 204.17, 0.00},  // binary_op_init_common/pack, n=12, min=201.00, max=209.00, median=204.50
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 204.17, 0.00},  // binary_op_init_common/pack, n=12, min=201.00, max=209.00, median=204.50
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 202.17, 0.00},  // binary_op_init_common/pack, n=12, min=198.00, max=208.00, median=201.50
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 204.17, 0.00},  // binary_op_init_common/pack, n=12, min=201.00, max=209.00, median=204.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 204.17, 0.00},  // binary_op_init_common/pack, n=12, min=201.00, max=209.00, median=204.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 206.33, 0.00},  // binary_op_init_common/pack, n=12, min=203.00, max=208.00, median=206.50
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 202.17, 0.00},  // binary_op_init_common/pack, n=12, min=198.00, max=208.00, median=201.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 171.25, 0.00},  // binary_op_init_common/unpack, n=4, min=171.00, max=172.00, median=171.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 168.75, 0.00},  // binary_op_init_common/unpack, n=4, min=168.00, max=171.00, median=168.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 171.25, 0.00},  // binary_op_init_common/unpack, n=4, min=171.00, max=172.00, median=171.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 168.75, 0.00},  // binary_op_init_common/unpack, n=4, min=168.00, max=171.00, median=168.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 171.25, 0.00},  // binary_op_init_common/unpack, n=4, min=171.00, max=172.00, median=171.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 171.25, 0.00},  // binary_op_init_common/unpack, n=4, min=171.00, max=172.00, median=171.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 170.00, 0.00},  // binary_op_init_common/unpack, n=6, min=169.00, max=172.00, median=169.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 170.00, 0.00},  // binary_op_init_common/unpack, n=6, min=169.00, max=172.00, median=169.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 170.00, 0.00},  // binary_op_init_common/unpack, n=6, min=169.00, max=172.00, median=169.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 170.00, 0.00},  // binary_op_init_common/unpack, n=6, min=169.00, max=172.00, median=169.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 172.33, 0.00},  // binary_op_init_common/unpack, n=6, min=172.00, max=173.00, median=172.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 595.03, 0.00},  // binary_right_shift_tile/math, n=2, min=594.97, max=595.09
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 48.50, 0.00},  // binary_shift_tile_init/math, n=6, min=47.00, max=53.00, median=48.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_and_binary_tile/math, n=2, min=180.72, max=180.94
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 48.88, 0.00},  // bitwise_not_tile/math, n=2, min=48.78, max=48.97
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 57.00, 0.00},  // bitwise_not_tile_init/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_or_binary_tile/math, n=2, min=180.72, max=180.94
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_xor_binary_tile/math, n=2, min=180.72, max=180.94
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.03, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.58, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.03, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.58, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.03, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.41, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.41, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.41, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.48, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // ceil_tile/math, n=2, min=122.25, max=122.59
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // ceil_tile/math, n=2, min=122.25, max=122.59
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 122.53, 0.00},  // ceil_tile/math, n=2, min=122.44, max=122.62
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 122.86, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.30, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.77, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.31, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.77, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.30, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.70, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.86, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.86, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.77, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 121.38, 0.00},  // clamp_tile/math, n=2, min=121.19, max=121.56
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 121.38, 0.00},  // clamp_tile/math, n=2, min=121.19, max=121.56
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 121.39, 0.00},  // clamp_tile/math, n=2, min=121.38, max=121.41
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // clamp_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // clamp_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.50, 0.00},  // clamp_tile_init/math, n=2, min=51.00, max=52.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 58.00, 0.00},  // clamp_tile_init/math, n=2, min=57.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // clamp_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // clamp_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=4, min=51.00, max=59.00, median=54.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 93.23, 0.00},  // compute_kernel_hw_startup/math, n=13, min=89.00, max=96.00, median=94.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 94.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 93.08, 0.00},  // compute_kernel_hw_startup/math, n=13, min=89.00, max=96.00, median=94.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 94.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 93.23, 0.00},  // compute_kernel_hw_startup/math, n=13, min=89.00, max=96.00, median=94.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 91.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 93.12, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=94.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 93.28, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=95.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 93.12, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=94.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 91.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 93.12, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=94.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 95.50, 0.00},  // compute_kernel_hw_startup/math, n=2, min=95.00, max=96.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 93.12, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=94.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 95.50, 0.00},  // compute_kernel_hw_startup/math, n=2, min=95.00, max=96.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 93.12, 0.00},  // compute_kernel_hw_startup/math, n=25, min=89.00, max=97.00, median=94.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 92.00, 0.00},  // compute_kernel_hw_startup/math, n=2, min=91.00, max=93.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 208.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 209.00, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=208.00, max=210.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 208.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 209.00, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=208.00, max=210.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 208.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 205.00, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=202.00, max=208.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 207.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=207.00, max=208.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 207.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=207.00, max=208.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 204.00, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=200.00, max=208.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 205.00, 0.00},  // compute_kernel_hw_startup/pack, n=4, min=201.00, max=208.00, median=205.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 205.00, 0.00},  // compute_kernel_hw_startup/pack, n=4, min=201.00, max=208.00, median=205.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 206.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, min=205.00, max=208.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 204.25, 0.00},  // compute_kernel_hw_startup/pack, n=4, min=200.00, max=208.00, median=204.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 145.23, 0.00},  // compute_kernel_hw_startup/unpack, n=13, min=143.00, max=147.00, median=146.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 145.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 145.23, 0.00},  // compute_kernel_hw_startup/unpack, n=13, min=143.00, max=147.00, median=146.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 145.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 145.23, 0.00},  // compute_kernel_hw_startup/unpack, n=13, min=143.00, max=147.00, median=146.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 145.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 144.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 144.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 152.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 143.50, 0.00},  // compute_kernel_hw_startup/unpack, n=2, min=143.00, max=144.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 143.50, 0.00},  // compute_kernel_hw_startup/unpack, n=2, min=143.00, max=144.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 145.20, 0.00},  // compute_kernel_hw_startup/unpack, n=25, min=144.00, max=148.00, median=144.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 143.50, 0.00},  // compute_kernel_hw_startup/unpack, n=2, min=143.00, max=144.00
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.94, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.70, 0.00},  // copy_tile/math, n=2, min=18.59, max=18.81
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.94, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.70, 0.00},  // copy_tile/math, n=2, min=18.59, max=18.81
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.94, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.81, 0.00},  // copy_tile/math, n=2, min=18.56, max=19.06
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.05, 0.00},  // copy_tile/math, n=2, min=18.94, max=19.16
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.80, 0.00},  // copy_tile/math, n=2, min=18.72, max=18.88
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.91, 0.00},  // copy_tile/math, n=2, min=18.88, max=18.94
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.80, 0.00},  // copy_tile/math, n=2, min=18.72, max=18.88
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.05, 0.00},  // copy_tile/math, n=2, min=18.94, max=19.16
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.70, 0.00},  // copy_tile/math, n=2, min=18.34, max=19.06
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.05, 0.00},  // copy_tile/math, n=2, min=18.94, max=19.16
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.80, 0.00},  // copy_tile/math, n=2, min=18.72, max=18.88
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.05, 0.00},  // copy_tile/math, n=2, min=18.94, max=19.16
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.80, 0.00},  // copy_tile/math, n=2, min=18.72, max=18.88
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.05, 0.00},  // copy_tile/math, n=2, min=18.94, max=19.16
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.77, 0.00},  // copy_tile/math, n=2, min=18.47, max=19.06
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=234, min=41.38, max=41.72, median=41.56
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.59, 0.00},  // copy_tile/unpack, n=234, min=41.44, max=41.72, median=41.59
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=234, min=41.38, max=41.72, median=41.56
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.59, 0.00},  // copy_tile/unpack, n=234, min=41.44, max=41.72, median=41.59
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=234, min=41.38, max=41.72, median=41.56
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=234, min=41.38, max=41.72, median=41.56
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=234, min=41.41, max=41.91, median=41.78
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.76, 0.00},  // copy_tile/unpack, n=234, min=41.50, max=41.94, median=41.75
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=234, min=41.41, max=41.91, median=41.78
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.75, 0.00},  // copy_tile/unpack, n=234, min=41.50, max=41.94, median=41.75
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=234, min=41.41, max=41.91, median=41.78
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=234, min=41.50, max=41.91, median=41.78
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=234, min=41.97, max=42.47, median=42.34
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.81, 0.00},  // copy_tile/unpack, n=234, min=120.66, max=122.00, median=120.78
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.19, 0.00},  // copy_tile/unpack, n=234, min=41.94, max=42.38, median=42.19
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=234, min=41.97, max=42.47, median=42.34
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.81, 0.00},  // copy_tile/unpack, n=234, min=120.66, max=122.00, median=120.78
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.19, 0.00},  // copy_tile/unpack, n=234, min=41.94, max=42.38, median=42.19
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=234, min=41.97, max=42.47, median=42.34
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.80, 0.00},  // copy_tile/unpack, n=234, min=120.59, max=121.69, median=120.78
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.21, 0.00},  // copy_tile/unpack, n=234, min=41.84, max=42.34, median=42.22
    {"Int32", "Int32", false, 4, "unpack_to_dest=true", Unit::PerTile, 120.33, 0.00},  // copy_tile/unpack, n=44, min=120.25, max=120.41, median=120.28
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 86.00, 0.00},  // copy_tile_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 86.00, 0.00},  // copy_tile_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 84.50, 0.00},  // copy_tile_init/math, n=2, min=83.00, max=86.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 84.50, 0.00},  // copy_tile_init/math, n=2, min=83.00, max=86.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 84.50, 0.00},  // copy_tile_init/math, n=2, min=83.00, max=86.00
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 104.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 104.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerCall, 104.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 280.19, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 283.77, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 280.22, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 283.77, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 280.19, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 283.75, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 283.75, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 283.75, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 283.80, 0.00},  // cos_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 281.58, 0.00},  // cos_tile/math, n=2, min=281.50, max=281.66
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 281.58, 0.00},  // cos_tile/math, n=2, min=281.50, max=281.66
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 281.41, 0.00},  // cos_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 280.28, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 67.50, 0.00},  // cos_tile_init/math, n=2, min=67.00, max=68.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // cos_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // cos_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // cos_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // cos_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.84, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1169.06, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 1072.72, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1169.06, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 1072.84, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1169.05, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.69, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1168.88, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 1072.69, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1168.88, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 1072.69, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1169.05, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1170.17, 0.00},  // div_binary_tile/math, n=2, min=1170.00, max=1170.34
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.70, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1170.17, 0.00},  // div_binary_tile/math, n=2, min=1170.00, max=1170.34
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 1072.84, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1170.06, 0.00},  // div_binary_tile/math, n=2, min=1169.75, max=1170.38
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 1072.69, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=50.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=50.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=51.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=50.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=56.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2, min=49.00, max=50.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 50.75, 0.00},  // div_binary_tile_init/math, n=4, min=49.00, max=56.00, median=49.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.41, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 470.00, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 469.41, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 470.00, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 469.41, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 470.05, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.88, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 469.88, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // eq_binary_tile/math, n=2, min=470.97, max=471.06
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // eq_binary_tile/math, n=2, min=470.97, max=471.06
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 470.83, 0.00},  // eq_binary_tile/math, n=2, min=470.66, max=471.00
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 469.84, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // eq_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // eq_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // eq_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // eq_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // eq_binary_tile_init/math, n=2, min=55.00, max=57.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // eq_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // eq_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // eq_binary_tile_init/math, n=4, min=50.00, max=57.00, median=52.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.27, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.70, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 58.41, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 58.72, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 58.41, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 58.59, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.67, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 58.67, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 58.62, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // eqz_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // eqz_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 57.70, 0.00},  // eqz_tile/math, n=2, min=57.69, max=57.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 58.34, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // eqz_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // eqz_tile_init/math, n=2, min=63.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // eqz_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // eqz_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // eqz_tile_init/math, n=4, min=62.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 387.30, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 387.78, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 387.42, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 387.78, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 387.30, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 387.91, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 387.38, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 387.75, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 387.44, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 387.86, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 387.38, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 387.58, 0.00},  // erf_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 385.69, 0.00},  // erf_tile/math, n=2, min=385.53, max=385.84
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 387.38, 0.00},  // erf_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 385.69, 0.00},  // erf_tile/math, n=2, min=385.53, max=385.84
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 387.38, 0.00},  // erf_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 385.52, 0.00},  // erf_tile/math, n=2, min=385.41, max=385.62
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 387.38, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // erf_tile_init/math, n=2, min=55.00, max=62.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // erf_tile_init/math, n=2, min=55.00, max=62.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // erf_tile_init/math, n=2, min=55.00, max=59.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // erf_tile_init/math, n=2, min=55.00, max=62.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 60.00, 0.00},  // erf_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=55.00, max=61.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, min=55.00, max=59.00, median=56.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 650.44, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 651.25, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 650.44, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 651.25, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 650.44, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 651.45, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 651.09, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 651.08, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 651.20, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 650.05, 0.00},  // erfc_tile/math, n=2, min=649.88, max=650.22
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 650.05, 0.00},  // erfc_tile/math, n=2, min=649.88, max=650.22
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 650.06, 0.00},  // erfc_tile/math, n=2, min=649.91, max=650.22
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 651.02, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, min=53.00, max=58.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2, min=52.00, max=54.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, min=53.00, max=58.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2, min=52.00, max=54.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, min=53.00, max=58.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 54.50, 0.00},  // erfc_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // erfc_tile_init/math, n=2, min=54.00, max=60.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2, min=52.00, max=54.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // erfc_tile_init/math, n=2, min=53.00, max=57.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=4, min=52.00, max=54.00, median=53.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=4, min=52.00, max=54.00, median=53.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.75, 0.00},  // erfc_tile_init/math, n=4, min=52.00, max=57.00, median=53.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 218.22, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 234.91, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 218.38, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 234.91, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 218.22, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 234.84, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 218.56, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 234.86, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 218.56, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 234.86, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 218.56, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 234.88, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 233.83, 0.00},  // exp2_tile/math, n=2, min=233.69, max=233.97
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 218.34, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 233.83, 0.00},  // exp2_tile/math, n=2, min=233.69, max=233.97
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 218.56, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 233.62, 0.00},  // exp2_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, min=53.00, max=58.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 53.00, 0.00},  // exp2_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, min=53.00, max=58.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, min=53.00, max=58.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, min=53.00, max=58.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.00, 0.00},  // exp2_tile_init/math, n=4, min=65.00, max=68.00, median=67.50
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.97, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.36, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.34, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.94, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.09, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.36, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.36, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.94, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.09, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.58, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.58, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.81, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.73, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.70, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.72, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.91, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.62, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.14, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.05, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.72, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.72, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.91, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.62, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.97, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.42, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.53, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.77, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.75, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2, min=330.69, max=331.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2, min=330.69, max=331.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.31, 0.00},  // exp_tile/math, n=2, min=31.25, max=31.38
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.53, 0.00},  // exp_tile/math, n=2, min=113.50, max=113.56
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.88, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2, min=330.69, max=331.00
    {"Float32", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.70, 0.00},  // exp_tile/math, n=2, min=330.69, max=330.72
    {"Float32", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.47, 0.00},  // exp_tile/math, n=2, min=31.38, max=31.56
    {"Float32", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.61, 0.00},  // exp_tile/math, n=2, min=113.50, max=113.72
    {"Float32", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.86, 0.00},  // exp_tile/math, n=2, min=330.81, max=330.91
    {"Float32", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.86, 0.00},  // exp_tile/math, n=2, min=330.81, max=330.91
    {"Float32", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.45, 0.00},  // exp_tile/math, n=2, min=31.41, max=31.50
    {"Float32", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.50, 0.00},  // exp_tile/math, n=2, min=113.41, max=113.59
    {"Float32", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=136.00
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=137.00
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2, min=87.00, max=88.00
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=136.00
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=137.00
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2, min=87.00, max=88.00
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=136.00
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 64.50, 0.00},  // exp_tile_init/math, n=2, min=63.00, max=66.00
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 64.50, 0.00},  // exp_tile_init/math, n=2, min=63.00, max=66.00
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 143.00, 0.00},  // exp_tile_init/math, n=2, min=142.00, max=144.00
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.50, 0.00},  // exp_tile_init/math, n=2, min=87.00, max=90.00
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=129.00
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, min=59.00, max=65.00
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, min=59.00, max=65.00
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 140.00, 0.00},  // exp_tile_init/math, n=2, min=138.00, max=142.00
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2, min=86.00, max=89.00
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 131.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=135.00
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, min=59.00, max=65.00
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, min=59.00, max=65.00
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 140.00, 0.00},  // exp_tile_init/math, n=2, min=138.00, max=142.00
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2, min=86.00, max=89.00
    {"Float16_b", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float16_b", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=129.00
    {"Float16_b", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2, min=85.00, max=90.00
    {"Float16_b", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 60.50, 0.00},  // exp_tile_init/math, n=2, min=58.00, max=63.00
    {"Float16_b", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 60.50, 0.00},  // exp_tile_init/math, n=2, min=58.00, max=63.00
    {"Float16_b", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 134.00, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=140.00
    {"Float16_b", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2, min=87.00, max=89.00
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 74.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=129.00
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.00, 0.00},  // exp_tile_init/math, n=2, min=85.00, max=89.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, min=59.00, max=65.00, median=65.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, min=59.00, max=65.00, median=65.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 133.50, 0.00},  // exp_tile_init/math, n=4, min=126.00, max=142.00, median=133.00
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.75, 0.00},  // exp_tile_init/math, n=4, min=86.00, max=89.00, median=88.00
    {"Float32", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float32", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float32", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=129.00
    {"Float32", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, min=59.00, max=65.00, median=65.00
    {"Float32", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, min=59.00, max=65.00, median=65.00
    {"Float32", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 133.50, 0.00},  // exp_tile_init/math, n=4, min=126.00, max=142.00, median=133.00
    {"Float32", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.75, 0.00},  // exp_tile_init/math, n=4, min=86.00, max=89.00, median=88.00
    {"Float32", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float32", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, min=71.00, max=74.00
    {"Float32", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2, min=128.00, max=129.00
    {"Float32", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 61.50, 0.00},  // exp_tile_init/math, n=4, min=58.00, max=65.00, median=61.50
    {"Float32", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 61.50, 0.00},  // exp_tile_init/math, n=4, min=58.00, max=65.00, median=61.50
    {"Float32", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 130.50, 0.00},  // exp_tile_init/math, n=4, min=128.00, max=136.00, median=129.00
    {"Float32", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.00, 0.00},  // exp_tile_init/math, n=4, min=85.00, max=89.00, median=87.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.20, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 467.02, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 314.20, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 467.02, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 314.20, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 466.95, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.95, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 466.95, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 466.98, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 466.08, 0.00},  // expm1_tile/math, n=2, min=465.91, max=466.25
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 466.08, 0.00},  // expm1_tile/math, n=2, min=465.91, max=466.25
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 465.78, 0.00},  // expm1_tile/math, n=2, min=465.62, max=465.94
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 314.22, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // expm1_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 27.33, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 27.81, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 27.33, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 27.81, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 27.33, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 27.61, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 27.73, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 27.73, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 27.70, 0.00},  // fill_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 26.11, 0.00},  // fill_tile/math, n=2, min=26.03, max=26.19
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 26.11, 0.00},  // fill_tile/math, n=2, min=26.03, max=26.19
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 26.05, 0.00},  // fill_tile/math, n=2, min=26.03, max=26.06
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 26.52, 0.00},  // fill_tile/math, n=2, min=26.38, max=26.66
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // fill_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // fill_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // fill_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // fill_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // fill_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.03, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.58, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.03, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.58, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.03, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.41, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.41, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.41, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.45, 0.00},  // floor_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // floor_tile/math, n=2, min=122.25, max=122.59
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // floor_tile/math, n=2, min=122.25, max=122.59
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 122.53, 0.00},  // floor_tile/math, n=2, min=122.44, max=122.62
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 122.86, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 108.36, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 108.67, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 108.36, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 108.78, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 108.36, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 108.67, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 108.20, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 108.84, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 108.23, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 108.84, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 108.23, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 108.78, 0.00},  // frac_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 106.50, 0.00},  // frac_tile/math, n=2, min=106.44, max=106.56
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 108.23, 0.00},  // frac_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 106.50, 0.00},  // frac_tile/math, n=2, min=106.44, max=106.56
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 108.23, 0.00},  // frac_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 106.44, 0.00},  // frac_tile/math, n=2, min=106.16, max=106.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 108.23, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.38, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.97, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 437.38, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 437.97, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 437.38, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 438.02, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.84, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 437.91, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // ge_binary_tile/math, n=2, min=438.97, max=439.06
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // ge_binary_tile/math, n=2, min=438.97, max=439.06
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 438.83, 0.00},  // ge_binary_tile/math, n=2, min=438.66, max=439.00
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 437.81, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ge_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ge_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ge_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ge_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // ge_binary_tile_init/math, n=2, min=55.00, max=57.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ge_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ge_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // ge_binary_tile_init/math, n=4, min=50.00, max=57.00, median=52.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 755.02, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 755.02, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 755.31, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 754.94, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 714.44, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 754.94, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 754.97, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 754.02, 0.00},  // gelu_tile/math, n=2, min=753.88, max=754.16
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 754.02, 0.00},  // gelu_tile/math, n=2, min=753.88, max=754.16
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 714.33, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 753.89, 0.00},  // gelu_tile/math, n=2, min=753.88, max=753.91
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 714.30, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // gelu_tile_init/math, n=2, min=55.00, max=62.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // gelu_tile_init/math, n=2, min=55.00, max=62.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // gelu_tile_init/math, n=2, min=55.00, max=59.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, min=53.00, max=58.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // gelu_tile_init/math, n=2, min=55.00, max=61.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.00, 0.00},  // gelu_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // gelu_tile_init/math, n=2, min=55.00, max=61.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, min=53.00, max=58.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // gelu_tile_init/math, n=2, min=56.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, min=53.00, max=58.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, min=53.00, max=58.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, min=53.00, max=58.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, min=55.00, max=59.00, median=56.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 92.08, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 92.06, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 91.50, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 91.84, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.62, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 91.94, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 91.64, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 91.98, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 91.62, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 91.86, 0.00},  // gez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 91.14, 0.00},  // gez_tile/math, n=2, min=90.97, max=91.31
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 91.62, 0.00},  // gez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 91.14, 0.00},  // gez_tile/math, n=2, min=90.97, max=91.31
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 91.56, 0.00},  // gez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 91.16, 0.00},  // gez_tile/math, n=2, min=91.03, max=91.28
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 91.64, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.50, 0.00},  // gez_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.50, 0.00},  // gez_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=2, min=64.00, max=65.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=4, min=62.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2, min=64.00, max=66.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=4, min=62.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // gez_tile_init/math, n=4, min=62.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.42, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.95, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 373.42, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 373.92, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 373.42, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 373.83, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.53, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.72, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 373.53, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 373.72, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 373.53, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 373.50, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 374.61, 0.00},  // gt_binary_tile/math, n=2, min=374.50, max=374.72
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 373.66, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // gt_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 373.53, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 374.61, 0.00},  // gt_binary_tile/math, n=2, min=374.50, max=374.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 373.53, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.00, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // gt_binary_tile_init/math, n=2, min=56.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gt_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gt_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.00, 0.00},  // gt_binary_tile_init/math, n=2, min=50.00, max=56.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.00, 0.00},  // gt_binary_tile_init/math, n=4, min=50.00, max=57.00, median=50.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // gt_binary_tile_init/math, n=4, min=50.00, max=57.00, median=53.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.25, 0.00},  // gt_binary_tile_init/math, n=4, min=50.00, max=55.00, median=50.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.39, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.83, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 75.39, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 75.83, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 75.39, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 75.75, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.88, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 75.88, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 75.83, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 75.03, 0.00},  // gtz_tile/math, n=2, min=74.91, max=75.16
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 75.03, 0.00},  // gtz_tile/math, n=2, min=74.91, max=75.16
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 74.86, 0.00},  // gtz_tile/math, n=2, min=74.69, max=75.03
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 75.55, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gtz_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gtz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gtz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 63.50, 0.00},  // gtz_tile_init/math, n=2, min=63.00, max=64.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gtz_tile_init/math, n=4, min=62.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gtz_tile_init/math, n=4, min=62.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.50, 0.00},  // gtz_tile_init/math, n=4, min=62.00, max=65.00, median=63.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 121.86, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 122.84, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 121.86, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 122.84, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 121.86, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 122.67, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 121.81, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 122.83, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 121.81, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 122.83, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 121.81, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 122.66, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 121.33, 0.00},  // hardsigmoid_tile/math, n=2, min=121.22, max=121.44
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.03, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 121.33, 0.00},  // hardsigmoid_tile/math, n=2, min=121.22, max=121.44
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 121.81, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 121.12, 0.00},  // hardsigmoid_tile/math, n=2, min=121.09, max=121.16
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 121.81, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // hardsigmoid_tile_init/math, n=2, min=65.00, max=66.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // hardsigmoid_tile_init/math, n=2, min=65.00, max=66.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.50, 0.00},  // hardsigmoid_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // hardsigmoid_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 63.00, 0.00},  // hardsigmoid_tile_init/math, n=4, min=58.00, max=65.00, median=64.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 63.00, 0.00},  // hardsigmoid_tile_init/math, n=4, min=58.00, max=65.00, median=64.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 62.75, 0.00},  // hardsigmoid_tile_init/math, n=4, min=58.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 138.09, 0.00},  // init_sfpu/math, n=116, min=132.00, max=143.00, median=138.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 136.00, 0.00},  // init_sfpu/math, n=116, min=132.00, max=140.00, median=137.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 138.06, 0.00},  // init_sfpu/math, n=116, min=132.00, max=143.00, median=138.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 136.00, 0.00},  // init_sfpu/math, n=116, min=132.00, max=140.00, median=137.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 138.20, 0.00},  // init_sfpu/math, n=116, min=132.00, max=143.00, median=138.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 137.66, 0.00},  // init_sfpu/math, n=116, min=132.00, max=144.00, median=138.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 137.79, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 135.99, 0.00},  // init_sfpu/math, n=116, min=133.00, max=140.00, median=137.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 137.83, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 135.99, 0.00},  // init_sfpu/math, n=116, min=133.00, max=140.00, median=137.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 137.92, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 137.47, 0.00},  // init_sfpu/math, n=116, min=133.00, max=144.00, median=138.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 137.74, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 135.83, 0.00},  // init_sfpu/math, n=232, min=132.00, max=140.00, median=136.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 137.88, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 135.83, 0.00},  // init_sfpu/math, n=232, min=132.00, max=140.00, median=136.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 137.92, 0.00},  // init_sfpu/math, n=116, min=133.00, max=143.00, median=138.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 137.69, 0.00},  // init_sfpu/math, n=232, min=131.00, max=144.00, median=138.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 141.18, 0.00},  // init_sfpu/math, n=28, min=134.00, max=143.00, median=143.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 201.96, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.50
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 203.65, 0.00},  // init_sfpu/pack, n=232, min=196.00, max=209.00, median=205.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 201.92, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.50
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 203.66, 0.00},  // init_sfpu/pack, n=232, min=196.00, max=209.00, median=205.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 201.94, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.50
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 201.75, 0.00},  // init_sfpu/pack, n=232, min=196.00, max=209.00, median=202.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 201.14, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 202.69, 0.00},  // init_sfpu/pack, n=232, min=196.00, max=210.00, median=203.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 201.14, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 202.78, 0.00},  // init_sfpu/pack, n=232, min=196.00, max=210.00, median=203.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 203.57, 0.00},  // init_sfpu/pack, n=232, min=201.00, max=211.00, median=204.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 203.00, 0.00},  // init_sfpu/pack, n=464, min=196.00, max=210.00, median=203.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 201.14, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 202.97, 0.00},  // init_sfpu/pack, n=464, min=196.00, max=210.00, median=203.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=232, min=195.00, max=208.00, median=202.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 202.89, 0.00},  // init_sfpu/pack, n=464, min=196.00, max=211.00, median=204.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 203.05, 0.00},  // init_sfpu/pack, n=44, min=202.00, max=205.00, median=203.00
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.38, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=205.00, median=200.00
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 199.91, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=201.00, median=200.00
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.39, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=207.00, median=200.00
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 199.93, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=201.00, median=200.00
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.38, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=207.00, median=200.00
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 200.38, 0.00},  // init_sfpu/unpack, n=232, min=199.00, max=205.00, median=200.00
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.13, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.88, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=202.00, median=198.00
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.14, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.87, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=199.00, median=198.00
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.17, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 198.08, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.16, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 205.32, 0.00},  // init_sfpu/unpack, n=232, min=205.00, max=206.00, median=205.00
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.86, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=199.00, median=198.00
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.11, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 205.33, 0.00},  // init_sfpu/unpack, n=232, min=205.00, max=206.00, median=205.00
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.89, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=202.00, median=198.00
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.19, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerCall, 206.79, 0.00},  // init_sfpu/unpack, n=232, min=205.00, max=210.00, median=206.00
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 198.04, 0.00},  // init_sfpu/unpack, n=232, min=197.00, max=203.00, median=198.00
    {"Int32", "Int32", false, 4, "unpack_to_dest=true", Unit::PerCall, 206.61, 0.00},  // init_sfpu/unpack, n=44, min=205.00, max=207.00, median=207.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.38, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.97, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 437.38, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 437.97, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 437.38, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 438.02, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 437.81, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 437.61, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // le_binary_tile/math, n=2, min=438.97, max=439.06
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 437.81, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // le_binary_tile/math, n=2, min=438.97, max=439.06
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 437.94, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 438.83, 0.00},  // le_binary_tile/math, n=2, min=438.66, max=439.00
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 437.81, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // le_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // le_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // le_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // le_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // le_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // le_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 56.50, 0.00},  // le_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // le_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.75, 0.00},  // le_binary_tile_init/math, n=4, min=50.00, max=57.00, median=50.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.69, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 91.98, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 91.69, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 91.98, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 91.69, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 91.88, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.64, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 91.97, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 91.64, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 91.97, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 91.64, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 91.89, 0.00},  // lez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 91.16, 0.00},  // lez_tile/math, n=2, min=91.09, max=91.22
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 91.64, 0.00},  // lez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 91.16, 0.00},  // lez_tile/math, n=2, min=91.09, max=91.22
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 91.66, 0.00},  // lez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 91.14, 0.00},  // lez_tile/math, n=2, min=91.03, max=91.25
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 91.66, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // lez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // lez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // lez_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // lez_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // lez_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // lez_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // lez_tile_init/math, n=4, min=62.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 274.28, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 370.83, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 274.28, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 370.83, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 274.28, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 370.78, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 370.77, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 370.77, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 370.83, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 369.62, 0.00},  // log1p_tile/math, n=2, min=369.56, max=369.69
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 369.62, 0.00},  // log1p_tile/math, n=2, min=369.56, max=369.69
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 369.70, 0.00},  // log1p_tile/math, n=2, min=369.47, max=369.94
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 274.38, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // log1p_tile_init/math, n=2, min=69.00, max=70.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // log1p_tile_init/math, n=2, min=69.00, max=70.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log1p_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log1p_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log1p_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // log1p_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 154.20, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 154.66, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 154.20, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 154.66, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 154.20, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 154.53, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 154.88, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 154.88, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 154.58, 0.00},  // log_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 153.30, 0.00},  // log_tile/math, n=2, min=153.19, max=153.41
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 153.30, 0.00},  // log_tile/math, n=2, min=153.19, max=153.41
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 153.27, 0.00},  // log_tile/math, n=2, min=153.22, max=153.31
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 154.22, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=65.00, max=68.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log_tile_init/math, n=2, min=68.00, max=69.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log_tile_init/math, n=2, min=68.00, max=69.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=65.00, max=68.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log_tile_init/math, n=4, min=67.00, max=69.00, median=67.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log_tile_init/math, n=4, min=67.00, max=69.00, median=67.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, min=66.00, max=67.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // log_tile_init/math, n=4, min=64.00, max=68.00, median=66.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 65.59, 0.00},  // logical_not_tile/math, n=2, min=65.50, max=65.69
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // logical_not_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.42, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.92, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 373.47, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 373.92, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 373.42, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 373.83, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.45, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.72, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 373.53, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 373.69, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 373.53, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 373.50, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // lt_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 373.53, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // lt_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 373.44, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 374.61, 0.00},  // lt_binary_tile/math, n=2, min=374.50, max=374.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 373.53, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // lt_binary_tile_init/math, n=2, min=56.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // lt_binary_tile_init/math, n=2, min=56.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // lt_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // lt_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // lt_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // lt_binary_tile_init/math, n=4, min=50.00, max=57.00, median=53.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // lt_binary_tile_init/math, n=4, min=50.00, max=57.00, median=53.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.25, 0.00},  // lt_binary_tile_init/math, n=4, min=50.00, max=55.00, median=50.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.62, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.92, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 75.62, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 75.92, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 75.62, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 75.81, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.91, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 75.91, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 75.83, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 75.09, 0.00},  // ltz_tile/math, n=2, min=75.03, max=75.16
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 75.09, 0.00},  // ltz_tile/math, n=2, min=75.03, max=75.16
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 75.02, 0.00},  // ltz_tile/math, n=2, min=74.84, max=75.19
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 75.58, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // ltz_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // ltz_tile_init/math, n=2, min=63.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // ltz_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // ltz_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // ltz_tile_init/math, n=4, min=62.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.66, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.62, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // mul_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // mul_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // mul_binary_tile/math, n=2, min=180.50, max=180.66
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 47.50, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // mul_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // mul_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // mul_binary_tile_init/math, n=4, min=47.00, max=55.00, median=47.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 287.41, 0.00},  // mul_int_tile/math, n=2, min=287.12, max=287.69
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 72.50, 0.00},  // mul_int_tile_init/math, n=2, min=70.00, max=75.00
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.34, 0.00},  // mul_tiles/math, n=2, min=36.25, max=36.44
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.09, 0.00},  // mul_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.12, 0.00},  // mul_tiles/math, n=2, min=36.00, max=36.25
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.17, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.34, 0.00},  // mul_tiles/math, n=2, min=36.25, max=36.44
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.09, 0.00},  // mul_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.12, 0.00},  // mul_tiles/math, n=2, min=36.00, max=36.25
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.17, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.34, 0.00},  // mul_tiles/math, n=2, min=36.25, max=36.44
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.05, 0.00},  // mul_tiles/math, n=2, min=19.00, max=19.09
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.23, 0.00},  // mul_tiles/math, n=2, min=36.16, max=36.31
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.25, 0.00},  // mul_tiles/math, n=2, min=18.09, max=18.41
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.31, 0.00},  // mul_tiles/math, n=2, min=36.25, max=36.38
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.09, 0.00},  // mul_tiles/math, n=2, min=35.91, max=36.28
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.02, 0.00},  // mul_tiles/math, n=2, min=59.91, max=60.12
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.19, 0.00},  // mul_tiles/math, n=2, min=84.09, max=84.28
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.19, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.25
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.41, 0.00},  // mul_tiles/math, n=2, min=36.38, max=36.44
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.09, 0.00},  // mul_tiles/math, n=2, min=35.91, max=36.28
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.00, 0.00},  // mul_tiles/math, n=2, min=59.91, max=60.09
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.19, 0.00},  // mul_tiles/math, n=2, min=84.09, max=84.28
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.16, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.19
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.41, 0.00},  // mul_tiles/math, n=2, min=36.38, max=36.44
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float16_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.28, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.12, 0.00},  // mul_tiles/math, n=2, min=60.09, max=60.16
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.09, 0.00},  // mul_tiles/math, n=2, min=84.03, max=84.16
    {"Float16_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.19, 0.00},  // mul_tiles/math, n=2, min=18.09, max=18.28
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.41, 0.00},  // mul_tiles/math, n=2, min=36.38, max=36.44
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.09, 0.00},  // mul_tiles/math, n=2, min=35.91, max=36.28
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.00, 0.00},  // mul_tiles/math, n=2, min=59.91, max=60.09
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.19, 0.00},  // mul_tiles/math, n=2, min=84.09, max=84.28
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.16, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.19
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.41, 0.00},  // mul_tiles/math, n=2, min=36.38, max=36.44
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float32", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.09, 0.00},  // mul_tiles/math, n=2, min=35.91, max=36.28
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.00, 0.00},  // mul_tiles/math, n=2, min=59.91, max=60.09
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.19, 0.00},  // mul_tiles/math, n=2, min=84.09, max=84.28
    {"Float32", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.16, 0.00},  // mul_tiles/math, n=2, min=18.12, max=18.19
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.41, 0.00},  // mul_tiles/math, n=2, min=36.38, max=36.44
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.34, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.11, 0.00},  // mul_tiles/math, n=2, min=83.97, max=84.25
    {"Float32", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.94, 0.00},  // mul_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.28, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.12, 0.00},  // mul_tiles/math, n=2, min=60.09, max=60.16
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.09, 0.00},  // mul_tiles/math, n=2, min=84.03, max=84.16
    {"Float32", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.19, 0.00},  // mul_tiles/math, n=2, min=18.09, max=18.28
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.18, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.19, median=37.19
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.21, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.28, median=37.20
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.18, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.19, median=37.19
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.21, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.28, median=37.20
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.18, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.19, median=37.19
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.18, 0.00},  // mul_tiles/unpack, n=4, min=37.16, max=37.19, median=37.19
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.12, median=37.09
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.16, median=37.09
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.12, median=37.09
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.16, median=37.09
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.11, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.12, median=37.09
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.11, 0.00},  // mul_tiles/unpack, n=8, min=37.09, max=37.12, median=37.09
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.06, 0.00},  // mul_tiles/unpack, n=8, min=40.00, max=40.09, median=40.06
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.77, 0.00},  // mul_tiles/unpack, n=8, min=37.69, max=37.88, median=37.75
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.06, 0.00},  // mul_tiles/unpack, n=8, min=40.00, max=40.09, median=40.06
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // mul_tiles/unpack, n=8, min=37.69, max=37.88, median=37.75
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.06, 0.00},  // mul_tiles/unpack, n=8, min=40.00, max=40.09, median=40.06
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.75, 0.00},  // mul_tiles/unpack, n=8, min=37.69, max=37.81, median=37.73
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2, min=64.00, max=65.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2, min=64.00, max=65.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.25, 0.00},  // mul_tiles_init/unpack, n=4, min=66.00, max=67.00, median=66.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.41, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 470.00, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 469.41, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 470.00, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 469.41, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 470.05, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.84, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 469.97, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 469.84, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 469.84, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // ne_binary_tile/math, n=2, min=470.97, max=471.06
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 469.53, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // ne_binary_tile/math, n=2, min=470.97, max=471.06
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 469.97, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 470.91, 0.00},  // ne_binary_tile/math, n=2, min=470.66, max=471.16
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 469.84, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ne_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ne_binary_tile_init/math, n=2, min=52.00, max=58.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=51.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ne_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 56.50, 0.00},  // ne_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ne_binary_tile_init/math, n=2, min=52.00, max=57.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // ne_binary_tile_init/math, n=2, min=55.00, max=57.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 52.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // ne_binary_tile_init/math, n=4, min=50.00, max=57.00, median=52.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 56.50, 0.00},  // ne_binary_tile_init/math, n=2, min=56.00, max=57.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ne_binary_tile_init/math, n=4, min=50.00, max=57.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, min=50.00, max=57.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=4, min=50.00, max=57.00, median=53.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.30, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.81, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 42.30, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 42.81, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 42.30, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 42.61, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.73, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 42.73, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 42.70, 0.00},  // negative_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 41.09, 0.00},  // negative_tile/math, n=2, min=40.97, max=41.22
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 41.09, 0.00},  // negative_tile/math, n=2, min=40.97, max=41.22
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 41.28, 0.00},  // negative_tile/math, n=2, min=41.19, max=41.38
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 42.20, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // negative_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // negative_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // negative_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // negative_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.25, 0.00},  // negative_tile_init/math, n=4, min=51.00, max=59.00, median=51.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.41, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.70, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 58.41, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 58.70, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 58.41, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 58.59, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.67, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 58.67, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 58.34, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 58.62, 0.00},  // nez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // nez_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // nez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // nez_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 58.34, 0.00},  // nez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 57.70, 0.00},  // nez_tile/math, n=2, min=57.69, max=57.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 58.36, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // nez_tile_init/math, n=2, min=64.00, max=65.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // nez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // nez_tile_init/math, n=2, min=63.00, max=65.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // nez_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2, min=65.00, max=66.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // nez_tile_init/math, n=4, min=65.00, max=66.00, median=65.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // nez_tile_init/math, n=2, min=64.00, max=65.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // nez_tile_init/math, n=4, min=62.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 29.97, 0.00},  // pack_tile/pack, n=242, min=29.50, max=30.34, median=29.97
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 29.20, 0.00},  // pack_tile/pack, n=242, min=28.78, max=29.50, median=29.22
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 38.93, 0.00},  // pack_tile/pack, n=242, min=38.44, max=39.31, median=38.94
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 29.07, 0.00},  // pack_tile/pack, n=242, min=28.66, max=29.38, median=29.09
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 71.04, 0.00},  // pack_tile/pack, n=242, min=70.56, max=71.38, median=71.06
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 43.85, 0.00},  // pack_tile/pack, n=242, min=43.44, max=44.16, median=43.81
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 29.96, 0.00},  // pack_tile/pack, n=246, min=29.53, max=30.34, median=29.97
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 29.19, 0.00},  // pack_tile/pack, n=246, min=28.81, max=29.69, median=29.12
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 29.84, 0.00},  // pack_tile/pack, n=246, min=29.41, max=30.22, median=29.84
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 29.06, 0.00},  // pack_tile/pack, n=246, min=28.69, max=29.53, median=29.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 39.09, 0.00},  // pack_tile/pack, n=246, min=38.75, max=39.47, median=39.12
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 43.97, 0.00},  // pack_tile/pack, n=246, min=43.41, max=44.19, median=44.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 29.97, 0.00},  // pack_tile/pack, n=246, min=29.53, max=30.34, median=29.97
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 29.21, 0.00},  // pack_tile/pack, n=480, min=28.75, max=29.69, median=29.19
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 29.84, 0.00},  // pack_tile/pack, n=246, min=29.41, max=30.22, median=29.84
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 29.08, 0.00},  // pack_tile/pack, n=480, min=28.62, max=29.53, median=29.06
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 39.10, 0.00},  // pack_tile/pack, n=246, min=38.75, max=39.47, median=39.12
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 43.98, 0.00},  // pack_tile/pack, n=480, min=43.41, max=44.19, median=44.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 44.60, 0.00},  // pack_tile/pack, n=44, min=44.34, max=45.06, median=44.66
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.86, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 3217.02, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 3216.86, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 3217.23, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 3216.95, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 3217.06, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 3216.84, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 3216.84, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 3217.27, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 3218.16, 0.00},  // power_binary_tile/math, n=2, min=3218.00, max=3218.31
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 3218.16, 0.00},  // power_binary_tile/math, n=2, min=3218.00, max=3218.31
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 3217.94, 0.00},  // power_binary_tile/math, n=2, min=3217.72, max=3218.16
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 3216.69, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=57.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 52.50, 0.00},  // power_binary_tile_init/math, n=2, min=50.00, max=55.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 53.50, 0.00},  // power_binary_tile_init/math, n=2, min=51.00, max=56.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 53.50, 0.00},  // power_binary_tile_init/math, n=2, min=51.00, max=56.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=2, min=49.00, max=55.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.50, 0.00},  // power_binary_tile_init/math, n=4, min=49.00, max=56.00, median=50.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 761.66, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1228.19, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 761.75, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1228.19, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 761.75, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1228.05, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1228.41, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1228.42, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1228.34, 0.00},  // power_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1239.50, 0.00},  // power_tile/math, n=2, min=1239.22, max=1239.78
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1239.50, 0.00},  // power_tile/math, n=2, min=1239.22, max=1239.78
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1239.25, 0.00},  // power_tile/math, n=2, min=1238.94, max=1239.56
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 761.84, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // power_tile_init/math, n=2, min=65.00, max=67.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // power_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // power_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 63.00, 0.00},  // power_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=2, min=63.00, max=65.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 63.00, 0.00},  // power_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 64.75, 0.00},  // power_tile_init/math, n=4, min=63.00, max=66.00, median=65.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 63.50, 0.00},  // power_tile_init/math, n=4, min=60.00, max=66.00, median=64.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, min=63.00, max=66.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=4, min=63.00, max=65.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 76.25, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 56.86, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 76.25, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 56.86, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 76.25, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 57.02, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 76.28, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 56.88, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 76.06, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 56.88, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 76.06, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 56.58, 0.00},  // recip_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 58.06, 0.00},  // recip_tile/math, n=2, min=58.00, max=58.12
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 76.06, 0.00},  // recip_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 58.06, 0.00},  // recip_tile/math, n=2, min=58.00, max=58.12
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 76.06, 0.00},  // recip_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 58.22, 0.00},  // recip_tile/math, n=2, min=58.12, max=58.31
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 76.06, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 106.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 106.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 109.00, 0.00},  // recip_tile_init/math, n=2, min=108.00, max=110.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 106.50, 0.00},  // recip_tile_init/math, n=2, min=106.00, max=107.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 106.50, 0.00},  // recip_tile_init/math, n=2, min=106.00, max=107.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 112.50, 0.00},  // recip_tile_init/math, n=2, min=112.00, max=113.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 106.75, 0.00},  // recip_tile_init/math, n=4, min=106.00, max=107.00, median=107.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 106.75, 0.00},  // recip_tile_init/math, n=4, min=106.00, max=107.00, median=107.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2, min=78.00, max=79.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 109.50, 0.00},  // recip_tile_init/math, n=4, min=106.00, max=113.00, median=109.50
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 57.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 48.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 57.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 48.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 57.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 48.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 135.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 79.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 80.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 52.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 137.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 134.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 133.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerCall, 51.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerCall, 81.00, 0.00},  // reduce_init/math, n=1
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerCall, 54.00, 0.00},  // reduce_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, min=63.00, max=67.00, median=64.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, min=58.00, max=65.00, median=61.50
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, min=56.00, max=68.00, median=60.50
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, min=63.00, max=67.00, median=64.00
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, min=58.00, max=65.00, median=61.50
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, min=56.00, max=68.00, median=60.50
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, min=63.00, max=67.00, median=64.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, min=58.00, max=65.00, median=61.50
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, min=56.00, max=68.00, median=60.50
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=58.00
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 59.38, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=58.00
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 59.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=59.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=58.00
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=58.00
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, min=56.00, max=68.00, median=56.00
    {"Float32", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, min=56.00, max=64.00, median=58.00
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, min=56.00, max=65.00, median=56.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 73.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.50, 0.00},  // reduce_init/unpack, n=2, min=69.00, max=70.00
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 73.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2, min=68.00, max=69.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 72.50, 0.00},  // reduce_init/unpack, n=2, min=72.00, max=73.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.25, 0.00},  // reduce_init/unpack, n=4, min=67.00, max=68.00, median=67.00
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, min=70.00, max=73.00, median=71.50
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, min=69.00, max=71.00, median=71.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2, min=19.28, max=19.53
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.03, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.31
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 38.11, 0.00},  // reduce_tile/math, n=2, min=37.94, max=38.28
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.03, max=19.38
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2, min=85.41, max=85.75
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.78
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2, min=27.91, max=28.16
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2, min=43.81, max=43.94
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.31
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.94, 0.00},  // reduce_tile/math, n=2, min=72.88, max=73.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2, min=19.28, max=19.53
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.03, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.31
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 38.11, 0.00},  // reduce_tile/math, n=2, min=37.94, max=38.28
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.03, max=19.38
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2, min=85.41, max=85.75
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.78
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2, min=27.91, max=28.16
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2, min=43.81, max=43.94
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.31
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.92, 0.00},  // reduce_tile/math, n=2, min=72.88, max=72.97
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2, min=19.28, max=19.53
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.03, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.31
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.03, max=19.38
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2, min=85.41, max=85.75
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.78
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2, min=27.91, max=28.16
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2, min=43.81, max=43.94
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.31
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.94, 0.00},  // reduce_tile/math, n=2, min=72.88, max=73.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.41
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.50, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.72
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 39.97, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2, min=43.84, max=43.97
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2, min=72.94, max=73.03
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.12
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.41
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.69
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2, min=39.97, max=40.03
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2, min=72.94, max=73.03
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.12
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.41
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.69
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2, min=39.97, max=40.03
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2, min=43.84, max=43.97
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2, min=72.94, max=73.03
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.12
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.41
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.69
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2, min=39.97, max=40.03
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2, min=43.84, max=43.97
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2, min=72.94, max=73.03
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 44.11, 0.00},  // reduce_tile/math, n=2, min=44.09, max=44.12
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.16, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.19
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.25, max=85.69
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2, min=39.97, max=40.03
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2, min=43.84, max=43.97
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2, min=43.91, max=43.97
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2, min=72.94, max=73.03
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.12
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.41
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.28
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2, min=18.75, max=19.03
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2, min=37.88, max=37.94
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2, min=61.88, max=62.03
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2, min=85.88, max=85.94
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2, min=19.12, max=19.50
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.66
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2, min=85.28, max=85.69
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2, min=27.97, max=28.16
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2, min=39.97, max=40.03
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2, min=51.91, max=52.16
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2, min=15.06, max=15.25
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2, min=43.84, max=43.97
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2, min=43.88, max=43.97
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2, min=43.72, max=43.94
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.91, 0.00},  // reduce_tile/math, n=2, min=72.88, max=72.94
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2, min=102.78, max=102.84
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2, min=132.94, max=133.03
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2, min=43.81, max=44.12
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.91, max=37.06, median=36.98
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4, min=36.97, max=37.19, median=37.00
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.70
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.62
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.97, 0.00},  // reduce_tile/unpack, n=4, min=36.91, max=37.03, median=36.97
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4, min=36.97, max=37.19, median=37.00
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.70
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.62
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.91, max=37.06, median=36.98
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4, min=36.94, max=37.06, median=36.97
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4, min=36.97, max=37.19, median=37.00
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.70
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4, min=43.59, max=43.75, median=43.62
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 37.00, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.06, median=37.02
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8, min=37.03, max=37.16, median=37.05
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.31, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.31
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.34
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 37.00, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.06, median=37.02
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8, min=37.03, max=37.16, median=37.05
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.31, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.31
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.34
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.06, median=37.00
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8, min=36.94, max=37.03, median=37.02
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8, min=37.03, max=37.16, median=37.05
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.33
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8, min=51.25, max=51.38, median=51.34
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.90, 0.00},  // reduce_tile/unpack, n=8, min=39.81, max=39.94, median=39.92
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.84, max=39.97, median=39.92
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.88, max=39.94, median=39.92
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8, min=37.66, max=37.78, median=37.69
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.38, median=71.31
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.34, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.44, median=71.38
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.88, max=39.94, median=39.92
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.84, max=39.97, median=39.92
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.88, max=39.94, median=39.92
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8, min=37.66, max=37.78, median=37.69
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.38, median=71.31
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.34, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.44, median=71.38
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.88, max=39.94, median=39.92
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.84, max=39.97, median=39.92
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8, min=39.88, max=39.94, median=39.92
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8, min=37.66, max=37.78, median=37.69
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.38, median=71.31
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.34, 0.00},  // reduce_tile/unpack, n=8, min=71.25, max=71.44, median=71.38
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 56.00, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=59.00, median=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.12, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=59.00, median=55.50
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.62, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=58.00, median=55.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.62, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=58.00, median=55.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.12, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=52.00, median=51.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 54.75, 0.00},  // rounding_op_tile_init/math, n=8, min=51.00, max=59.00, median=54.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // rounding_op_tile_init/math, n=16, min=51.00, max=58.00, median=51.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // rounding_op_tile_init/math, n=16, min=51.00, max=58.00, median=51.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // rounding_op_tile_init/math, n=16, min=51.00, max=59.00, median=51.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 258.08, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 250.72, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 258.30, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 250.72, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 258.30, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 250.64, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 258.28, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 250.80, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 258.31, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 250.80, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 258.31, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 250.55, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2, min=249.22, max=249.59
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 258.31, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2, min=249.22, max=249.59
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 258.31, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2, min=249.31, max=249.50
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 258.31, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 70.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 70.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // rsqrt_tile_init/math, n=2, min=67.00, max=69.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // rsqrt_tile_init/math, n=2, min=69.00, max=70.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // rsqrt_tile_init/math, n=2, min=69.00, max=70.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // rsqrt_tile_init/math, n=2, min=67.00, max=68.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.75, 0.00},  // rsqrt_tile_init/math, n=4, min=66.00, max=70.00, median=69.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.75, 0.00},  // rsqrt_tile_init/math, n=4, min=66.00, max=70.00, median=69.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2, min=68.00, max=70.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // rsqrt_tile_init/math, n=4, min=66.00, max=69.00, median=67.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 410.36, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 402.81, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 410.36, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 402.70, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 410.36, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 402.70, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 402.91, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 402.91, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 402.81, 0.00},  // selu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 401.22, 0.00},  // selu_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 401.23, 0.00},  // selu_tile/math, n=2, min=401.22, max=401.25
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 401.22, 0.00},  // selu_tile/math, n=2, min=401.00, max=401.44
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 410.27, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // selu_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 59.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // selu_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // selu_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // selu_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // selu_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // selu_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // selu_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.28, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 426.75, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 314.28, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 426.75, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 314.28, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 426.92, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.38, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 426.89, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 314.36, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 426.89, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 314.36, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 426.61, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 425.52, 0.00},  // sigmoid_tile/math, n=2, min=425.44, max=425.59
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 314.34, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 425.52, 0.00},  // sigmoid_tile/math, n=2, min=425.44, max=425.59
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 314.36, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 425.47, 0.00},  // sigmoid_tile/math, n=2, min=425.41, max=425.53
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 314.36, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 59.50, 0.00},  // sigmoid_tile_init/math, n=2, min=57.00, max=62.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 59.50, 0.00},  // sigmoid_tile_init/math, n=2, min=57.00, max=62.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // sigmoid_tile_init/math, n=2, min=55.00, max=59.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=61.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=61.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 56.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.75, 0.00},  // sigmoid_tile_init/math, n=4, min=55.00, max=61.00, median=55.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.75, 0.00},  // sigmoid_tile_init/math, n=4, min=55.00, max=61.00, median=55.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // sigmoid_tile_init/math, n=4, min=55.00, max=59.00, median=56.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 90.36, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 90.81, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 90.23, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 90.81, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 90.36, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 90.70, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 90.27, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 90.97, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 90.27, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 90.91, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 90.27, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 90.81, 0.00},  // sign_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 89.28, 0.00},  // sign_tile/math, n=2, min=89.16, max=89.41
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 90.31, 0.00},  // sign_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 89.28, 0.00},  // sign_tile/math, n=2, min=89.16, max=89.41
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 90.27, 0.00},  // sign_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 89.16, 0.00},  // sign_tile/math, n=2, min=89.09, max=89.22
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 90.27, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=52.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // sign_tile_init/math, n=2, min=51.00, max=52.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // sign_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // sign_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // sign_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.81, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 19.95, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.72, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 19.95, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.72, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 20.20, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 20.14, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 20.20, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 20.23, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 20.95, 0.00},  // signbit_tile/math, n=2, min=20.91, max=21.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 20.95, 0.00},  // signbit_tile/math, n=2, min=20.91, max=21.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 20.75, 0.00},  // signbit_tile/math, n=2, min=20.72, max=20.78
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 19.84, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 71.00, 0.00},  // signbit_tile_init/math, n=2, min=68.00, max=74.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 71.00, 0.00},  // signbit_tile_init/math, n=2, min=68.00, max=74.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2, min=69.00, max=71.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=68.00, max=73.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=68.00, max=73.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 70.25, 0.00},  // signbit_tile_init/math, n=4, min=68.00, max=73.00, median=70.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // signbit_tile_init/math, n=4, min=68.00, max=73.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, min=70.00, max=71.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=4, min=68.00, max=73.00, median=70.50
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 347.27, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 347.83, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 347.27, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 347.86, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 347.27, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 347.80, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 347.17, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 347.95, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 347.20, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 347.95, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 347.17, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 347.75, 0.00},  // silu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 345.64, 0.00},  // silu_tile/math, n=2, min=345.47, max=345.81
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 347.17, 0.00},  // silu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 345.64, 0.00},  // silu_tile/math, n=2, min=345.47, max=345.81
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 347.17, 0.00},  // silu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 345.67, 0.00},  // silu_tile/math, n=2, min=345.62, max=345.72
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 347.17, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // silu_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // silu_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, min=51.00, max=58.00, median=51.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, min=51.00, max=58.00, median=51.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 240.19, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 243.77, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 240.30, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 243.77, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 240.19, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 243.75, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 240.28, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 243.75, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 240.36, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 243.75, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 240.36, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 243.73, 0.00},  // sin_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 241.58, 0.00},  // sin_tile/math, n=2, min=241.53, max=241.62
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 240.28, 0.00},  // sin_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 241.58, 0.00},  // sin_tile/math, n=2, min=241.53, max=241.62
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 240.28, 0.00},  // sin_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 241.66, 0.00},  // sin_tile/math, n=2, min=241.41, max=241.91
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 240.28, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sin_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sin_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sin_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // sin_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // sin_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // sin_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 139.25, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 139.62, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 139.25, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 139.81, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 139.25, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 139.89, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 139.86, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 139.95, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 139.58, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 137.42, 0.00},  // softsign_tile/math, n=2, min=137.34, max=137.50
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 137.42, 0.00},  // softsign_tile/math, n=2, min=137.34, max=137.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 137.38, 0.00},  // softsign_tile/math, n=2, min=137.31, max=137.44
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 139.33, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 62.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=55.00, max=61.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // softsign_tile_init/math, n=2, min=55.00, max=59.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=55.00, max=61.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=55.00, max=61.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, min=55.00, max=61.00, median=55.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, min=56.00, max=60.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, min=55.00, max=59.00, median=56.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 226.19, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 218.88, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 226.19, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 218.88, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 226.19, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 218.78, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 226.28, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 218.86, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 226.28, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 218.86, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 226.28, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 218.75, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 217.41, 0.00},  // sqrt_tile/math, n=2, min=217.38, max=217.44
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 226.28, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 217.41, 0.00},  // sqrt_tile/math, n=2, min=217.38, max=217.44
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 226.28, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 217.11, 0.00},  // sqrt_tile/math, n=2, min=217.06, max=217.16
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 226.19, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // sqrt_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 43.39, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 43.75, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 43.34, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 43.78, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 43.36, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 43.67, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 43.23, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 43.88, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 43.28, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 43.88, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 43.23, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 43.78, 0.00},  // square_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 50.36, 0.00},  // square_tile/math, n=2, min=50.31, max=50.41
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 43.23, 0.00},  // square_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 50.38, 0.00},  // square_tile/math, n=2, min=50.31, max=50.44
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 43.23, 0.00},  // square_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 50.30, 0.00},  // square_tile/math, n=2, min=50.25, max=50.34
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 43.23, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // square_tile_init/math, n=2, min=51.00, max=52.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // square_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // square_tile_init/math, n=2, min=51.00, max=59.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // square_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // square_tile_init/math, n=2, min=51.00, max=52.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // square_tile_init/math, n=2, min=51.00, max=58.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // square_tile_init/math, n=2, min=51.00, max=59.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // square_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // square_tile_init/math, n=4, min=51.00, max=58.00, median=51.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // square_tile_init/math, n=4, min=51.00, max=59.00, median=51.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.66, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.08, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.66, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.62, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.92, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // sub_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // sub_binary_tile/math, n=2, min=180.47, max=180.78
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // sub_binary_tile/math, n=2, min=180.50, max=180.66
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.56, 0.00},  // sub_binary_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.89, 0.00},  // sub_binary_tile/math, n=2, min=370.88, max=370.91
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 47.50, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=48.00
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=54.00
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=55.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // sub_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // sub_binary_tile_init/math, n=4, min=47.00, max=54.00, median=47.00
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, min=47.00, max=53.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // sub_binary_tile_init/math, n=4, min=47.00, max=55.00, median=47.00
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, min=48.00, max=53.00
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.41, 0.00},  // sub_int_tile/math, n=2, min=41.28, max=41.53
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 61.00, 0.00},  // sub_int_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.09, 0.00},  // sub_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.09, 0.00},  // sub_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.22
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.09, 0.00},  // sub_tiles/math, n=2, min=19.00, max=19.19
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 18.25, 0.00},  // sub_tiles/math, n=2, min=18.09, max=18.41
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // sub_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.19
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 19.05, 0.00},  // sub_tiles/math, n=2, min=18.91, max=19.19
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.19
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 18.94, 0.00},  // sub_tiles/math, n=2, min=18.91, max=18.97
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 18.19, 0.00},  // sub_tiles/math, n=2, min=18.09, max=18.28
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // sub_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 18.19, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.25
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 18.94, 0.00},  // sub_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 18.16, 0.00},  // sub_tiles/math, n=2, min=18.12, max=18.19
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 18.94, 0.00},  // sub_tiles/math, n=2, min=18.91, max=18.97
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 18.19, 0.00},  // sub_tiles/math, n=2, min=18.09, max=18.28
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.20, 0.00},  // sub_tiles/unpack, n=2, min=37.19, max=37.22
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.20, 0.00},  // sub_tiles/unpack, n=2, min=37.19, max=37.22
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.19, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.19, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.16
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.11, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.12, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.16
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.11, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.11, 0.00},  // sub_tiles/unpack, n=2, min=37.09, max=37.12
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.08, 0.00},  // sub_tiles/unpack, n=2, min=40.06, max=40.09
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // sub_tiles/unpack, n=2, min=37.75, max=37.81
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.05, 0.00},  // sub_tiles/unpack, n=2, min=40.03, max=40.06
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.81, 0.00},  // sub_tiles/unpack, n=2, min=37.75, max=37.88
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.08, 0.00},  // sub_tiles/unpack, n=2, min=40.06, max=40.09
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.75, 0.00},  // sub_tiles/unpack, n=2, min=37.69, max=37.81
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 322.19, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 482.80, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 322.19, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 482.80, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 322.19, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 482.78, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 482.81, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 482.81, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 482.69, 0.00},  // tan_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 481.52, 0.00},  // tan_tile/math, n=2, min=481.44, max=481.59
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 481.52, 0.00},  // tan_tile/math, n=2, min=481.44, max=481.59
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 481.59, 0.00},  // tan_tile/math, n=2, min=481.38, max=481.81
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 322.28, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tan_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tan_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tan_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tan_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // tan_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // tan_tile_init/math, n=4, min=65.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // tan_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 135.98, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 362.91, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 135.98, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 362.91, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 135.98, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 362.84, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 136.16, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 362.86, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 136.16, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 362.86, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 136.16, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 362.88, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 361.81, 0.00},  // tanh_tile/math, n=2, min=361.69, max=361.94
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 136.16, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 361.81, 0.00},  // tanh_tile/math, n=2, min=361.69, max=361.94
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 136.16, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 361.64, 0.00},  // tanh_tile/math, n=2, min=361.62, max=361.66
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 136.14, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tanh_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tanh_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tanh_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=2, min=68.00, max=69.00
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=4, min=68.00, max=69.00, median=68.50
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // tanh_tile_init/math, n=2, min=67.00, max=69.00
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // tanh_tile_init/math, n=4, min=65.00, max=69.00, median=68.00
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 98.94, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 99.48, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 98.94, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 99.48, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 98.94, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 99.31, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 98.77, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 99.31, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 98.77, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 99.31, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 98.86, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 99.23, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 98.33, 0.00},  // trunc_tile/math, n=2, min=98.16, max=98.50
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 98.77, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 98.33, 0.00},  // trunc_tile/math, n=2, min=98.16, max=98.50
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 98.77, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 98.44, 0.00},  // trunc_tile/math, n=2, min=98.34, max=98.53
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 98.77, 0.00},  // trunc_tile/math, n=1
    {"Int32", "UInt16", true, 4, "", Unit::PerTile, 129.84, 0.00},  // typecast_tile/math, n=1
    {"UInt32", "Float16_b", true, 4, "", Unit::PerTile, 141.69, 0.00},  // typecast_tile/math, n=1
};

// The cost table. One entry per ttkernel operation; an engine with no
// value is an engine the operation does not run on, and an entry with no
// engine at all runs nowhere.
static const OpCost kCostTable[] = {
    {"TensorAccessor", EngineCost{0, 0}, {}, {}, {}},
    {"TensorAccessorArgs", {}, {}, {}, {}},
    {"abs_tile", {}, {}, EngineCost{0, 18}, {}},
    {"abs_tile_init", {}, {}, EngineCost{18, 19}, {}},
    {"abs_tile_int32", {}, {}, EngineCost{37, 1}, {}},
    {"acos_tile", {}, {}, EngineCost{38, 18}, {}},
    {"acos_tile_init", {}, {}, EngineCost{56, 18}, {}},
    {"add_binary_tile", {}, {}, EngineCost{74, 19}, {}},
    {"add_binary_tile_init", {}, {}, EngineCost{93, 19}, {}},
    {"add_int_tile", {}, {}, EngineCost{112, 1}, {}},
    {"add_int_tile_init", {}, {}, EngineCost{113, 1}, {}},
    {"add_tiles", {}, EngineCost{132, 18}, EngineCost{114, 18}, {}},
    {"add_tiles_init", {}, EngineCost{168, 18}, EngineCost{150, 18}, {}},
    {"add_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"add_unary_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"asin_tile", {}, {}, EngineCost{186, 18}, {}},
    {"asin_tile_init", {}, {}, EngineCost{204, 18}, {}},
    {"atan2_binary_tile", {}, {}, EngineCost{222, 18}, {}},
    {"atan2_binary_tile_init", {}, {}, EngineCost{240, 18}, {}},
    {"atan_tile", {}, {}, EngineCost{258, 18}, {}},
    {"atan_tile_init", {}, {}, EngineCost{276, 18}, {}},
    {"bfloat16_greater", {}, {}, {}, {}},
    {"binary_bitwise_tile_init", {}, {}, EngineCost{294, 1}, {}},
    {"binary_dest_reuse_tiles", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"binary_dest_reuse_tiles_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"binary_left_shift_tile", {}, {}, EngineCost{295, 1}, {}},
    {"binary_logical_right_shift_tile", {}, {}, EngineCost{296, 1}, {}},
    {"binary_max_int32_tile", {}, {}, EngineCost{297, 1}, {}},
    {"binary_max_int32_tile_init", {}, {}, EngineCost{298, 1}, {}},
    {"binary_max_tile", {}, {}, EngineCost{299, 18}, {}},
    {"binary_max_tile_init", {}, {}, EngineCost{317, 18}, {}},
    {"binary_min_int32_tile", {}, {}, EngineCost{335, 1}, {}},
    {"binary_min_int32_tile_init", {}, {}, EngineCost{336, 1}, {}},
    {"binary_min_tile", {}, {}, EngineCost{337, 18}, {}},
    {"binary_min_tile_init", {}, {}, EngineCost{355, 18}, {}},
    {"binary_op_init_common", {}, EngineCost{409, 18}, EngineCost{373, 18}, EngineCost{391, 18}},
    {"binary_right_shift_tile", {}, {}, EngineCost{427, 1}, {}},
    {"binary_shift_tile_init", {}, {}, EngineCost{428, 1}, {}},
    {"binop_with_scalar_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"bitcast", {}, {}, {}, {}},
    {"bitwise_and_binary_tile", {}, {}, EngineCost{429, 1}, {}},
    {"bitwise_not_tile", {}, {}, EngineCost{430, 1}, {}},
    {"bitwise_not_tile_init", {}, {}, EngineCost{431, 1}, {}},
    {"bitwise_or_binary_tile", {}, {}, EngineCost{432, 1}, {}},
    {"bitwise_xor_binary_tile", {}, {}, EngineCost{433, 1}, {}},
    {"cast_to_l1_addr", {}, {}, {}, {}},
    {"cb_pop_front", EngineCost{0, 0}, EngineCost{0, 0}, {}, {}},
    {"cb_push_back", EngineCost{0, 0}, {}, {}, EngineCost{0, 0}},
    {"cb_reserve_back", EngineCost{0, 0}, {}, {}, EngineCost{0, 0}},
    {"cb_wait_front", EngineCost{0, 0}, EngineCost{0, 0}, {}, {}},
    {"ceil_tile", {}, {}, EngineCost{434, 18}, {}},
    {"clamp_tile", {}, {}, EngineCost{452, 18}, {}},
    {"clamp_tile_init", {}, {}, EngineCost{470, 18}, {}},
    {"clamp_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"compute_kernel_hw_startup", {}, EngineCost{524, 18}, EngineCost{488, 18}, EngineCost{506, 18}},
    {"copy_block_matmul_partials", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"copy_dest_values", {}, {}, EngineCost{0, 0}, {}},
    {"copy_dest_values_init", {}, {}, EngineCost{0, 0}, {}},
    {"copy_tile", {}, EngineCost{560, 22}, EngineCost{542, 18}, {}},
    {"copy_tile_init", {}, EngineCost{600, 21}, EngineCost{582, 18}, {}},
    {"cos_tile", {}, {}, EngineCost{621, 18}, {}},
    {"cos_tile_init", {}, {}, EngineCost{639, 18}, {}},
    {"div_binary_tile", {}, {}, EngineCost{657, 18}, {}},
    {"div_binary_tile_init", {}, {}, EngineCost{675, 18}, {}},
    {"div_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"dprint", EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"eq_binary_tile", {}, {}, EngineCost{693, 18}, {}},
    {"eq_binary_tile_init", {}, {}, EngineCost{711, 18}, {}},
    {"eqz_tile", {}, {}, EngineCost{729, 18}, {}},
    {"eqz_tile_init", {}, {}, EngineCost{747, 18}, {}},
    {"eqz_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"erf_tile", {}, {}, EngineCost{765, 18}, {}},
    {"erf_tile_init", {}, {}, EngineCost{783, 18}, {}},
    {"erfc_tile", {}, {}, EngineCost{801, 18}, {}},
    {"erfc_tile_init", {}, {}, EngineCost{819, 18}, {}},
    {"exp2_tile", {}, {}, EngineCost{837, 18}, {}},
    {"exp2_tile_init", {}, {}, EngineCost{855, 18}, {}},
    {"exp_tile", {}, {}, EngineCost{873, 72}, {}},
    {"exp_tile_init", {}, {}, EngineCost{945, 72}, {}},
    {"experimental.close_fabric_connections", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.constant_table_lookup", {}, {}, {}, {}},
    {"experimental.convert_logical_x_to_translated", {}, {}, {}, {}},
    {"experimental.convert_logical_y_to_translated", {}, {}, {}, {}},
    {"experimental.create_fabric_connection_manager", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.fabric_fast_write_any_len", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.fabric_mcast_fast_write_any_len", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.fabric_mcast_sem_inc", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.fabric_sem_inc", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.fill_arange_tile", {}, {}, EngineCost{0, 0}, {}},
    {"experimental.get_device_id_from_logical_mesh_position", {}, {}, {}, {}},
    {"experimental.get_my_device_id", {}, {}, {}, {}},
    {"experimental.get_my_logical_mesh_position", {}, {}, {}, {}},
    {"experimental.matmul_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"experimental.pack_untilize_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"experimental.semaphore_wait", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.semaphore_wait_min", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.setup_fabric_connections", EngineCost{0, 0}, {}, {}, {}},
    {"experimental.tilize_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"experimental.unpack_stall_on_pack", {}, EngineCost{0, 0}, {}, {}},
    {"experimental.untilize_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"experimental.write_col_mask_tile", {}, {}, EngineCost{0, 0}, {}},
    {"experimental.write_row_mask_tile", {}, {}, EngineCost{0, 0}, {}},
    {"expm1_tile", {}, {}, EngineCost{1017, 18}, {}},
    {"expm1_tile_init", {}, {}, EngineCost{1035, 18}, {}},
    {"fill_tile", {}, {}, EngineCost{1053, 19}, {}},
    {"fill_tile_init", {}, {}, EngineCost{1072, 19}, {}},
    {"fill_tile_int", {}, {}, EngineCost{0, 0}, {}},
    {"float32_greater", {}, {}, {}, {}},
    {"floor_tile", {}, {}, EngineCost{1091, 18}, {}},
    {"frac_tile", {}, {}, EngineCost{1109, 18}, {}},
    {"ge_binary_tile", {}, {}, EngineCost{1127, 18}, {}},
    {"ge_binary_tile_init", {}, {}, EngineCost{1145, 18}, {}},
    {"gelu_tile", {}, {}, EngineCost{1163, 18}, {}},
    {"gelu_tile_init", {}, {}, EngineCost{1181, 18}, {}},
    {"get_arg_val", EngineCost{0, 0}, {}, {}, {}},
    {"get_common_arg_val", EngineCost{0, 0}, {}, {}, {}},
    {"get_compile_time_arg_val", {}, {}, {}, {}},
    {"get_dataformat", {}, {}, {}, {}},
    {"get_dfb_id", {}, {}, {}, {}},
    {"get_noc_addr", EngineCost{0, 0}, {}, {}, {}},
    {"get_noc_addr_from_bank_id", EngineCost{0, 0}, {}, {}, {}},
    {"get_noc_multicast_addr", EngineCost{0, 0}, {}, {}, {}},
    {"get_read_ptr", EngineCost{0, 0}, {}, {}, {}},
    {"get_semaphore", {}, {}, {}, {}},
    {"get_tile_size", {}, {}, {}, {}},
    {"get_write_ptr", EngineCost{0, 0}, {}, {}, {}},
    {"gez_tile", {}, {}, EngineCost{1199, 18}, {}},
    {"gez_tile_init", {}, {}, EngineCost{1217, 18}, {}},
    {"gez_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"gt_binary_tile", {}, {}, EngineCost{1235, 18}, {}},
    {"gt_binary_tile_init", {}, {}, EngineCost{1253, 18}, {}},
    {"gtz_tile", {}, {}, EngineCost{1271, 18}, {}},
    {"gtz_tile_init", {}, {}, EngineCost{1289, 18}, {}},
    {"gtz_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"hardsigmoid_tile", {}, {}, EngineCost{1307, 18}, {}},
    {"hardsigmoid_tile_init", {}, {}, EngineCost{1325, 18}, {}},
    {"init_sfpu", {}, EngineCost{1381, 22}, EngineCost{1343, 19}, EngineCost{1362, 19}},
    {"invoke_sfpi", {}, {}, EngineCost{0, 0}, {}},
    {"le_binary_tile", {}, {}, EngineCost{1403, 18}, {}},
    {"le_binary_tile_init", {}, {}, EngineCost{1421, 18}, {}},
    {"lez_tile", {}, {}, EngineCost{1439, 18}, {}},
    {"lez_tile_init", {}, {}, EngineCost{1457, 18}, {}},
    {"lez_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"load_from_l1", EngineCost{0, 0}, {}, {}, {}},
    {"log1p_tile", {}, {}, EngineCost{1475, 18}, {}},
    {"log1p_tile_init", {}, {}, EngineCost{1493, 18}, {}},
    {"log_tile", {}, {}, EngineCost{1511, 18}, {}},
    {"log_tile_init", {}, {}, EngineCost{1529, 18}, {}},
    {"logical_not_tile", {}, {}, EngineCost{1547, 1}, {}},
    {"logical_not_tile_init", {}, {}, EngineCost{1548, 1}, {}},
    {"lt_binary_tile", {}, {}, EngineCost{1549, 18}, {}},
    {"lt_binary_tile_init", {}, {}, EngineCost{1567, 18}, {}},
    {"ltz_tile", {}, {}, EngineCost{1585, 18}, {}},
    {"ltz_tile_init", {}, {}, EngineCost{1603, 18}, {}},
    {"ltz_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"matmul_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"matmul_tiles", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mem_zeros_base", {}, {}, {}, {}},
    {"mem_zeros_size", {}, {}, {}, {}},
    {"mm_block_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"mm_block_init_short", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mm_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"mm_init_short", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mul_binary_tile", {}, {}, EngineCost{1621, 18}, {}},
    {"mul_binary_tile_init", {}, {}, EngineCost{1639, 18}, {}},
    {"mul_int_tile", {}, {}, EngineCost{1657, 1}, {}},
    {"mul_int_tile_init", {}, {}, EngineCost{1658, 1}, {}},
    {"mul_tiles", {}, EngineCost{1719, 18}, EngineCost{1659, 60}, {}},
    {"mul_tiles_init", {}, EngineCost{1797, 18}, EngineCost{1737, 60}, {}},
    {"mul_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"my_logical_x_", {}, {}, {}, {}},
    {"my_logical_y_", {}, {}, {}, {}},
    {"my_x", {}, {}, {}, {}},
    {"my_y", {}, {}, {}, {}},
    {"ne_binary_tile", {}, {}, EngineCost{1815, 18}, {}},
    {"ne_binary_tile_init", {}, {}, EngineCost{1833, 18}, {}},
    {"negative_tile", {}, {}, EngineCost{1851, 18}, {}},
    {"negative_tile_init", {}, {}, EngineCost{1869, 18}, {}},
    {"negative_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"nez_tile", {}, {}, EngineCost{1887, 18}, {}},
    {"nez_tile_init", {}, {}, EngineCost{1905, 18}, {}},
    {"nez_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"noc_async_atomic_barrier", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read_barrier", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read_barrier_with_trid", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read_one_packet_set_state", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read_one_packet_with_state", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_read_tile", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_barrier", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_barrier_with_trid", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_multicast", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_multicast_loopback_src", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_multicast_one_packet", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_one_packet_set_state", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_one_packet_with_state", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_one_packet_with_trid", EngineCost{0, 0}, {}, {}, {}},
    {"noc_async_write_tile", EngineCost{0, 0}, {}, {}, {}},
    {"noc_inline_dw_write", EngineCost{0, 0}, {}, {}, {}},
    {"noc_semaphore_inc", EngineCost{0, 0}, {}, {}, {}},
    {"noc_semaphore_inc_multicast", EngineCost{0, 0}, {}, {}, {}},
    {"noc_semaphore_set", EngineCost{0, 0}, {}, {}, {}},
    {"noc_semaphore_set_multicast", EngineCost{0, 0}, {}, {}, {}},
    {"noc_semaphore_set_multicast_loopback_src", EngineCost{0, 0}, {}, {}, {}},
    {"opaque_call", EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"pack_reconfig_data_format", {}, {}, {}, EngineCost{0, 0}},
    {"pack_reconfig_l1_acc", {}, {}, {}, EngineCost{0, 0}},
    {"pack_tile", {}, {}, {}, EngineCost{1923, 19}},
    {"pack_tile_block", {}, {}, {}, EngineCost{0, 0}},
    {"pack_untilize_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"pack_untilize_uninit", {}, {}, {}, EngineCost{0, 0}},
    {"power_binary_tile", {}, {}, EngineCost{1942, 18}, {}},
    {"power_binary_tile_init", {}, {}, EngineCost{1960, 18}, {}},
    {"power_tile", {}, {}, EngineCost{1978, 18}, {}},
    {"power_tile_init", {}, {}, EngineCost{1996, 18}, {}},
    {"rand_tile", {}, {}, EngineCost{0, 0}, {}},
    {"rand_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"recip_tile", {}, {}, EngineCost{2014, 18}, {}},
    {"recip_tile_init", {}, {}, EngineCost{2032, 18}, {}},
    {"reduce_init", {}, EngineCost{2257, 54}, EngineCost{2050, 180}, EngineCost{2230, 27}},
    {"reduce_tile", {}, EngineCost{2491, 54}, EngineCost{2311, 180}, {}},
    {"reduce_uninit", {}, {}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"reinterpret_cast", {}, {}, {}, {}},
    {"relu_tile", {}, {}, EngineCost{0, 0}, {}},
    {"relu_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"relu_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"remote_sram_write_u32", EngineCost{0, 0}, {}, {}, {}},
    {"rounding_op_tile_init", {}, {}, EngineCost{2545, 18}, {}},
    {"rsqrt_tile", {}, {}, EngineCost{2563, 18}, {}},
    {"rsqrt_tile_init", {}, {}, EngineCost{2581, 18}, {}},
    {"selu_tile", {}, {}, EngineCost{2599, 18}, {}},
    {"selu_tile_init", {}, {}, EngineCost{2617, 18}, {}},
    {"sfpu_reduce", {}, {}, EngineCost{0, 0}, {}},
    {"sfpu_reduce_init", {}, {}, EngineCost{0, 0}, {}},
    {"sigmoid_tile", {}, {}, EngineCost{2635, 18}, {}},
    {"sigmoid_tile_init", {}, {}, EngineCost{2653, 18}, {}},
    {"sign_tile", {}, {}, EngineCost{2671, 18}, {}},
    {"sign_tile_init", {}, {}, EngineCost{2689, 18}, {}},
    {"signbit_tile", {}, {}, EngineCost{2707, 18}, {}},
    {"signbit_tile_init", {}, {}, EngineCost{2725, 18}, {}},
    {"silu_tile", {}, {}, EngineCost{2743, 18}, {}},
    {"silu_tile_init", {}, {}, EngineCost{2761, 18}, {}},
    {"sin_tile", {}, {}, EngineCost{2779, 18}, {}},
    {"sin_tile_init", {}, {}, EngineCost{2797, 18}, {}},
    {"softsign_tile", {}, {}, EngineCost{2815, 18}, {}},
    {"softsign_tile_init", {}, {}, EngineCost{2833, 18}, {}},
    {"sqrt_tile", {}, {}, EngineCost{2851, 18}, {}},
    {"sqrt_tile_init", {}, {}, EngineCost{2869, 18}, {}},
    {"square_tile", {}, {}, EngineCost{2887, 18}, {}},
    {"square_tile_init", {}, {}, EngineCost{2905, 18}, {}},
    {"store_to_l1", EngineCost{0, 0}, {}, {}, {}},
    {"sub_binary_tile", {}, {}, EngineCost{2923, 19}, {}},
    {"sub_binary_tile_init", {}, {}, EngineCost{2942, 19}, {}},
    {"sub_int_tile", {}, {}, EngineCost{2961, 1}, {}},
    {"sub_int_tile_init", {}, {}, EngineCost{2962, 1}, {}},
    {"sub_tiles", {}, EngineCost{2981, 18}, EngineCost{2963, 18}, {}},
    {"sub_tiles_init", {}, EngineCost{3017, 18}, EngineCost{2999, 18}, {}},
    {"sub_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"sub_unary_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"tan_tile", {}, {}, EngineCost{3035, 18}, {}},
    {"tan_tile_init", {}, {}, EngineCost{3053, 18}, {}},
    {"tanh_tile", {}, {}, EngineCost{3071, 18}, {}},
    {"tanh_tile_init", {}, {}, EngineCost{3089, 18}, {}},
    {"tensor_accessor.get_bank_and_offset", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.get_noc_addr", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.get_shard_noc_addr", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.is_local_addr", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.is_local_bank", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.is_local_page", EngineCost{0, 0}, {}, {}, {}},
    {"tensor_accessor.is_local_shard", EngineCost{0, 0}, {}, {}, {}},
    {"tile_regs_acquire", {}, {}, EngineCost{0, 0}, {}},
    {"tile_regs_commit", {}, {}, EngineCost{0, 0}, {}},
    {"tile_regs_release", {}, {}, {}, EngineCost{0, 0}},
    {"tile_regs_wait", {}, {}, {}, EngineCost{0, 0}},
    {"tilize_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"tilize_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"tilize_uninit", {}, EngineCost{0, 0}, {}, EngineCost{0, 0}},
    {"topk_local_sort", {}, {}, EngineCost{0, 0}, {}},
    {"topk_merge", {}, {}, EngineCost{0, 0}, {}},
    {"topk_rebuild", {}, {}, EngineCost{0, 0}, {}},
    {"topk_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"transpose_wh_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"transpose_wh_tile", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"trunc_tile", {}, {}, EngineCost{3107, 18}, {}},
    {"typecast_tile", {}, {}, EngineCost{3125, 2}, {}},
    {"typecast_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"unary_bcast", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"unary_bcast_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"unary_op_init_common", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"unreachable", {}, {}, {}, {}},
    {"untilize_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"untilize_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"untilize_uninit", {}, EngineCost{0, 0}, {}, {}},
    {"where_tile", {}, {}, EngineCost{0, 0}, {}},
    {"where_tile_init", {}, {}, EngineCost{0, 0}, {}},
};
