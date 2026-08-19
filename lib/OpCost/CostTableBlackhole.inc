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
// Generated 2026-08-19 22:10:13 UTC, from 11222 CSV rows over 7 benchmarks,
// measured on blackhole; costs are not transferable across architectures. The
// CSVs are in neither tree, so the revisions they were produced against are:
//
//   tt-lang-ops-and-models  4f761072   the sweeps and this generator
//   tt-lang                 7c35360c8  the dialect the keys are checked against
//   tt-lang's tt-metal      ea042c4ad  tt-llk and the perf harness
//
// Included into a scope that defines `MeasuredCost`, `EngineCost` and `OpCost`;
// this file is data only. See include/ttlang/OpCost/OpCost.h. What the table
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
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.16, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.83, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 42.35, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 42.83, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 42.27, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 42.76, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.75, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 42.75, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 42.26, 0.00},  // abs_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 42.72, 0.00},  // abs_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 41.17, 0.00},  // abs_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 42.25, 0.00},  // abs_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 41.17, 0.00},  // abs_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // abs_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 41.22, 0.00},  // abs_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 42.26, 0.00},  // abs_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // abs_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // abs_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // abs_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // abs_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.25, 0.00},  // abs_tile_init/math, n=4, spread=8.9%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // abs_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // abs_tile_init/math, n=4, spread=11.3%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 57.00, 0.00},  // abs_tile_init/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.48, 0.00},  // abs_tile_int32/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 586.34, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 490.84, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 586.34, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 490.85, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 586.34, 0.00},  // acos_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 490.76, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 586.26, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 490.85, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 586.26, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 490.85, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 586.26, 0.00},  // acos_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 490.77, 0.00},  // acos_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 489.69, 0.00},  // acos_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 586.25, 0.00},  // acos_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 489.69, 0.00},  // acos_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 586.26, 0.00},  // acos_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 489.77, 0.00},  // acos_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 586.26, 0.00},  // acos_tile/math, n=1
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // acos_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // acos_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // acos_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // acos_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.54, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.03, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.55, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.02, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.54, 0.00},  // add_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.82, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.60, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.58, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.60, 0.00},  // add_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // add_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.69, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // add_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // add_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // add_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.59, 0.00},  // add_binary_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.89, 0.00},  // add_binary_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 47.50, 0.00},  // add_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // add_binary_tile_init/math, n=2, spread=7.8%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // add_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // add_binary_tile_init/math, n=4, spread=10.8%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // add_binary_tile_init/math, n=4, spread=10.8%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // add_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // add_binary_tile_init/math, n=4, spread=12.2%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // add_binary_tile_init/math, n=2, spread=5.0%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.41, 0.00},  // add_int_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 61.00, 0.00},  // add_int_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.12, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.20, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.12, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 18.20, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.05, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 18.27, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 18.31, 0.00},  // add_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // add_tiles/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 18.95, 0.00},  // add_tiles/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 18.31, 0.00},  // add_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.17, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.23, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.17, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.23, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.17, 0.00},  // add_tiles/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.17, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.14, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.09, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.09, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.09, 0.00},  // add_tiles/unpack, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.83, 0.00},  // add_tiles/unpack, n=2
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // add_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // add_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // add_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.00, 0.00},  // add_tiles_init/unpack, n=1
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
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 578.44, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.83, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 578.43, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 466.83, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 578.43, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 466.70, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.72, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 466.78, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 466.69, 0.00},  // asin_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 465.66, 0.00},  // asin_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 465.66, 0.00},  // asin_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 465.47, 0.00},  // asin_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 578.32, 0.00},  // asin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // asin_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // asin_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // asin_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.87, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1873.61, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 1264.88, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1873.59, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 1264.87, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1873.46, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.89, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1873.55, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 1264.87, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1873.55, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 1264.88, 0.00},  // atan2_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1873.29, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1874.69, 0.00},  // atan2_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 1264.87, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1874.69, 0.00},  // atan2_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 1264.88, 0.00},  // atan2_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1874.34, 0.00},  // atan2_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 1264.87, 0.00},  // atan2_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.00, 0.00},  // atan2_binary_tile_init/math, n=2, spread=5.6%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 47.50, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // atan2_binary_tile_init/math, n=2, spread=6.7%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // atan2_binary_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.50, 0.00},  // atan2_binary_tile_init/math, n=4, spread=3.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 47.00, 0.00},  // atan2_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 50.75, 0.00},  // atan2_binary_tile_init/math, n=4, spread=10.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 426.39, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 338.72, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 338.72, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 338.48, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 426.39, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 338.72, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 338.72, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 338.50, 0.00},  // atan_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 337.78, 0.00},  // atan_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 337.78, 0.00},  // atan_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 426.40, 0.00},  // atan_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 337.59, 0.00},  // atan_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 426.38, 0.00},  // atan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // atan_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // atan_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.50, 0.00},  // atan_tile_init/math, n=2, spread=2.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, spread=3.4%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 56.00, 0.00},  // atan_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // atan_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // atan_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // atan_tile_init/math, n=4, spread=3.1%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 47.50, 0.00},  // binary_bitwise_tile_init/math, n=6
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 338.94, 0.00},  // binary_left_shift_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.86, 0.00},  // binary_logical_right_shift_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 238.45, 0.00},  // binary_max_int32_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 73.50, 0.00},  // binary_max_int32_tile_init/math, n=2, spread=3.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.29, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.16, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.30, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 124.14, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.29, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.96, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_max_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_max_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_max_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_max_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 139.31, 0.00},  // binary_max_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_max_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_max_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_max_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 65.25, 0.00},  // binary_max_tile_init/math, n=4, spread=3.4%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 238.48, 0.00},  // binary_min_int32_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 73.50, 0.00},  // binary_min_int32_tile_init/math, n=2, spread=3.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.29, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 124.16, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.30, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 124.14, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.29, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.96, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.31, 0.00},  // binary_min_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.91, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_min_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 139.25, 0.00},  // binary_min_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // binary_min_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 139.31, 0.00},  // binary_min_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.26, 0.00},  // binary_min_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // binary_min_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // binary_min_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 65.25, 0.00},  // binary_min_tile_init/math, n=4, spread=3.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 95.75, 0.00},  // binary_op_init_common/math, n=4
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 93.25, 0.00},  // binary_op_init_common/math, n=4, spread=2.4%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 95.75, 0.00},  // binary_op_init_common/math, n=4
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 93.25, 0.00},  // binary_op_init_common/math, n=4, spread=2.4%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 94.75, 0.00},  // binary_op_init_common/math, n=4, spread=2.9%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 95.00, 0.00},  // binary_op_init_common/math, n=4
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 92.00, 0.00},  // binary_op_init_common/math, n=6
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 94.33, 0.00},  // binary_op_init_common/math, n=6, spread=3.5%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 92.00, 0.00},  // binary_op_init_common/math, n=6
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 94.33, 0.00},  // binary_op_init_common/math, n=6, spread=3.5%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 92.00, 0.00},  // binary_op_init_common/math, n=6
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 93.33, 0.00},  // binary_op_init_common/math, n=6, spread=2.5%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 92.50, 0.00},  // binary_op_init_common/math, n=6, spread=2.7%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 94.33, 0.00},  // binary_op_init_common/math, n=6, spread=3.5%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 92.00, 0.00},  // binary_op_init_common/math, n=6
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 94.33, 0.00},  // binary_op_init_common/math, n=6, spread=3.5%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 92.00, 0.00},  // binary_op_init_common/math, n=6
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 93.33, 0.00},  // binary_op_init_common/math, n=6, spread=2.5%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 201.75, 0.00},  // binary_op_init_common/pack, n=8
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 204.62, 0.00},  // binary_op_init_common/pack, n=8, spread=3.2%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 201.75, 0.00},  // binary_op_init_common/pack, n=8
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 204.62, 0.00},  // binary_op_init_common/pack, n=8, spread=3.2%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 201.75, 0.00},  // binary_op_init_common/pack, n=8
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 201.62, 0.00},  // binary_op_init_common/pack, n=8, spread=3.2%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 204.83, 0.00},  // binary_op_init_common/pack, n=12
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 203.25, 0.00},  // binary_op_init_common/pack, n=12, spread=2.3%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 205.08, 0.00},  // binary_op_init_common/pack, n=12
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 202.67, 0.00},  // binary_op_init_common/pack, n=12
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 204.92, 0.00},  // binary_op_init_common/pack, n=12
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 203.50, 0.00},  // binary_op_init_common/pack, n=12, spread=2.7%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 204.83, 0.00},  // binary_op_init_common/pack, n=12
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 202.42, 0.00},  // binary_op_init_common/pack, n=12, spread=2.2%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 204.83, 0.00},  // binary_op_init_common/pack, n=12
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 202.67, 0.00},  // binary_op_init_common/pack, n=12
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 204.83, 0.00},  // binary_op_init_common/pack, n=12
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 203.42, 0.00},  // binary_op_init_common/pack, n=12, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 169.75, 0.00},  // binary_op_init_common/unpack, n=4, spread=3.1%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 168.50, 0.00},  // binary_op_init_common/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 169.75, 0.00},  // binary_op_init_common/unpack, n=4, spread=3.1%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 169.00, 0.00},  // binary_op_init_common/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 169.75, 0.00},  // binary_op_init_common/unpack, n=4, spread=3.1%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 169.75, 0.00},  // binary_op_init_common/unpack, n=4, spread=3.1%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 169.67, 0.00},  // binary_op_init_common/unpack, n=6
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 169.67, 0.00},  // binary_op_init_common/unpack, n=6
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 171.50, 0.00},  // binary_op_init_common/unpack, n=6, spread=3.2%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 170.00, 0.00},  // binary_op_init_common/unpack, n=6
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 169.67, 0.00},  // binary_op_init_common/unpack, n=6
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 171.33, 0.00},  // binary_op_init_common/unpack, n=6, spread=2.7%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 595.03, 0.00},  // binary_right_shift_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 48.50, 0.00},  // binary_shift_tile_init/math, n=6, spread=9.3%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_and_binary_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 48.88, 0.00},  // bitwise_not_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 57.00, 0.00},  // bitwise_not_tile_init/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_or_binary_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 180.83, 0.00},  // bitwise_xor_binary_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.02, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.59, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.01, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.60, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.01, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.56, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.92, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.43, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 122.91, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.42, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 122.92, 0.00},  // ceil_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.47, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // ceil_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.91, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // ceil_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 122.92, 0.00},  // ceil_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 122.53, 0.00},  // ceil_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 122.92, 0.00},  // ceil_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.28, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.78, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.27, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.78, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.27, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.85, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.34, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.88, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 123.41, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.88, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 123.34, 0.00},  // clamp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.79, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 121.38, 0.00},  // clamp_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 123.33, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 121.38, 0.00},  // clamp_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 123.34, 0.00},  // clamp_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 121.39, 0.00},  // clamp_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 123.34, 0.00},  // clamp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // clamp_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // clamp_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.50, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 58.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // clamp_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // clamp_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // clamp_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 54.50, 0.00},  // clamp_tile_init/math, n=4, spread=8.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 93.23, 0.00},  // compute_kernel_hw_startup/math, n=13, spread=4.5%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 93.23, 0.00},  // compute_kernel_hw_startup/math, n=13, spread=4.5%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 93.23, 0.00},  // compute_kernel_hw_startup/math, n=13, spread=4.5%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 94.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 93.16, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.5%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 93.16, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 95.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 93.16, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.5%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 94.00, 0.00},  // compute_kernel_hw_startup/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 93.16, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 95.50, 0.00},  // compute_kernel_hw_startup/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 93.28, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.6%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 95.50, 0.00},  // compute_kernel_hw_startup/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 93.28, 0.00},  // compute_kernel_hw_startup/math, n=25, spread=4.6%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 93.00, 0.00},  // compute_kernel_hw_startup/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 202.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, spread=2.7%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 202.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, spread=2.7%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 202.50, 0.00},  // compute_kernel_hw_startup/pack, n=2, spread=2.7%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 204.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 208.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 204.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 207.00, 0.00},  // compute_kernel_hw_startup/pack, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 207.00, 0.00},  // compute_kernel_hw_startup/pack, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 206.00, 0.00},  // compute_kernel_hw_startup/pack, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 204.75, 0.00},  // compute_kernel_hw_startup/pack, n=4, spread=2.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 145.38, 0.00},  // compute_kernel_hw_startup/unpack, n=13
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 145.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 145.38, 0.00},  // compute_kernel_hw_startup/unpack, n=13
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 143.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 145.38, 0.00},  // compute_kernel_hw_startup/unpack, n=13
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 147.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 145.36, 0.00},  // compute_kernel_hw_startup/unpack, n=25
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 143.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 145.36, 0.00},  // compute_kernel_hw_startup/unpack, n=25
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 143.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 145.40, 0.00},  // compute_kernel_hw_startup/unpack, n=25
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 148.00, 0.00},  // compute_kernel_hw_startup/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 145.36, 0.00},  // compute_kernel_hw_startup/unpack, n=25
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 143.00, 0.00},  // compute_kernel_hw_startup/unpack, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 145.48, 0.00},  // compute_kernel_hw_startup/unpack, n=25, spread=3.8%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 144.50, 0.00},  // compute_kernel_hw_startup/unpack, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 145.36, 0.00},  // compute_kernel_hw_startup/unpack, n=25
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 145.50, 0.00},  // compute_kernel_hw_startup/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.12, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.75, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.06, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.75, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.06, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.64, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.11, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.81, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.11, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.81, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.11, 0.00},  // copy_tile/math, n=2
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.69, 0.00},  // copy_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 19.03, 0.00},  // copy_tile/math, n=2
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.81, 0.00},  // copy_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.94, 0.00},  // copy_tile/math, n=2
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.81, 0.00},  // copy_tile/math, n=2
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 18.97, 0.00},  // copy_tile/math, n=2
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 18.77, 0.00},  // copy_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=238
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.59, 0.00},  // copy_tile/unpack, n=238
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=238
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.59, 0.00},  // copy_tile/unpack, n=238
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=238
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.54, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.76, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.76, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=238
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 41.77, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.81, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.19, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.81, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.19, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerTile, 42.33, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerTile, 120.80, 0.00},  // copy_tile/unpack, n=238
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerTile, 42.21, 0.00},  // copy_tile/unpack, n=238
    {"Int32", "Int32", false, 4, "unpack_to_dest=true", Unit::PerTile, 120.33, 0.00},  // copy_tile/unpack, n=44
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 83.00, 0.00},  // copy_tile_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // copy_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 94.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 105.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 92.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 105.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerCall, 105.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 93.00, 0.00},  // copy_tile_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 280.22, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 283.66, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 280.21, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 283.66, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 280.21, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 283.64, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 280.17, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 283.69, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 280.16, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 283.69, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 280.17, 0.00},  // cos_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 283.69, 0.00},  // cos_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 281.58, 0.00},  // cos_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 280.16, 0.00},  // cos_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 281.58, 0.00},  // cos_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 280.17, 0.00},  // cos_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 281.41, 0.00},  // cos_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 280.17, 0.00},  // cos_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // cos_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // cos_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // cos_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // cos_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.85, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1169.27, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 1072.86, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1169.25, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 1072.85, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1169.12, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.82, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1169.05, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 1072.80, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1169.05, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 1072.81, 0.00},  // div_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1169.05, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1170.17, 0.00},  // div_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 1072.81, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1170.17, 0.00},  // div_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 1072.81, 0.00},  // div_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1170.23, 0.00},  // div_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 1072.76, 0.00},  // div_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 50.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // div_binary_tile_init/math, n=2, spread=6.7%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.50, 0.00},  // div_binary_tile_init/math, n=4, spread=3.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // div_binary_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // div_binary_tile_init/math, n=2, spread=5.8%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 50.75, 0.00},  // div_binary_tile_init/math, n=4, spread=10.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.48, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.95, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 469.49, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 469.94, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 469.48, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 469.95, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.79, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 469.77, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 469.78, 0.00},  // eq_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 469.87, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 470.91, 0.00},  // eq_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 469.78, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // eq_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 469.78, 0.00},  // eq_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 470.83, 0.00},  // eq_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 469.77, 0.00},  // eq_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // eq_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // eq_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // eq_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // eq_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // eq_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // eq_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // eq_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // eq_binary_tile_init/math, n=4, spread=7.5%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.74, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 58.30, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 58.74, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 58.30, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 58.73, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.72, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 58.72, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 58.69, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // eqz_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 58.30, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // eqz_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 57.70, 0.00},  // eqz_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 58.31, 0.00},  // eqz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // eqz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // eqz_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // eqz_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // eqz_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // eqz_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // eqz_tile_init/math, n=4, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 387.31, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 387.91, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 387.30, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 387.91, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 387.30, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 387.76, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 387.32, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 387.90, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 387.29, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 387.89, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 387.29, 0.00},  // erf_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 387.69, 0.00},  // erf_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 385.69, 0.00},  // erf_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 387.28, 0.00},  // erf_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 385.69, 0.00},  // erf_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 387.29, 0.00},  // erf_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 385.52, 0.00},  // erf_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 387.29, 0.00},  // erf_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // erf_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // erf_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // erf_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // erf_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // erf_tile_init/math, n=2, spread=2.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erf_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // erf_tile_init/math, n=4, spread=4.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 650.58, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 651.28, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 650.57, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 651.28, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 650.57, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 651.26, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 650.76, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 651.10, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 650.76, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 651.10, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 650.76, 0.00},  // erfc_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 651.24, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 650.05, 0.00},  // erfc_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 650.75, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 650.05, 0.00},  // erfc_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 650.76, 0.00},  // erfc_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 650.06, 0.00},  // erfc_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 650.84, 0.00},  // erfc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 56.00, 0.00},  // erfc_tile_init/math, n=2, spread=7.1%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 54.50, 0.00},  // erfc_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // erfc_tile_init/math, n=2, spread=3.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // erfc_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // erfc_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // erfc_tile_init/math, n=2, spread=4.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.75, 0.00},  // erfc_tile_init/math, n=4, spread=6.0%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 218.36, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 234.80, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 218.35, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 234.80, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 218.35, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 234.73, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 234.80, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 234.80, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 234.77, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 233.83, 0.00},  // exp2_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 218.30, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 233.83, 0.00},  // exp2_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 233.62, 0.00},  // exp2_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 218.31, 0.00},  // exp2_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, spread=4.5%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // exp2_tile_init/math, n=2, spread=4.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // exp2_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // exp2_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // exp2_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.09, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.36, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.39, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.94, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 152.12, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 151.79, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.09, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.36, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.39, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.94, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.78, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.13, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.80, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.09, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.58, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.65, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.81, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.73, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.03, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.88, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.93, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.91, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.62, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.72, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.77, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.91, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.62, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.03, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 331.42, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 331.49, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 28.77, 0.00},  // exp_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 112.75, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.47, 0.00},  // exp_tile/math, n=2
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.53, 0.00},  // exp_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.88, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float16_b", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.84, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.47, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float16_b", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.53, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.03, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float16_b", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.80, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 330.86, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float32", true, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 330.86, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 31.45, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float32", true, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 113.50, 0.00},  // exp_tile/math, n=2
    {"Float32", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=false", Unit::PerTile, 151.95, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=false;iterations=8;input_clamping=true", Unit::PerTile, 152.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=false", Unit::PerTile, 29.02, 0.00},  // exp_tile/math, n=1
    {"Float32", "Float32", false, 4, "approx_mode=true;iterations=8;input_clamping=true", Unit::PerTile, 111.89, 0.00},  // exp_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, spread=3.0%
    {"Bfp8_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.50, 0.00},  // exp_tile_init/math, n=2, spread=3.4%
    {"Bfp8_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, spread=3.0%
    {"Bfp8_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 66.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.50, 0.00},  // exp_tile_init/math, n=2, spread=3.4%
    {"Bfp8_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 73.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 132.00, 0.00},  // exp_tile_init/math, n=2, spread=3.0%
    {"Bfp8_b", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 64.50, 0.00},  // exp_tile_init/math, n=2, spread=2.3%
    {"Bfp8_b", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 64.50, 0.00},  // exp_tile_init/math, n=2, spread=2.3%
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 143.00, 0.00},  // exp_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 140.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 62.00, 0.00},  // exp_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 140.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float16_b", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 60.50, 0.00},  // exp_tile_init/math, n=2, spread=4.1%
    {"Float16_b", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 60.50, 0.00},  // exp_tile_init/math, n=2, spread=4.1%
    {"Float16_b", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 134.00, 0.00},  // exp_tile_init/math, n=2, spread=4.5%
    {"Float16_b", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 88.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Bfp8_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.00, 0.00},  // exp_tile_init/math, n=2, spread=2.3%
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, spread=7.1%
    {"Float32", "Bfp8_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, spread=7.1%
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 133.50, 0.00},  // exp_tile_init/math, n=4, spread=6.4%
    {"Float32", "Bfp8_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.75, 0.00},  // exp_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Float16_b", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Float16_b", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float16_b", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 62.25, 0.00},  // exp_tile_init/math, n=4, spread=5.2%
    {"Float32", "Float16_b", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 63.50, 0.00},  // exp_tile_init/math, n=4, spread=7.1%
    {"Float32", "Float16_b", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 133.50, 0.00},  // exp_tile_init/math, n=4, spread=6.4%
    {"Float32", "Float16_b", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.75, 0.00},  // exp_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Float32", false, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 72.50, 0.00},  // exp_tile_init/math, n=2, spread=2.1%
    {"Float32", "Float32", false, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 128.50, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float32", false, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 85.00, 0.00},  // exp_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "approx_mode=false;input_clamping=false", Unit::PerCall, 61.75, 0.00},  // exp_tile_init/math, n=4, spread=6.1%
    {"Float32", "Float32", true, 4, "approx_mode=false;input_clamping=true", Unit::PerCall, 61.50, 0.00},  // exp_tile_init/math, n=4, spread=5.7%
    {"Float32", "Float32", true, 4, "approx_mode=true;input_clamping=false", Unit::PerCall, 131.50, 0.00},  // exp_tile_init/math, n=4, spread=6.5%
    {"Float32", "Float32", true, 4, "approx_mode=true;input_clamping=true", Unit::PerCall, 87.00, 0.00},  // exp_tile_init/math, n=4, spread=2.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.28, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.91, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 314.21, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 466.91, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 314.27, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 466.84, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.37, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 466.90, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 314.37, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 466.89, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 314.37, 0.00},  // expm1_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 466.88, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 466.08, 0.00},  // expm1_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 314.36, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 466.08, 0.00},  // expm1_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 314.37, 0.00},  // expm1_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 465.78, 0.00},  // expm1_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 314.37, 0.00},  // expm1_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // expm1_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // expm1_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // expm1_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 27.31, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 27.83, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 27.30, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 27.83, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 27.30, 0.00},  // fill_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 27.76, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 27.37, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 27.75, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 27.37, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 27.75, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 27.37, 0.00},  // fill_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 27.72, 0.00},  // fill_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 26.17, 0.00},  // fill_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 27.36, 0.00},  // fill_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 26.11, 0.00},  // fill_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 27.37, 0.00},  // fill_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 26.05, 0.00},  // fill_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 27.37, 0.00},  // fill_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 26.52, 0.00},  // fill_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // fill_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // fill_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // fill_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // fill_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // fill_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 52.25, 0.00},  // fill_tile_init/math, n=4, spread=7.2%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // fill_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 123.02, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.59, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 123.01, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 123.60, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 123.01, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 123.56, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.92, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 123.43, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 122.91, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 123.42, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 122.92, 0.00},  // floor_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 123.47, 0.00},  // floor_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // floor_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.91, 0.00},  // floor_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 122.42, 0.00},  // floor_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 122.92, 0.00},  // floor_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 122.53, 0.00},  // floor_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 122.92, 0.00},  // floor_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 108.34, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 108.80, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 108.34, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 108.80, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 108.21, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 108.82, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 108.29, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 108.86, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 108.29, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 108.86, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 108.29, 0.00},  // frac_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 108.80, 0.00},  // frac_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 106.50, 0.00},  // frac_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 108.28, 0.00},  // frac_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 106.50, 0.00},  // frac_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 108.29, 0.00},  // frac_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 106.44, 0.00},  // frac_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 108.29, 0.00},  // frac_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.44, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.92, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 437.46, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 437.91, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 437.44, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 437.84, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.76, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 437.74, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 437.75, 0.00},  // ge_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 437.83, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // ge_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 437.75, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // ge_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 437.75, 0.00},  // ge_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 438.83, 0.00},  // ge_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 437.74, 0.00},  // ge_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ge_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ge_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ge_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 56.50, 0.00},  // ge_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // ge_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ge_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ge_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ge_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.75, 0.00},  // ge_binary_tile_init/math, n=4, spread=10.1%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 714.47, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 755.14, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 714.46, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 755.14, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 714.46, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 755.01, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 714.40, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 754.97, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 714.40, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 754.97, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 714.40, 0.00},  // gelu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 754.97, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 754.02, 0.00},  // gelu_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 714.39, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 754.02, 0.00},  // gelu_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 714.40, 0.00},  // gelu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 753.89, 0.00},  // gelu_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 714.37, 0.00},  // gelu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // gelu_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // gelu_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // gelu_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // gelu_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, spread=4.5%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // gelu_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, spread=4.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // gelu_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.00, 0.00},  // gelu_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // gelu_tile_init/math, n=2, spread=2.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, spread=4.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, spread=4.5%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 55.50, 0.00},  // gelu_tile_init/math, n=2, spread=4.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gelu_tile_init/math, n=4, spread=4.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.50, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 92.11, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 91.49, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 92.11, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 91.49, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 91.92, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 91.99, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 91.99, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 91.93, 0.00},  // gez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 91.14, 0.00},  // gez_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 91.58, 0.00},  // gez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 91.14, 0.00},  // gez_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 91.16, 0.00},  // gez_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 91.59, 0.00},  // gez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.50, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.50, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gez_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gez_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=4, spread=3.9%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gez_tile_init/math, n=4, spread=3.9%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // gez_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // gez_tile_init/math, n=4, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.49, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.88, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 373.51, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 373.86, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 373.49, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 373.73, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.48, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.73, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 373.46, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 373.72, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 373.47, 0.00},  // gt_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 373.52, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // gt_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 373.47, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // gt_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 373.47, 0.00},  // gt_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 374.61, 0.00},  // gt_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 373.46, 0.00},  // gt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // gt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // gt_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // gt_binary_tile_init/math, n=2, spread=4.8%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // gt_binary_tile_init/math, n=4, spread=6.5%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // gt_binary_tile_init/math, n=4, spread=6.5%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // gt_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.25, 0.00},  // gt_binary_tile_init/math, n=4, spread=7.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.30, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.86, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 75.29, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 75.86, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 75.29, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 75.82, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.51, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.93, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 75.51, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 75.92, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 75.51, 0.00},  // gtz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 75.89, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 75.03, 0.00},  // gtz_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 75.50, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 75.03, 0.00},  // gtz_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 75.56, 0.00},  // gtz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 74.86, 0.00},  // gtz_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 75.51, 0.00},  // gtz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // gtz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 63.50, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 63.25, 0.00},  // gtz_tile_init/math, n=4, spread=5.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // gtz_tile_init/math, n=4, spread=3.9%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // gtz_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.50, 0.00},  // gtz_tile_init/math, n=4, spread=2.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.09, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 122.77, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 122.09, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 122.85, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 122.09, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 122.70, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 122.72, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 122.72, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 122.60, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 121.33, 0.00},  // hardsigmoid_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 121.33, 0.00},  // hardsigmoid_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 121.14, 0.00},  // hardsigmoid_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 122.21, 0.00},  // hardsigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 63.50, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.50, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 63.00, 0.00},  // hardsigmoid_tile_init/math, n=4, spread=7.9%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 63.00, 0.00},  // hardsigmoid_tile_init/math, n=4, spread=7.9%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // hardsigmoid_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 62.75, 0.00},  // hardsigmoid_tile_init/math, n=4, spread=7.6%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 138.25, 0.00},  // init_sfpu/math, n=118, spread=4.5%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 135.98, 0.00},  // init_sfpu/math, n=118, spread=3.0%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 138.15, 0.00},  // init_sfpu/math, n=118, spread=4.5%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 135.98, 0.00},  // init_sfpu/math, n=118, spread=3.0%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 138.14, 0.00},  // init_sfpu/math, n=118, spread=4.4%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 137.75, 0.00},  // init_sfpu/math, n=118, spread=4.5%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 137.83, 0.00},  // init_sfpu/math, n=118, spread=3.8%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 135.97, 0.00},  // init_sfpu/math, n=118, spread=3.0%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 137.75, 0.00},  // init_sfpu/math, n=118, spread=3.8%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 135.97, 0.00},  // init_sfpu/math, n=118, spread=3.0%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 137.84, 0.00},  // init_sfpu/math, n=118, spread=3.7%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 137.53, 0.00},  // init_sfpu/math, n=118, spread=4.7%
    {"Float16_b", "UInt16", false, 4, "", Unit::PerCall, 168.00, 0.00},  // init_sfpu/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 137.93, 0.00},  // init_sfpu/math, n=118, spread=3.7%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 135.81, 0.00},  // init_sfpu/math, n=236, spread=3.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 137.87, 0.00},  // init_sfpu/math, n=118, spread=3.7%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 135.81, 0.00},  // init_sfpu/math, n=236, spread=3.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 137.93, 0.00},  // init_sfpu/math, n=118, spread=3.7%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 137.65, 0.00},  // init_sfpu/math, n=236, spread=4.8%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 141.18, 0.00},  // init_sfpu/math, n=28, spread=5.1%
    {"Int32", "UInt16", true, 4, "", Unit::PerCall, 176.00, 0.00},  // init_sfpu/math, n=1
    {"UInt16", "Float32", false, 4, "", Unit::PerCall, 175.00, 0.00},  // init_sfpu/math, n=1
    {"UInt16", "UInt32", false, 4, "", Unit::PerCall, 175.00, 0.00},  // init_sfpu/math, n=1
    {"UInt32", "Float16_b", true, 4, "", Unit::PerCall, 177.00, 0.00},  // init_sfpu/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 201.95, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 203.73, 0.00},  // init_sfpu/pack, n=236, spread=3.8%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 201.94, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 203.74, 0.00},  // init_sfpu/pack, n=236, spread=3.8%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 201.95, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 201.77, 0.00},  // init_sfpu/pack, n=236, spread=3.1%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 202.74, 0.00},  // init_sfpu/pack, n=236, spread=3.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 202.71, 0.00},  // init_sfpu/pack, n=236, spread=3.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 203.61, 0.00},  // init_sfpu/pack, n=236, spread=3.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 202.98, 0.00},  // init_sfpu/pack, n=472, spread=3.5%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 202.98, 0.00},  // init_sfpu/pack, n=472, spread=3.5%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 201.15, 0.00},  // init_sfpu/pack, n=236, spread=3.4%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 202.97, 0.00},  // init_sfpu/pack, n=472, spread=4.0%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 203.05, 0.00},  // init_sfpu/pack, n=44
    {"Bfp8_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.37, 0.00},  // init_sfpu/unpack, n=236, spread=2.3%
    {"Bfp8_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 199.92, 0.00},  // init_sfpu/unpack, n=236
    {"Bfp8_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.37, 0.00},  // init_sfpu/unpack, n=236, spread=2.3%
    {"Bfp8_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 199.93, 0.00},  // init_sfpu/unpack, n=236
    {"Bfp8_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 200.37, 0.00},  // init_sfpu/unpack, n=236, spread=2.3%
    {"Bfp8_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 200.37, 0.00},  // init_sfpu/unpack, n=236, spread=2.3%
    {"Float16_b", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.17, 0.00},  // init_sfpu/unpack, n=236, spread=2.4%
    {"Float16_b", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.89, 0.00},  // init_sfpu/unpack, n=236, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.11, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Float16_b", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.89, 0.00},  // init_sfpu/unpack, n=236
    {"Float16_b", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.14, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Float16_b", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 198.03, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Float32", "Bfp8_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.16, 0.00},  // init_sfpu/unpack, n=236, spread=2.4%
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 205.33, 0.00},  // init_sfpu/unpack, n=236
    {"Float32", "Bfp8_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.94, 0.00},  // init_sfpu/unpack, n=236, spread=2.1%
    {"Float32", "Float16_b", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.14, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=true", Unit::PerCall, 205.33, 0.00},  // init_sfpu/unpack, n=236
    {"Float32", "Float16_b", true, 4, "unpack_to_dest=false", Unit::PerCall, 197.89, 0.00},  // init_sfpu/unpack, n=236, spread=2.1%
    {"Float32", "Float32", false, 4, "unpack_to_dest=false", Unit::PerCall, 198.13, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Float32", "Float32", true, 4, "unpack_to_dest=true", Unit::PerCall, 206.78, 0.00},  // init_sfpu/unpack, n=236
    {"Float32", "Float32", true, 4, "unpack_to_dest=false", Unit::PerCall, 198.06, 0.00},  // init_sfpu/unpack, n=236, spread=2.5%
    {"Int32", "Int32", false, 4, "unpack_to_dest=true", Unit::PerCall, 206.66, 0.00},  // init_sfpu/unpack, n=44
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.44, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.92, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 437.46, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 437.91, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 437.41, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 437.92, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 437.76, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 437.74, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 437.88, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 437.75, 0.00},  // le_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 437.83, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // le_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 437.75, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 439.02, 0.00},  // le_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 437.75, 0.00},  // le_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 438.83, 0.00},  // le_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 437.74, 0.00},  // le_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // le_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // le_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // le_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // le_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // le_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // le_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // le_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // le_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // le_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // le_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // le_binary_tile_init/math, n=4, spread=7.5%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 92.02, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 91.59, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 92.02, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 91.59, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 91.95, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 92.02, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 92.02, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 91.96, 0.00},  // lez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 91.16, 0.00},  // lez_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 91.16, 0.00},  // lez_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 91.14, 0.00},  // lez_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 91.60, 0.00},  // lez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // lez_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // lez_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // lez_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // lez_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // lez_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // lez_tile_init/math, n=4, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 274.23, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 370.72, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 274.23, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 370.72, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 274.23, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 370.67, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 274.37, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 370.71, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 274.37, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 370.71, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 274.37, 0.00},  // log1p_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 370.72, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 369.64, 0.00},  // log1p_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 274.36, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 369.62, 0.00},  // log1p_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 274.37, 0.00},  // log1p_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 369.70, 0.00},  // log1p_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 274.37, 0.00},  // log1p_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // log1p_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 68.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.75, 0.00},  // log1p_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log1p_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 67.00, 0.00},  // log1p_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // log1p_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 154.28, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 154.66, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 154.27, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 154.66, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 154.27, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 154.42, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 154.37, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 154.80, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 154.37, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 154.80, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 154.37, 0.00},  // log_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 154.47, 0.00},  // log_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 153.30, 0.00},  // log_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 154.36, 0.00},  // log_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 153.30, 0.00},  // log_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 154.37, 0.00},  // log_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 153.27, 0.00},  // log_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 154.37, 0.00},  // log_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 69.00, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, spread=2.3%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // log_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2, spread=2.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // log_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.50, 0.00},  // log_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // log_tile_init/math, n=4, spread=3.0%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 65.66, 0.00},  // logical_not_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // logical_not_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.49, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.88, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 373.51, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 373.86, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 373.49, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 373.73, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 373.48, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 373.73, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 373.46, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 373.72, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 373.47, 0.00},  // lt_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 373.52, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // lt_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 373.47, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 374.72, 0.00},  // lt_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 373.47, 0.00},  // lt_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 374.61, 0.00},  // lt_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 373.46, 0.00},  // lt_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 57.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // lt_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // lt_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // lt_binary_tile_init/math, n=2, spread=4.8%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // lt_binary_tile_init/math, n=4, spread=6.5%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.50, 0.00},  // lt_binary_tile_init/math, n=4, spread=6.5%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // lt_binary_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.25, 0.00},  // lt_binary_tile_init/math, n=4, spread=7.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.53, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.95, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 75.52, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 75.96, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 75.52, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 75.89, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 75.54, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 75.96, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 75.54, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 75.96, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 75.54, 0.00},  // ltz_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 75.89, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 75.09, 0.00},  // ltz_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 75.53, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 75.09, 0.00},  // ltz_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 75.54, 0.00},  // ltz_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 75.02, 0.00},  // ltz_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 75.54, 0.00},  // ltz_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // ltz_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // ltz_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // ltz_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // ltz_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // ltz_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // ltz_tile_init/math, n=4, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.54, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.03, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.55, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.02, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.54, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.71, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.60, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.58, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.60, 0.00},  // mul_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // mul_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // mul_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // mul_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // mul_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.59, 0.00},  // mul_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=7.8%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // mul_binary_tile_init/math, n=4, spread=10.8%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // mul_binary_tile_init/math, n=4, spread=10.8%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // mul_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // mul_binary_tile_init/math, n=4, spread=12.2%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 287.41, 0.00},  // mul_int_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 72.50, 0.00},  // mul_int_tile_init/math, n=2, spread=3.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.33, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.12, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.08, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.20, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.33, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.12, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.09, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.20, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.33, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 19.12, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.23, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.27, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.23, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.11, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.03, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.17, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.11, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.20, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.17, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.16, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.22, 0.00},  // mul_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.31, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.28, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.22, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.11, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.03, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.17, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.19, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.11, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.03, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.11, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.20, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.16, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerTile, 18.95, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerTile, 36.16, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerTile, 60.14, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerTile, 84.22, 0.00},  // mul_tiles/math, n=2
    {"Float32", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerTile, 18.31, 0.00},  // mul_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // mul_tiles/unpack, n=4
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.22, 0.00},  // mul_tiles/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.19, 0.00},  // mul_tiles/unpack, n=4
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.24, 0.00},  // mul_tiles/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.19, 0.00},  // mul_tiles/unpack, n=4
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.19, 0.00},  // mul_tiles/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // mul_tiles/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // mul_tiles/unpack, n=8
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.12, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.07, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.79, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.07, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.79, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.07, 0.00},  // mul_tiles/unpack, n=8
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.78, 0.00},  // mul_tiles/unpack, n=8
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
    {"Bfp8_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
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
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 90.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "math_fidelity=LoFi", Unit::PerCall, 87.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi2", Unit::PerCall, 85.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi3", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=HiFi4", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "math_fidelity=LoFi", Unit::PerCall, 86.00, 0.00},  // mul_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 65.00, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 64.50, 0.00},  // mul_tiles_init/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // mul_tiles_init/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.48, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.95, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 469.49, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 469.94, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 469.48, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 469.95, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 469.79, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 469.77, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 469.91, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 469.78, 0.00},  // ne_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 469.87, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // ne_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 469.78, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 471.02, 0.00},  // ne_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 469.78, 0.00},  // ne_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 470.83, 0.00},  // ne_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 469.77, 0.00},  // ne_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ne_binary_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ne_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 53.00, 0.00},  // ne_binary_tile_init/math, n=2, spread=5.7%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // ne_binary_tile_init/math, n=2, spread=5.5%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // ne_binary_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 52.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=4.8%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 56.50, 0.00},  // ne_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=4.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=6.5%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 56.00, 0.00},  // ne_binary_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ne_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.25, 0.00},  // ne_binary_tile_init/math, n=4, spread=9.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 53.50, 0.00},  // ne_binary_tile_init/math, n=2, spread=6.5%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // ne_binary_tile_init/math, n=4, spread=7.5%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.28, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.83, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 42.27, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 42.83, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 42.27, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 42.76, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 42.75, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 42.75, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 42.26, 0.00},  // negative_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 42.72, 0.00},  // negative_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 41.09, 0.00},  // negative_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 42.25, 0.00},  // negative_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 41.09, 0.00},  // negative_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 42.26, 0.00},  // negative_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 41.28, 0.00},  // negative_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 42.26, 0.00},  // negative_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // negative_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // negative_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // negative_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // negative_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // negative_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // negative_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.25, 0.00},  // negative_tile_init/math, n=4, spread=10.8%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.74, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 58.30, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 58.74, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 58.30, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 58.67, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 58.72, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 58.72, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 58.69, 0.00},  // nez_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // nez_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 58.30, 0.00},  // nez_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 57.72, 0.00},  // nez_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 57.70, 0.00},  // nez_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 58.31, 0.00},  // nez_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.00, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // nez_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // nez_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 65.25, 0.00},  // nez_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 65.50, 0.00},  // nez_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 63.75, 0.00},  // nez_tile_init/math, n=4, spread=2.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 29.97, 0.00},  // pack_tile/pack, n=246
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 29.20, 0.00},  // pack_tile/pack, n=246
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 38.93, 0.00},  // pack_tile/pack, n=246
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 29.07, 0.00},  // pack_tile/pack, n=246
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 71.04, 0.00},  // pack_tile/pack, n=246
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 43.85, 0.00},  // pack_tile/pack, n=246
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 29.96, 0.00},  // pack_tile/pack, n=250
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 29.19, 0.00},  // pack_tile/pack, n=250
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 29.85, 0.00},  // pack_tile/pack, n=250
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 29.06, 0.00},  // pack_tile/pack, n=250
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 39.10, 0.00},  // pack_tile/pack, n=250
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 43.98, 0.00},  // pack_tile/pack, n=250
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 29.96, 0.00},  // pack_tile/pack, n=250
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 29.21, 0.00},  // pack_tile/pack, n=488
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 29.84, 0.00},  // pack_tile/pack, n=250
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 29.08, 0.00},  // pack_tile/pack, n=488
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 39.10, 0.00},  // pack_tile/pack, n=250
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 43.98, 0.00},  // pack_tile/pack, n=488
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 44.60, 0.00},  // pack_tile/pack, n=44
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.87, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 3217.22, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 3216.88, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 3217.20, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 3216.87, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 3217.14, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.82, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 3217.02, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 3216.80, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 3217.02, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 3216.81, 0.00},  // power_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 3217.27, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 3218.09, 0.00},  // power_binary_tile/math, n=4
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 3216.81, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 3218.16, 0.00},  // power_binary_tile/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 3216.81, 0.00},  // power_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 3217.94, 0.00},  // power_binary_tile/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 3216.81, 0.00},  // power_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 53.50, 0.00},  // power_binary_tile_init/math, n=4, spread=4.7%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=4
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 53.50, 0.00},  // power_binary_tile_init/math, n=4, spread=4.7%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=8
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.00, 0.00},  // power_binary_tile_init/math, n=8
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 52.00, 0.00},  // power_binary_tile_init/math, n=4, spread=5.8%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 51.50, 0.00},  // power_binary_tile_init/math, n=8, spread=8.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 761.66, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1228.22, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 761.65, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 1228.22, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 761.55, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 1228.12, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 761.81, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 1228.47, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 761.81, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 1228.47, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 761.81, 0.00},  // power_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 1228.30, 0.00},  // power_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 1239.50, 0.00},  // power_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 761.80, 0.00},  // power_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 1239.50, 0.00},  // power_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 761.81, 0.00},  // power_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 1239.25, 0.00},  // power_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 761.81, 0.00},  // power_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // power_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // power_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // power_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 63.00, 0.00},  // power_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 63.00, 0.00},  // power_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 64.75, 0.00},  // power_tile_init/math, n=4, spread=2.7%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 64.75, 0.00},  // power_tile_init/math, n=4, spread=2.7%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 64.50, 0.00},  // power_tile_init/math, n=2, spread=2.3%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 64.00, 0.00},  // power_tile_init/math, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 76.16, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 56.91, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 76.15, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 56.91, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 76.15, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 56.99, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 76.09, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 56.91, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 76.09, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 56.91, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 76.10, 0.00},  // recip_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 56.71, 0.00},  // recip_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 58.06, 0.00},  // recip_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 76.08, 0.00},  // recip_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 58.06, 0.00},  // recip_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 76.10, 0.00},  // recip_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 58.22, 0.00},  // recip_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 76.09, 0.00},  // recip_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 106.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 106.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 78.00, 0.00},  // recip_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 109.00, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 106.50, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 106.50, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 112.50, 0.00},  // recip_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 106.75, 0.00},  // recip_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 106.75, 0.00},  // recip_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 78.50, 0.00},  // recip_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 109.50, 0.00},  // recip_tile_init/math, n=4, spread=3.2%
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
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerCall, 78.00, 0.00},  // reduce_init/math, n=1
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
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerCall, 132.00, 0.00},  // reduce_init/math, n=1
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
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, spread=3.9%
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, spread=5.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, spread=11.0%
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, spread=3.9%
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, spread=5.7%
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, spread=11.0%
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 64.50, 0.00},  // reduce_init/pack, n=4, spread=3.9%
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 61.50, 0.00},  // reduce_init/pack, n=4, spread=5.7%
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 61.25, 0.00},  // reduce_init/pack, n=4, spread=11.0%
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, spread=16.2%
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, spread=9.2%
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.12, 0.00},  // reduce_init/pack, n=8, spread=11.8%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, spread=16.2%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, spread=9.2%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, spread=11.6%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, spread=16.2%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, spread=9.2%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, spread=11.6%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, spread=16.2%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 59.50, 0.00},  // reduce_init/pack, n=8, spread=7.6%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, spread=11.6%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn", Unit::PerCall, 58.50, 0.00},  // reduce_init/pack, n=8, spread=16.2%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, spread=9.2%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, spread=11.6%
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn", Unit::PerCall, 59.38, 0.00},  // reduce_init/pack, n=8, spread=14.5%
    {"Float32", "Float32", false, 4, "mathop=ReduceRow", Unit::PerCall, 58.62, 0.00},  // reduce_init/pack, n=8, spread=9.2%
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar", Unit::PerCall, 58.25, 0.00},  // reduce_init/pack, n=8, spread=11.6%
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 73.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 73.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 69.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 68.50, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 73.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.00, 0.00},  // reduce_init/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.25, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.25, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerCall, 67.25, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerCall, 67.00, 0.00},  // reduce_init/unpack, n=4
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerCall, 71.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerCall, 70.50, 0.00},  // reduce_init/unpack, n=4, spread=2.1%
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.03, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 38.11, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.94, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.03, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 38.11, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.94, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.41, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 19.11, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 38.11, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.58, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.53, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.03, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.88, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 44.06, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.94, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.94, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.12, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.42, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.84, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.48, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.83, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 19.27, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 19.20, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 18.89, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 37.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 61.95, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 85.97, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 19.31, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 85.47, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 28.06, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 40.00, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 52.03, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 15.16, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi2", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi3", Unit::PerTile, 43.91, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=HiFi4", Unit::PerTile, 43.92, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max;math_fidelity=LoFi", Unit::PerTile, 43.83, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi2", Unit::PerTile, 72.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi3", Unit::PerTile, 102.81, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=HiFi4", Unit::PerTile, 132.98, 0.00},  // reduce_tile/math, n=2
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum;math_fidelity=LoFi", Unit::PerTile, 43.97, 0.00},  // reduce_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.97, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.98, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.04, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 43.69, 0.00},  // reduce_tile/unpack, n=4
    {"Bfp8_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 43.65, 0.00},  // reduce_tile/unpack, n=4
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 37.00, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 37.00, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 37.00, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 36.99, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.07, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 51.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float16_b", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 51.32, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Bfp8_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.32, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float16_b", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.34, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceColumn;reduce_pool_type=Sum", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Max", Unit::PerTile, 39.91, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceRow;reduce_pool_type=Sum", Unit::PerTile, 37.70, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Max", Unit::PerTile, 71.31, 0.00},  // reduce_tile/unpack, n=8
    {"Float32", "Float32", false, 4, "mathop=ReduceScalar;reduce_pool_type=Sum", Unit::PerTile, 71.34, 0.00},  // reduce_tile/unpack, n=8
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.12, 0.00},  // rounding_op_tile_init/math, n=8, spread=7.5%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.12, 0.00},  // rounding_op_tile_init/math, n=8, spread=7.5%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.12, 0.00},  // rounding_op_tile_init/math, n=8
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.62, 0.00},  // rounding_op_tile_init/math, n=8, spread=6.6%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.62, 0.00},  // rounding_op_tile_init/math, n=8, spread=6.6%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 54.62, 0.00},  // rounding_op_tile_init/math, n=8, spread=8.0%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.12, 0.00},  // rounding_op_tile_init/math, n=16, spread=9.2%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // rounding_op_tile_init/math, n=16, spread=9.4%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // rounding_op_tile_init/math, n=8
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // rounding_op_tile_init/math, n=16, spread=11.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 258.27, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 250.69, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 258.26, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 250.78, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 258.26, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 250.67, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 250.69, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 250.69, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 250.66, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 249.41, 0.00},  // rsqrt_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 258.32, 0.00},  // rsqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 70.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 69.50, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.75, 0.00},  // rsqrt_tile_init/math, n=4, spread=4.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 68.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.75, 0.00},  // rsqrt_tile_init/math, n=4, spread=4.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // rsqrt_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // rsqrt_tile_init/math, n=4, spread=2.2%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 410.34, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 402.83, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 410.34, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 402.83, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 410.34, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 402.85, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 402.93, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 402.92, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 402.83, 0.00},  // selu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 401.22, 0.00},  // selu_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 401.22, 0.00},  // selu_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 401.22, 0.00},  // selu_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 410.32, 0.00},  // selu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.50, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // selu_tile_init/math, n=2, spread=6.4%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // selu_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // selu_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // selu_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // selu_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // selu_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // selu_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // selu_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // selu_tile_init/math, n=4, spread=11.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.12, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 426.88, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 314.29, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 426.88, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 314.29, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 426.78, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 314.25, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 426.93, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 314.27, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 426.92, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 314.28, 0.00},  // sigmoid_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 426.72, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 425.52, 0.00},  // sigmoid_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 314.27, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 425.52, 0.00},  // sigmoid_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 314.28, 0.00},  // sigmoid_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 425.52, 0.00},  // sigmoid_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 314.27, 0.00},  // sigmoid_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.50, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 59.50, 0.00},  // sigmoid_tile_init/math, n=2, spread=4.2%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 59.50, 0.00},  // sigmoid_tile_init/math, n=2, spread=4.2%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // sigmoid_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // sigmoid_tile_init/math, n=2, spread=4.3%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // sigmoid_tile_init/math, n=2, spread=4.3%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // sigmoid_tile_init/math, n=2, spread=2.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.75, 0.00},  // sigmoid_tile_init/math, n=4, spread=7.5%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.75, 0.00},  // sigmoid_tile_init/math, n=4, spread=7.5%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // sigmoid_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // sigmoid_tile_init/math, n=4, spread=4.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 90.34, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 90.83, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 90.34, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 90.83, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 90.34, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 90.85, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 90.93, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 90.92, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 90.83, 0.00},  // sign_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 89.28, 0.00},  // sign_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 89.28, 0.00},  // sign_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 89.14, 0.00},  // sign_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 90.32, 0.00},  // sign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // sign_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // sign_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // sign_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // sign_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // sign_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // sign_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // sign_tile_init/math, n=4, spread=11.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.83, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 20.00, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.82, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 20.00, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.82, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 19.96, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 20.00, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 20.22, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 19.99, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 20.22, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 19.99, 0.00},  // signbit_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 20.02, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 20.95, 0.00},  // signbit_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 19.99, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 20.95, 0.00},  // signbit_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 19.99, 0.00},  // signbit_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 20.75, 0.00},  // signbit_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 19.99, 0.00},  // signbit_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 71.00, 0.00},  // signbit_tile_init/math, n=2, spread=4.2%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 71.00, 0.00},  // signbit_tile_init/math, n=2, spread=4.2%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 70.00, 0.00},  // signbit_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 71.00, 0.00},  // signbit_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 70.25, 0.00},  // signbit_tile_init/math, n=4, spread=3.9%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 70.25, 0.00},  // signbit_tile_init/math, n=4, spread=3.9%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 70.50, 0.00},  // signbit_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 70.25, 0.00},  // signbit_tile_init/math, n=4, spread=3.9%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 347.25, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 347.88, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 347.24, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 347.88, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 347.24, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 347.95, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 347.23, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 347.97, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 347.23, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 347.97, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 347.23, 0.00},  // silu_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 347.77, 0.00},  // silu_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 345.64, 0.00},  // silu_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 347.22, 0.00},  // silu_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 345.64, 0.00},  // silu_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 347.23, 0.00},  // silu_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 345.67, 0.00},  // silu_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 347.23, 0.00},  // silu_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // silu_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // silu_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // silu_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, spread=9.4%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, spread=9.4%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // silu_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.00, 0.00},  // silu_tile_init/math, n=4, spread=11.3%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 240.22, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 243.66, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 240.21, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 243.66, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 240.21, 0.00},  // sin_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 243.64, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 240.17, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 243.69, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 240.16, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 243.69, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 240.17, 0.00},  // sin_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 243.63, 0.00},  // sin_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 241.58, 0.00},  // sin_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 240.16, 0.00},  // sin_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 241.58, 0.00},  // sin_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 240.17, 0.00},  // sin_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 241.66, 0.00},  // sin_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 240.17, 0.00},  // sin_tile/math, n=1
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // sin_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.25, 0.00},  // sin_tile_init/math, n=4, spread=3.3%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sin_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // sin_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 139.27, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 139.84, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 139.26, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 139.85, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 139.26, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 139.74, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 139.25, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 139.90, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 139.20, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 139.89, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 139.24, 0.00},  // softsign_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 139.69, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 137.42, 0.00},  // softsign_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 139.24, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 137.42, 0.00},  // softsign_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 139.24, 0.00},  // softsign_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 137.38, 0.00},  // softsign_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 139.24, 0.00},  // softsign_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // softsign_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 58.50, 0.00},  // softsign_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 55.00, 0.00},  // softsign_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 57.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.5%
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 56.00, 0.00},  // softsign_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=5.2%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.4%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 57.50, 0.00},  // softsign_tile_init/math, n=2, spread=2.6%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.4%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, spread=8.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 58.00, 0.00},  // softsign_tile_init/math, n=2, spread=3.4%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 56.50, 0.00},  // softsign_tile_init/math, n=4, spread=4.4%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 226.22, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 218.77, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 226.21, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 218.77, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 226.21, 0.00},  // sqrt_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 218.67, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 226.17, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 218.80, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 226.16, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 218.80, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 226.17, 0.00},  // sqrt_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 218.64, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 217.41, 0.00},  // sqrt_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 226.16, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 217.41, 0.00},  // sqrt_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 226.17, 0.00},  // sqrt_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 217.11, 0.00},  // sqrt_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 226.17, 0.00},  // sqrt_tile/math, n=1
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // sqrt_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // sqrt_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // sqrt_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 43.33, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 43.80, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 43.32, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 43.80, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 43.32, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 43.82, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 43.29, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 43.90, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 43.29, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 43.89, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 43.29, 0.00},  // square_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 43.80, 0.00},  // square_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 50.38, 0.00},  // square_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 43.28, 0.00},  // square_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 50.38, 0.00},  // square_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 43.29, 0.00},  // square_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 50.30, 0.00},  // square_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 43.29, 0.00},  // square_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // square_tile_init/math, n=2, spread=6.4%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 55.00, 0.00},  // square_tile_init/math, n=2, spread=7.3%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // square_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 54.50, 0.00},  // square_tile_init/math, n=2, spread=6.4%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 55.00, 0.00},  // square_tile_init/math, n=2, spread=7.3%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // square_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 52.75, 0.00},  // square_tile_init/math, n=4, spread=10.0%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 51.00, 0.00},  // square_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 53.25, 0.00},  // square_tile_init/math, n=4, spread=10.8%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.54, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 181.03, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 180.55, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 181.02, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 180.54, 0.00},  // sub_binary_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 180.82, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 180.60, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 180.65, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 180.91, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 180.60, 0.00},  // sub_binary_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 180.80, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // sub_binary_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 180.62, 0.00},  // sub_binary_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 180.59, 0.00},  // sub_binary_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 180.58, 0.00},  // sub_binary_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 180.59, 0.00},  // sub_binary_tile/math, n=1
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 370.89, 0.00},  // sub_binary_tile/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=7.8%
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 47.50, 0.00},  // sub_binary_tile_init/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.9%
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 51.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=7.8%
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, spread=5.0%
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 48.75, 0.00},  // sub_binary_tile_init/math, n=4, spread=10.8%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 49.25, 0.00},  // sub_binary_tile_init/math, n=4, spread=9.6%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 50.00, 0.00},  // sub_binary_tile_init/math, n=2, spread=6.0%
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 49.00, 0.00},  // sub_binary_tile_init/math, n=4, spread=12.2%
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 50.50, 0.00},  // sub_binary_tile_init/math, n=2, spread=5.0%
    {"Int32", "Int32", false, 4, "", Unit::PerTile, 41.41, 0.00},  // sub_int_tile/math, n=2
    {"Int32", "Int32", false, 4, "", Unit::PerCall, 61.00, 0.00},  // sub_int_tile_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 19.12, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.20, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 19.12, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 18.20, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 19.12, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 18.27, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 18.23, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 18.31, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 18.17, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 18.95, 0.00},  // sub_tiles/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 18.31, 0.00},  // sub_tiles/math, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.17, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.23, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 37.17, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 37.23, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 37.17, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 37.17, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 37.14, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 40.09, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 40.09, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 37.78, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 40.09, 0.00},  // sub_tiles/unpack, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 37.83, 0.00},  // sub_tiles/unpack, n=2
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 90.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 87.00, 0.00},  // sub_tiles_init/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 86.00, 0.00},  // sub_tiles_init/math, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerCall, 65.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerCall, 64.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerCall, 65.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerCall, 65.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 66.00, 0.00},  // sub_tiles_init/unpack, n=1
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 322.22, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 482.69, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 322.21, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 482.69, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 322.21, 0.00},  // tan_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 482.67, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 322.17, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 482.75, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 322.16, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 482.75, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 322.17, 0.00},  // tan_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 482.66, 0.00},  // tan_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 481.52, 0.00},  // tan_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 322.16, 0.00},  // tan_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 481.52, 0.00},  // tan_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 322.17, 0.00},  // tan_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 481.59, 0.00},  // tan_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 322.17, 0.00},  // tan_tile/math, n=1
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // tan_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 67.75, 0.00},  // tan_tile_init/math, n=4, spread=4.1%
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tan_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // tan_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 136.02, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 362.80, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 136.01, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 362.80, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 136.01, 0.00},  // tanh_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 362.73, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 136.04, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 362.80, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 136.04, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 362.80, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 136.04, 0.00},  // tanh_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 362.77, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 361.81, 0.00},  // tanh_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 136.03, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 361.81, 0.00},  // tanh_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 136.04, 0.00},  // tanh_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 361.64, 0.00},  // tanh_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 136.04, 0.00},  // tanh_tile/math, n=1
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
    {"Float16_b", "Float32", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=4
    {"Float32", "Float16_b", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Float16_b", true, 4, "", Unit::PerCall, 68.50, 0.00},  // tanh_tile_init/math, n=4
    {"Float32", "Float32", false, 4, "", Unit::PerCall, 69.00, 0.00},  // tanh_tile_init/math, n=2
    {"Float32", "Float32", true, 4, "", Unit::PerCall, 67.50, 0.00},  // tanh_tile_init/math, n=4, spread=3.7%
    {"Bfp8_b", "Bfp8_b", false, 4, "", Unit::PerTile, 98.92, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Bfp8_b", true, 4, "", Unit::PerTile, 99.50, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float16_b", false, 4, "", Unit::PerTile, 98.91, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float16_b", true, 4, "", Unit::PerTile, 99.50, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float32", false, 4, "", Unit::PerTile, 98.88, 0.00},  // trunc_tile/math, n=1
    {"Bfp8_b", "Float32", true, 4, "", Unit::PerTile, 99.46, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Bfp8_b", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Bfp8_b", true, 4, "", Unit::PerTile, 99.33, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float16_b", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float16_b", true, 4, "", Unit::PerTile, 99.33, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float32", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
    {"Float16_b", "Float32", true, 4, "", Unit::PerTile, 99.38, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Bfp8_b", true, 4, "", Unit::PerTile, 98.33, 0.00},  // trunc_tile/math, n=2
    {"Float32", "Bfp8_b", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Float16_b", true, 4, "", Unit::PerTile, 98.31, 0.00},  // trunc_tile/math, n=2
    {"Float32", "Float16_b", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
    {"Float32", "Float32", true, 4, "", Unit::PerTile, 98.44, 0.00},  // trunc_tile/math, n=2
    {"Float32", "Float32", false, 4, "", Unit::PerTile, 98.82, 0.00},  // trunc_tile/math, n=1
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
    {"init_sfpu", {}, EngineCost{1386, 22}, EngineCost{1343, 24}, EngineCost{1367, 19}},
    {"invoke_sfpi", {}, {}, EngineCost{0, 0}, {}},
    {"le_binary_tile", {}, {}, EngineCost{1408, 18}, {}},
    {"le_binary_tile_init", {}, {}, EngineCost{1426, 18}, {}},
    {"lez_tile", {}, {}, EngineCost{1444, 18}, {}},
    {"lez_tile_init", {}, {}, EngineCost{1462, 18}, {}},
    {"lez_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"load_from_l1", EngineCost{0, 0}, {}, {}, {}},
    {"log1p_tile", {}, {}, EngineCost{1480, 18}, {}},
    {"log1p_tile_init", {}, {}, EngineCost{1498, 18}, {}},
    {"log_tile", {}, {}, EngineCost{1516, 18}, {}},
    {"log_tile_init", {}, {}, EngineCost{1534, 18}, {}},
    {"logical_not_tile", {}, {}, EngineCost{1552, 1}, {}},
    {"logical_not_tile_init", {}, {}, EngineCost{1553, 1}, {}},
    {"lt_binary_tile", {}, {}, EngineCost{1554, 18}, {}},
    {"lt_binary_tile_init", {}, {}, EngineCost{1572, 18}, {}},
    {"ltz_tile", {}, {}, EngineCost{1590, 18}, {}},
    {"ltz_tile_init", {}, {}, EngineCost{1608, 18}, {}},
    {"ltz_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"matmul_block", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"matmul_tiles", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mem_zeros_base", {}, {}, {}, {}},
    {"mem_zeros_size", {}, {}, {}, {}},
    {"mm_block_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"mm_block_init_short", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mm_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"mm_init_short", {}, EngineCost{0, 0}, EngineCost{0, 0}, {}},
    {"mul_binary_tile", {}, {}, EngineCost{1626, 18}, {}},
    {"mul_binary_tile_init", {}, {}, EngineCost{1644, 18}, {}},
    {"mul_int_tile", {}, {}, EngineCost{1662, 1}, {}},
    {"mul_int_tile_init", {}, {}, EngineCost{1663, 1}, {}},
    {"mul_tiles", {}, EngineCost{1724, 18}, EngineCost{1664, 60}, {}},
    {"mul_tiles_init", {}, EngineCost{1802, 18}, EngineCost{1742, 60}, {}},
    {"mul_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"my_logical_x_", {}, {}, {}, {}},
    {"my_logical_y_", {}, {}, {}, {}},
    {"my_x", {}, {}, {}, {}},
    {"my_y", {}, {}, {}, {}},
    {"ne_binary_tile", {}, {}, EngineCost{1820, 18}, {}},
    {"ne_binary_tile_init", {}, {}, EngineCost{1838, 18}, {}},
    {"negative_tile", {}, {}, EngineCost{1856, 18}, {}},
    {"negative_tile_init", {}, {}, EngineCost{1874, 18}, {}},
    {"negative_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"nez_tile", {}, {}, EngineCost{1892, 18}, {}},
    {"nez_tile_init", {}, {}, EngineCost{1910, 18}, {}},
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
    {"pack_tile", {}, {}, {}, EngineCost{1928, 19}},
    {"pack_tile_block", {}, {}, {}, EngineCost{0, 0}},
    {"pack_untilize_init", {}, EngineCost{0, 0}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"pack_untilize_uninit", {}, {}, {}, EngineCost{0, 0}},
    {"power_binary_tile", {}, {}, EngineCost{1947, 18}, {}},
    {"power_binary_tile_init", {}, {}, EngineCost{1965, 18}, {}},
    {"power_tile", {}, {}, EngineCost{1983, 18}, {}},
    {"power_tile_init", {}, {}, EngineCost{2001, 18}, {}},
    {"rand_tile", {}, {}, EngineCost{0, 0}, {}},
    {"rand_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"recip_tile", {}, {}, EngineCost{2019, 18}, {}},
    {"recip_tile_init", {}, {}, EngineCost{2037, 18}, {}},
    {"reduce_init", {}, EngineCost{2262, 54}, EngineCost{2055, 180}, EngineCost{2235, 27}},
    {"reduce_tile", {}, EngineCost{2496, 54}, EngineCost{2316, 180}, {}},
    {"reduce_uninit", {}, {}, EngineCost{0, 0}, EngineCost{0, 0}},
    {"reinterpret_cast", {}, {}, {}, {}},
    {"relu_tile", {}, {}, EngineCost{0, 0}, {}},
    {"relu_tile_init", {}, {}, EngineCost{0, 0}, {}},
    {"relu_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"remote_sram_write_u32", EngineCost{0, 0}, {}, {}, {}},
    {"rounding_op_tile_init", {}, {}, EngineCost{2550, 18}, {}},
    {"rsqrt_tile", {}, {}, EngineCost{2568, 18}, {}},
    {"rsqrt_tile_init", {}, {}, EngineCost{2586, 18}, {}},
    {"selu_tile", {}, {}, EngineCost{2604, 18}, {}},
    {"selu_tile_init", {}, {}, EngineCost{2622, 18}, {}},
    {"sfpu_reduce", {}, {}, EngineCost{0, 0}, {}},
    {"sfpu_reduce_init", {}, {}, EngineCost{0, 0}, {}},
    {"sigmoid_tile", {}, {}, EngineCost{2640, 18}, {}},
    {"sigmoid_tile_init", {}, {}, EngineCost{2658, 18}, {}},
    {"sign_tile", {}, {}, EngineCost{2676, 18}, {}},
    {"sign_tile_init", {}, {}, EngineCost{2694, 18}, {}},
    {"signbit_tile", {}, {}, EngineCost{2712, 18}, {}},
    {"signbit_tile_init", {}, {}, EngineCost{2730, 18}, {}},
    {"silu_tile", {}, {}, EngineCost{2748, 18}, {}},
    {"silu_tile_init", {}, {}, EngineCost{2766, 18}, {}},
    {"sin_tile", {}, {}, EngineCost{2784, 18}, {}},
    {"sin_tile_init", {}, {}, EngineCost{2802, 18}, {}},
    {"softsign_tile", {}, {}, EngineCost{2820, 18}, {}},
    {"softsign_tile_init", {}, {}, EngineCost{2838, 18}, {}},
    {"sqrt_tile", {}, {}, EngineCost{2856, 18}, {}},
    {"sqrt_tile_init", {}, {}, EngineCost{2874, 18}, {}},
    {"square_tile", {}, {}, EngineCost{2892, 18}, {}},
    {"square_tile_init", {}, {}, EngineCost{2910, 18}, {}},
    {"store_to_l1", EngineCost{0, 0}, {}, {}, {}},
    {"sub_binary_tile", {}, {}, EngineCost{2928, 19}, {}},
    {"sub_binary_tile_init", {}, {}, EngineCost{2947, 19}, {}},
    {"sub_int_tile", {}, {}, EngineCost{2966, 1}, {}},
    {"sub_int_tile_init", {}, {}, EngineCost{2967, 1}, {}},
    {"sub_tiles", {}, EngineCost{2986, 18}, EngineCost{2968, 18}, {}},
    {"sub_tiles_init", {}, EngineCost{3022, 18}, EngineCost{3004, 18}, {}},
    {"sub_unary_tile", {}, {}, EngineCost{0, 0}, {}},
    {"sub_unary_tile_int32", {}, {}, EngineCost{0, 0}, {}},
    {"tan_tile", {}, {}, EngineCost{3040, 18}, {}},
    {"tan_tile_init", {}, {}, EngineCost{3058, 18}, {}},
    {"tanh_tile", {}, {}, EngineCost{3076, 18}, {}},
    {"tanh_tile_init", {}, {}, EngineCost{3094, 18}, {}},
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
    {"trunc_tile", {}, {}, EngineCost{3112, 18}, {}},
    {"typecast_tile", {}, {}, EngineCost{3130, 2}, {}},
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
