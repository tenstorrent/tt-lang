// RUN: ttlang-opcost-test | FileCheck %s

// The operation-cost library, checked against whatever table is compiled into
// it. No IR: the table is the input, and the checks derive their probes from it
// rather than naming an operation, a format or a cycle count -- see
// test/lib/Analysis/OpCost/OpCostTest.cpp for why none of the measurements are
// pinned here.
//
// What the table covers is a separate question, reported by llk-perf's
// coverage_report.py in tt-lang-ops-and-models, where the sweeps live.

// CHECK: PASS every measured row is reachable by its own key
// CHECK: PASS knob matching needs every knob a row names, and no more
// CHECK: PASS formats and face count are part of the key, dstSync is not
// CHECK: PASS a slot with no measurements answers nothing
// CHECK: PASS the operation list, the predicates and the counts agree
// CHECK: PASS an unknown operation answers nothing on every engine
// CHECK: PASS an architecture with no table answers nothing
// CHECK: opcost: all checks passed
