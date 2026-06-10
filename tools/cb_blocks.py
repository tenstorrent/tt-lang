# Per-branch CB credit counts in TTL kernel C++. Splits kernel_main into
# top-level if-blocks (the SPMD role guards) and reports per-CB
# wait/pop/reserve/push per block. Loops multiply by literal trip count.
import re
import sys
from collections import defaultdict

OPS = ("wait_front", "pop_front", "reserve_back", "push_back")
FOR_RE = re.compile(
    r"for\s*\(\s*\w+\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\d+)\s*;")
CB_RE = re.compile(r"(cb_ctarg_\d+)\.(%s)\(" % "|".join(OPS))


def analyze(path):
    depth = 0
    block = -1
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    mult = {0: 1}
    pending_trip = None
    in_main = False
    for line in open(path):
        if "kernel_main" in line:
            in_main = True
        if not in_main:
            continue
        m = FOR_RE.search(line)
        cur = mult.get(depth, 1)
        for cm in CB_RE.finditer(line):
            counts[block][cm.group(1)][cm.group(2)] += cur
        if m:
            pending_trip = max(0, int(m.group(3)) - int(m.group(2)))
        opens, closes = line.count("{"), line.count("}")
        for _ in range(opens):
            if depth == 1:
                block += 1
            depth += 1
            trip = pending_trip if pending_trip else 1
            mult[depth] = mult.get(depth - 1, 1) * trip
            pending_trip = None
        depth -= closes
        if in_main and depth <= 0 and block >= 0:
            break
    return counts


def main(paths):
    per = {p: analyze(p) for p in paths}
    all_blocks = sorted({b for c in per.values() for b in c})
    for b in all_blocks:
        cbs = sorted({cb for c in per.values() for cb in c.get(b, {})},
                     key=lambda s: int(s.rsplit("_", 1)[1]))
        if not cbs:
            continue
        print(f"=== block {b}")
        for cb in cbs:
            row = cb.ljust(14)
            for p in paths:
                c = per[p].get(b, {}).get(cb, {})
                row += ("%d/%d/%d/%d" % tuple(c.get(o, 0) for o in OPS)).ljust(16)
            print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
