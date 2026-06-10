# Count CB credit ops per cb_ctarg per kernel file. Loops multiply the static
# count by the literal trip count when the enclosing fors are literal-bound.
import re
import sys
from collections import defaultdict

OPS = ("wait_front", "pop_front", "reserve_back", "push_back")
FOR_RE = re.compile(
    r"for\s*\(\s*\w+\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\s*\+=\s*(\d+)")
CB_RE = re.compile(r"(cb_ctarg_\d+)\.(%s)\(" % "|".join(OPS))


def count(path):
    counts = defaultdict(lambda: defaultdict(int))
    depth_mult = [1]
    pend = None
    for line in open(path):
        if pend is not None:
            depth_mult.append(depth_mult[-1] * pend)
            pend = None
        m = FOR_RE.search(line)
        if m:
            lo, hi, st = int(m.group(2)), int(m.group(3)), int(m.group(4))
            pend = max(0, (hi - lo + st - 1) // st)
        depth_mult.extend([depth_mult[-1]] * (line.count("{") - (1 if m else 0)))
        for cm in CB_RE.finditer(line):
            counts[cm.group(1)][cm.group(2)] += depth_mult[-1]
        for _ in range(line.count("}")):
            if len(depth_mult) > 1:
                depth_mult.pop()
    return counts


def main(paths):
    per_file = {p: count(p) for p in paths}
    cbs = sorted({cb for c in per_file.values() for cb in c},
                 key=lambda s: int(s.rsplit("_", 1)[1]))
    hdr = "cb".ljust(12) + "".join(
        (p.rsplit("__", 1)[-1][:14]).ljust(34) for p in paths)
    print(hdr)
    print(" ".ljust(12) + "wait/pop/resv/push ".ljust(34) * len(paths))
    for cb in cbs:
        row = cb.ljust(12)
        for p in paths:
            c = per_file[p].get(cb, {})
            row += ("%d/%d/%d/%d" % tuple(c.get(o, 0) for o in OPS)).ljust(34)
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
