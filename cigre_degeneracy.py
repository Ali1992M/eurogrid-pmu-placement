"""Enumerate ALL tied-optimal placements on the CIGRE MV network.

Backs the report's degeneracy claim: at B=2 (uniform priorities, radial
operation) ten distinct two-bus placements all achieve the optimum of 5/12
covered lines; at B=3, seventeen placements tie. Run: python cigre_degeneracy.py
"""
import itertools
from cigre_mv import LINES, BUSES, coverage

radial = [l for l in LINES if not l[2]]
meshed = LINES

for name, edges in [("radial (ties open)", radial), ("meshed (ties closed)", meshed)]:
    w = [1.0] * len(edges)
    for B in (2, 3):
        vals = {c: coverage(set(c), edges, w)
                for c in itertools.combinations(BUSES, B)}
        best = max(vals.values())
        opt = sorted(sorted(c) for c, v in vals.items() if v == best)
        print(f"{name}  B={B}: optimum {best:.0f}/{len(edges)} lines; "
              f"{len(opt)} tied optimal placements")
        for c in opt:
            print(f"    {c}")
