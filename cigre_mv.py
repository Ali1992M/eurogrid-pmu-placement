"""Budgeted max weighted coverage on the CIGRE MV European benchmark network.

Topology per CIGRE Task Force C6.04.02 (as distributed in pandapower's
create_cigre_network_mv): buses 1..14 at 20 kV fed from bus 0 (110 kV) via
two transformers (0-1, 0-12). Lines listed below; S1, S2, S3 are the
normally-open tie switches. Radial operation = S1,S2,S3 open.
"""
import itertools, random, sys
from collections import defaultdict

# (u, v, tag)  -- MV lines only; transformers excluded as PMU "lines"
LINES = [
    (1, 2, ""), (2, 3, ""), (3, 4, ""), (4, 5, ""), (5, 6, ""),
    (7, 8, ""), (8, 9, ""), (9, 10, ""), (10, 11, ""), (3, 8, ""),
    (12, 13, ""), (13, 14, ""),
    (6, 7, "S1"), (4, 11, "S2"), (8, 14, "S3"),
]
BUSES = sorted({b for u, v, _ in LINES for b in (u, v)})

def coverage(placed, edges, w):
    return sum(w[i] for i, (u, v, _) in enumerate(edges) if u in placed or v in placed)

def greedy(B, edges, w):
    placed, val = set(), 0.0
    for _ in range(B):
        best, bestgain = None, -1
        for b in BUSES:
            if b in placed: continue
            g = coverage(placed | {b}, edges, w) - val
            if g > bestgain: best, bestgain = b, g
        placed.add(best); val += bestgain
    return placed, val

def exact(B, edges, w):
    best, bestset = -1, None
    for combo in itertools.combinations(BUSES, B):
        v = coverage(set(combo), edges, w)
        if v > best: best, bestset = v, set(combo)
    return bestset, best

def cycles_exist(edges):
    parent = {b: b for b in BUSES}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cyc = 0
    for u, v, _ in edges:
        ru, rv = find(u), find(v)
        if ru == rv: cyc += 1
        else: parent[ru] = rv
    return cyc

def run(config_name, edges):
    n_cyc = cycles_exist(edges)
    print(f"\n=== {config_name}: {len(BUSES)} buses, {len(edges)} lines, "
          f"{n_cyc} independent cycles ({'radial forest' if n_cyc==0 else 'meshed'}) ===")
    degs = defaultdict(int)
    for u, v, _ in edges: degs[u] += 1; degs[v] += 1
    print(f"max degree {max(degs.values())}")
    for B in (2, 3, 4, 5):
        # uniform weights
        w = [1.0]*len(edges)
        gset, gval = greedy(B, edges, w)
        eset, eval_ = exact(B, edges, w)
        print(f"B={B} uniform:  greedy {gval:.0f}/{len(edges)}  exact {eval_:.0f}/{len(edges)}"
              f"  ratio {gval/eval_:.4f}  greedy buses {sorted(gset)}  exact buses {sorted(eset)}")
        # random priorities, 50 seeds
        ratios = []
        for seed in range(50):
            rng = random.Random(seed)
            w = [rng.uniform(1, 10) for _ in edges]
            _, gv = greedy(B, edges, w)
            _, ev = exact(B, edges, w)
            ratios.append(gv/ev)
        print(f"        50 random-priority seeds: mean ratio {sum(ratios)/len(ratios):.4f}, "
              f"worst {min(ratios):.4f}")

radial = [l for l in LINES if not l[2]]          # S1,S2,S3 open
meshed = LINES                                    # all ties closed
run("Radial operation (ties open)", radial)
run("Meshed (all ties closed)", meshed)
