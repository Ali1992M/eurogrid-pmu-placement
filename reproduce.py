"""Reproduce every number in the report from pmu.py primitives.

Outputs CSVs into results/ so each table in the report maps to one file:
  table1_greedy_vs_exact.csv  -> report Table 1 (108 instances)
  table2_treewidth.csv        -> report Table 2
  qubo_identity.txt           -> the 31-couplings / 500-assignment check

Deterministic: all seeds fixed below; no RNG state leaks between studies.
"""
import csv, os
import numpy as np
from pmu import (radial_feeder, weakly_meshed, meshed_transmission,
                 line_weights, covered_weight, coverage_qubo, qubo_value,
                 greedy, exact, tree_dp_note)

os.makedirs("results", exist_ok=True)

# ---------- Table 1: greedy vs exact, 108 instances ----------
FAMILIES = [("radial feeder", lambda s: radial_feeder(30, 4, s)),
            ("weakly meshed", lambda s: weakly_meshed(30, 4, 4, s)),
            ("denser meshed", lambda s: meshed_transmission(30, s))]
BUDGETS, SEEDS, WSEED_OFFSET = (3, 4, 5), range(12), 50

rows, allr = [], []
for name, gen in FAMILIES:
    ratios = []
    for B in BUDGETS:
        for seed in SEEDS:
            G = gen(seed)
            w = line_weights(G, seed + WSEED_OFFSET)
            _, v_exact = exact(G, w, B)
            v_greedy = covered_weight(G, w, greedy(G, w, B))
            ratios.append(v_greedy / v_exact)
    allr += ratios
    rows.append([name, f"{np.mean(ratios):.4f}", f"{min(ratios):.4f}", len(ratios)])
rows.append(["overall", f"{np.mean(allr):.4f}", f"{min(allr):.4f}", len(allr)])

with open("results/table1_greedy_vs_exact.csv", "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["topology", "mean_ratio", "worst_observed", "n_instances"])
    wtr.writerows(rows)
print("Table 1:", *rows, sep="\n  ")

# ---------- QUBO identity check ----------
G = weakly_meshed(28, 4, 4, 0)
w = line_weights(G, 1)
Q, nodes = coverage_qubo(G, w)
rng = np.random.default_rng(0)
matches = 0
for _ in range(500):
    x = rng.integers(0, 2, len(nodes))
    S = [nodes[i] for i in range(len(nodes)) if x[i]]
    if abs(qubo_value(Q, x) - covered_weight(G, w, S)) < 1e-9:
        matches += 1
couplings = int(np.count_nonzero(np.triu(Q, 1)))
with open("results/qubo_identity.txt", "w") as f:
    f.write(f"28-bus weakly meshed feeder: {couplings} QUBO couplings, "
            f"{G.number_of_edges()} network lines\n"
            f"QUBO value == covered weight on {matches}/500 random assignments\n")
print(f"QUBO identity: {matches}/500; couplings={couplings}, lines={G.number_of_edges()}")

# ---------- Table 2: treewidth (min-degree heuristic upper bound) ----------
tw_rows = []
for name, G in [("radial feeder", radial_feeder(100, 4, 0)),
                ("radial feeder", radial_feeder(400, 4, 0)),
                ("weakly meshed (8 ties)", weakly_meshed(100, 8, 4, 0)),
                ("weakly meshed (30 ties)", weakly_meshed(400, 30, 4, 0))]:
    tw_rows.append([name, G.number_of_nodes(), tree_dp_note(G)])
with open("results/table2_treewidth.csv", "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["topology", "n_buses", "treewidth_upper_bound_min_degree"])
    wtr.writerows(tw_rows)
print("Table 2:", *tw_rows, sep="\n  ")
