"""Aggregate statistics behind the report's robust-variant table (tab:robustqaoa).

Extends robust_qaoa_demo.py's recorded run with:
  - robust-greedy aggregate over the same 20 weight seeds
    (suboptimal count, mean/worst ratio to the exhaustive robust optimum)
  - the neither-dominates observation: on robust greedy's worst seed,
    nominal greedy is optimal
  - showcase seed 12 (robust greedy's worst seed; 2 tied near-optima)

Run:  python robust_qaoa_aggregate.py            # classical aggregates (seconds)
      python robust_qaoa_aggregate.py --qaoa     # + QAOA showcases on seeds 1 and 12 (minutes)
"""
import sys
import numpy as np
from robust_qaoa_demo import (make_weights, scenarios, exhaustive_robust,
                              greedy, robust_greedy, robust_score,
                              RADIAL, EPS_DIVERSITY, showcase)

rows = []
for seed in range(20):
    w = make_weights(seed)
    S = scenarios(w)
    opt, _, vals = exhaustive_robust(S, w)
    ng = robust_score(greedy(RADIAL, w), S, w) / opt
    rg = robust_score(robust_greedy(S, w), S, w) / opt
    n_near = sum(1 for v in vals.values() if v >= (1 - EPS_DIVERSITY) * opt)
    rows.append((seed, ng, rg, n_near))

for name, col in (("nominal greedy", 1), ("robust greedy", 2)):
    r = [row[col] for row in rows]
    sub = sum(1 for v in r if v < 1 - 1e-9)
    print(f"{name:>15}: suboptimal {sub}/20, mean {np.mean(r):.4f}, worst {min(r):.4f}")

worst = min(rows, key=lambda t: t[2])
print(f"robust greedy's worst seed is {worst[0]} (ratio {worst[2]:.4f}); "
      f"nominal greedy there scores {worst[1]:.4f} -> neither heuristic dominates")
s12 = rows[12]
print(f"seed 12 near-optimal placements at eps={EPS_DIVERSITY:.0%}: {s12[3]}")

if "--qaoa" in sys.argv:
    for seed in (1, 12):
        r = showcase(seed)
        opt = r["opt"]
        for name in ("full", "xy"):
            q = r["qaoa"][name]
            print(f"seed {seed} QAOA-{name}: best ratio {q['best']/opt:.4f}, "
                  f"near-optimal found {q['n_near']}/{r['n_near_total']}, "
                  f"candidates {q['n_cands']}/364")
