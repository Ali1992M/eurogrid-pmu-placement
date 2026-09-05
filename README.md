# eurogrid-pmu-placement

Instance generators, classical baselines (greedy, exhaustive exact), an
ancilla-free QUBO encoding, and the benchmark pipeline behind the EuroGrid
sensor-placement use-case assessment (budgeted maximum weighted coverage /
Max k-Vertex Cover).

## Contents

| File | Purpose |
|---|---|
| `pmu.py` | Topology generators (radial feeder, weakly meshed, meshed), line-weight model, coverage objective, ancilla-free QUBO, greedy and exact solvers, treewidth bound. Running it prints a 28-bus demo. |
| `fig_pmu.py` | Regenerates `fig_pmu.png`: the 108-instance greedy-vs-exact study (left panel) and the QUBO-graph-is-the-grid illustration (right panel). |
| `reproduce.py` | Regenerates every reported number as CSVs in `results/`. |
| `results/` | `table1_greedy_vs_exact.csv`, `table2_treewidth.csv`, `qubo_identity.txt`. |

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py     # report Tables 1 & 2 + the 31-coupling QUBO identity check
python fig_pmu.py       # the two-panel figure (fig_pmu.png)
python pmu.py           # quick 28-bus demo across three topologies
```

Expected output of `reproduce.py` (deterministic; all seeds fixed in-source):

```
Table 1 (greedy value / exact optimum, 108 instances, n=30, B in {3,4,5}, 12 seeds):
  radial feeder   mean 0.9996  worst 0.9839
  weakly meshed   mean 0.9994  worst 0.9868
  denser meshed   mean 0.9984  worst 0.9811
  overall         mean 0.9991  worst 0.9811
QUBO identity: 500/500 random assignments match; 31 couplings = 31 lines (28-bus feeder)
Table 2 (treewidth, min-degree heuristic upper bound):
  radial 100 -> 1, radial 400 -> 1, weakly meshed (8 ties, 100) -> 4,
  weakly meshed (30 ties, 400) -> 11
```

## Notes and scope

- **Treewidth values are min-degree heuristic upper bounds**
  (`networkx.algorithms.approximation.treewidth_min_degree`), exact for trees
  (tw = 1). The report's tractability argument only needs an upper bound.
- The exact solver is exhaustive enumeration, feasible at the n = 30 study
  scale (C(30, 5) = 142,506 subsets per instance).
- **Not in this repository:** the CIGRE MV benchmark study (report Table 3),
  the Rydberg atom-count estimates (report Table 4), and the distance-gap
  study — these were produced separately and are not covered by
  `reproduce.py`.

## Environment

Python 3.10+; pinned dependencies in `requirements.txt`
(numpy 2.4.4, networkx 3.6.1, matplotlib 3.10.8). Treewidth heuristic and
layout outputs can vary slightly across networkx versions; the pinned
versions reproduce the reported numbers exactly.

## Citing

```bibtex
@misc{pmucode,
  author       = {Mahmoud, Ali},
  title        = {eurogrid-pmu-placement: instance generators, greedy/DP/ILP
                  baselines, and benchmark pipeline for budgeted PMU coverage},
  year         = {2026},
  howpublished = {\url{https://github.com/<user>/eurogrid-pmu-placement}},
  note         = {Version v1.0. DOI: 10.5281/zenodo.XXXXXXX}
}
```

(Fill in the GitHub URL after pushing, and the DOI after archiving the v1.0
release through the GitHub–Zenodo integration.)

## License

MIT — see `LICENSE`.
