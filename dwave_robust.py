"""Direct annealer run on the robust surrogate QUBO (report §5.3 route (ii)).

Builds the scenario-averaged surrogate of the 14-bus robust instance from
robust_qaoa_demo.py, sweeps a small set of Lagrangian prices, and submits to
D-Wave via DWaveSampler + EmbeddingComposite (never the hybrid solvers).
Candidates are scored with the EXACT robust evaluator; metrics follow the
report's protocol: embedding size, chain lengths, chain-break fraction,
time-to-solution over the full shot budget.

Run:
    python dwave_robust.py            # QPU if a token is configured,
                                      # else classical SimulatedAnnealing fallback
    python dwave_robust.py --local    # force the classical fallback (free)

Paste the full stdout and dwave_robust_results.json back into the chat.
"""
import sys, json, time, datetime, platform
import numpy as np
from robust_qaoa_demo import (make_weights, scenarios, robust_score,
                              exhaustive_robust, MESHED, N, B)

SEED = 1               # the report's first showcase seed
NUM_READS = 500        # per lambda; 3 lambdas -> 1500 anneals total
LAMBDAS = None         # set below from mean weight

w = make_weights(SEED)
S = scenarios(w)
opt, opt_x, _ = exhaustive_robust(S, w)

# scenario-averaged surrogate weights (uniform master round)
eff = {e: sum(wt for _, ed in S if e in ed) / len(S) for e, wt in w.items()}
wbar = float(np.mean(list(eff.values())))
LAMBDAS = [0.6 * wbar, 1.0 * wbar, 1.5 * wbar]

def surrogate_bqm(lam):
    """Ancilla-free QUBO of Eq.(qubo): Q_ii = weighted degree - lambda, Q_uv = -w_uv."""
    import dimod
    linear = {i: 0.0 for i in range(N)}
    quad = {}
    for (u, v), wt in eff.items():
        linear[u] += wt; linear[v] += wt
        quad[(u, v)] = quad.get((u, v), 0.0) - wt
    for i in linear:
        linear[i] -= lam
    # dimod minimizes; our objective maximizes -> negate
    return dimod.BinaryQuadraticModel({i: -l for i, l in linear.items()},
                                      {k: -q for k, q in quad.items()},
                                      0.0, dimod.BINARY)

def get_sampler(force_local):
    if not force_local:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
            base = DWaveSampler()
            print(f"[sampler] QPU: {base.solver.name}")
            return EmbeddingComposite(base), True
        except Exception as e:
            print(f"[sampler] QPU unavailable ({type(e).__name__}: {e}); "
                  f"falling back to classical SimulatedAnnealingSampler")
    from dwave.samplers import SimulatedAnnealingSampler
    return SimulatedAnnealingSampler(), False

def main():
    force_local = "--local" in sys.argv
    sampler, on_qpu = get_sampler(force_local)
    portfolio, embed_stats, qpu_time_us = {}, [], 0.0
    t0 = time.time()
    for lam in LAMBDAS:
        bqm = surrogate_bqm(lam)
        kwargs = dict(num_reads=NUM_READS)
        if on_qpu:
            kwargs["return_embedding"] = True
            kwargs["label"] = f"eurogrid-robust lam={lam:.3f}"
        ss = sampler.sample(bqm, **kwargs)
        # embedding + timing metrics (QPU only)
        if on_qpu:
            emb = ss.info.get("embedding_context", {}).get("embedding", {})
            chains = [len(c) for c in emb.values()]
            cbf = float(np.mean(ss.record.chain_break_fraction)) if \
                "chain_break_fraction" in ss.record.dtype.names else float("nan")
            embed_stats.append(dict(
                lam=round(lam, 4),
                physical_qubits=int(sum(chains)) if chains else None,
                max_chain=int(max(chains)) if chains else None,
                mean_chain=round(float(np.mean(chains)), 2) if chains else None,
                chain_break_fraction=round(cbf, 4)))
            qpu_time_us += float(ss.info.get("timing", {})
                                 .get("qpu_access_time", 0.0))
        # evaluate candidates exactly; repair to <= B by trimming lowest-degree picks
        for rec in ss.record:
            x = set(int(i) for i in np.flatnonzero(rec.sample))
            if len(x) > B:
                x = set(sorted(x, key=lambda i: -sum(
                    eff.get((min(i, j), max(i, j)), 0) for j in range(N)))[:B])
            key = tuple(sorted(x))
            if key not in portfolio:
                portfolio[key] = robust_score(set(key), S, w)
    wall = time.time() - t0
    best = max(portfolio, key=portfolio.get)
    near = [c for c, v in portfolio.items() if v >= 0.98 * opt]
    print(f"[instance] seed {SEED}, n={N}, B={B}, |S|={len(S)}; "
          f"exhaustive robust optimum {opt:.4f} at {tuple(b+1 for b in opt_x)}")
    print(f"[result]  best sampled robust {portfolio[best]:.4f} "
          f"(ratio {portfolio[best]/opt:.4f}) at {tuple(b+1 for b in best)}")
    print(f"[result]  distinct candidates {len(portfolio)}; within 2% of opt: {len(near)}")
    print(f"[tts]     wall {wall:.1f}s over {NUM_READS*len(LAMBDAS)} anneals"
          + (f"; qpu_access_time {qpu_time_us/1e6:.3f}s" if on_qpu else " (classical fallback)"))
    for st in embed_stats:
        print(f"[embed]   lam={st['lam']}: {st['physical_qubits']} physical qubits, "
              f"max chain {st['max_chain']}, mean {st['mean_chain']}, "
              f"chain-break fraction {st['chain_break_fraction']}")
    out = dict(script="dwave_robust.py",
               date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               environment=dict(python=platform.python_version(),
                                numpy=np.__version__, on_qpu=on_qpu),
               instance=dict(seed=SEED, n=N, B=B, n_scenarios=len(S),
                             lambdas=[round(l, 4) for l in LAMBDAS],
                             num_reads=NUM_READS),
               result=dict(robust_optimum=round(opt, 6),
                           best_ratio=round(portfolio[best]/opt, 6),
                           best_buses_1idx=[b+1 for b in best],
                           distinct_candidates=len(portfolio),
                           near_optimal_2pct=len(near),
                           wall_seconds=round(wall, 2),
                           qpu_access_seconds=round(qpu_time_us/1e6, 4) if on_qpu else None),
               embedding=embed_stats)
    with open("dwave_robust_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("results written to dwave_robust_results.json")

if __name__ == "__main__":
    main()
