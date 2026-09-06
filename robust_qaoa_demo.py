"""
robust_qaoa_demo.py
QAOA as a biased sampler for the max-min robust PMU placement variant,
demonstrated end-to-end on a synthetic CIGRE-style 14-bus instance.

Pipeline (matches the report's sample-and-evaluate architecture):
  master:   multiplicative-weights over scenarios -> surrogate QUBO
  sampler:  QAOA (statevector emulation), two budget encodings:
              (A) full space, transverse-field X mixer + quadratic penalty
              (B) feasible subspace |x|=B, complete-graph XY mixer, Dicke init
  evaluator: exact robust score  F(x) = min_s sum_{e alive in s} w_e cover_e(x)

Baselines: exhaustive robust optimum (C(14,3)=364), nominal greedy
(radial scenario) robust-evaluated, robust greedy.
"""

import itertools
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Instance: CIGRE-style two-feeder MV network (14 buses, 0-indexed)
# ----------------------------------------------------------------------
N = 14
FEEDER1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
           (7, 8), (8, 9), (9, 10)]                      # chain 1..11
FEEDER2 = [(11, 12), (12, 13)]                           # chain 12..14
RADIAL = FEEDER1 + FEEDER2                               # 12 lines
TIES = [(2, 7), (3, 10), (7, 13)]                        # 3 tie switches
MESHED = RADIAL + TIES                                   # 15 lines
B = 3                                                    # PMU budget
EPS_DIVERSITY = 0.02                                     # near-optimal band


def make_weights(seed):
    rng = np.random.default_rng(seed)
    w = {}
    for e in MESHED:
        w[e] = float(rng.uniform(0.5, 1.5))
    return w


def scenarios(w):
    """Contingency set S: switching states + N-1 outages of critical lines."""
    rad_sorted = sorted(RADIAL, key=lambda e: -w[e])
    mesh_sorted = sorted(MESHED, key=lambda e: -w[e])
    S = [
        ("radial",            RADIAL),
        ("meshed",            MESHED),
        ("radial - " + str(rad_sorted[0]),  [e for e in RADIAL if e != rad_sorted[0]]),
        ("radial - " + str(rad_sorted[1]),  [e for e in RADIAL if e != rad_sorted[1]]),
        ("meshed - " + str(mesh_sorted[0]), [e for e in MESHED if e != mesh_sorted[0]]),
    ]
    return S


# ----------------------------------------------------------------------
# Exact evaluator and baselines
# ----------------------------------------------------------------------
def coverage(x_set, edges, w):
    return sum(w[e] for e in edges if e[0] in x_set or e[1] in x_set)


def robust_score(x_set, S, w):
    return min(coverage(x_set, edges, w) for _, edges in S)


def all_feasible():
    return list(itertools.combinations(range(N), B))


def exhaustive_robust(S, w):
    best_val, best_x = -1.0, None
    vals = {}
    for combo in all_feasible():
        v = robust_score(set(combo), S, w)
        vals[combo] = v
        if v > best_val:
            best_val, best_x = v, combo
    return best_val, best_x, vals


def greedy(edges, w):
    """Plain greedy for a single scenario (weighted coverage)."""
    chosen = set()
    for _ in range(B):
        best_gain, best_i = -1.0, None
        base = coverage(chosen, edges, w)
        for i in range(N):
            if i in chosen:
                continue
            g = coverage(chosen | {i}, edges, w) - base
            if g > best_gain:
                best_gain, best_i = g, i
        chosen.add(best_i)
    return chosen


def robust_greedy(S, w):
    chosen = set()
    for _ in range(B):
        best_val, best_i = -1.0, None
        for i in range(N):
            if i in chosen:
                continue
            v = robust_score(chosen | {i}, S, w)
            if v > best_val:
                best_val, best_i = v, i
        chosen.add(best_i)
    return chosen


# ----------------------------------------------------------------------
# QAOA sampler A: full space, X mixer, quadratic budget penalty
# ----------------------------------------------------------------------
DIM = 1 << N
BITS = ((np.arange(DIM)[:, None] >> np.arange(N)) & 1).astype(np.int8)
HAMMING = BITS.sum(axis=1)


def surrogate_cost_diag(eff_w):
    """Diagonal of surrogate coverage objective on all 2^N states."""
    c = np.zeros(DIM)
    for (u, v), wt in eff_w.items():
        c += wt * np.logical_or(BITS[:, u], BITS[:, v])
    return c


def qaoa_full(cost_diag, penalty, p, restarts, rng):
    C = cost_diag - penalty * (HAMMING - B) ** 2

    def apply_mixer(psi, beta):
        t = psi.reshape((2,) * N)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for ax in range(N):
            t0 = np.take(t, 0, axis=ax)
            t1 = np.take(t, 1, axis=ax)
            new0 = c * t0 + s * t1
            new1 = s * t0 + c * t1
            t = np.stack([new0, new1], axis=ax)
        return t.reshape(DIM)

    def run(angles):
        gammas, betas = angles[:p], angles[p:]
        psi = np.full(DIM, 1 / np.sqrt(DIM), dtype=complex)
        for k in range(p):
            psi = psi * np.exp(-1j * gammas[k] * C)
            psi = apply_mixer(psi, betas[k])
        return psi

    def neg_expect(angles):
        psi = run(angles)
        return -float(np.real(np.vdot(psi, C * psi)))

    best = (np.inf, None)
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi / 2, size=2 * p)
        res = minimize(neg_expect, x0, method="COBYLA",
                       options={"maxiter": 250, "rhobeg": 0.3})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    psi = run(best[1])
    return np.abs(psi) ** 2


# ----------------------------------------------------------------------
# QAOA sampler B: feasible subspace, complete-graph XY mixer, Dicke init
# ----------------------------------------------------------------------
FEAS = all_feasible()
FEAS_IDX = {c: i for i, c in enumerate(FEAS)}
M = len(FEAS)


def build_xy_mixer():
    """Complete-graph XY mixer restricted to Hamming weight B: hops one PMU."""
    H = np.zeros((M, M))
    for i, c in enumerate(FEAS):
        cs = set(c)
        for u in c:
            for v in range(N):
                if v in cs:
                    continue
                c2 = tuple(sorted((cs - {u}) | {v}))
                H[i, FEAS_IDX[c2]] = 1.0
    return H


XY_EVALS, XY_EVECS = np.linalg.eigh(build_xy_mixer())


def qaoa_xy(cost_on_feas, p, restarts, rng):
    C = cost_on_feas

    def run(angles):
        gammas, betas = angles[:p], angles[p:]
        psi = np.full(M, 1 / np.sqrt(M), dtype=complex)   # Dicke state
        for k in range(p):
            psi = psi * np.exp(-1j * gammas[k] * C)
            phases = np.exp(-1j * betas[k] * XY_EVALS)
            psi = XY_EVECS @ (phases * (XY_EVECS.T @ psi))
        return psi

    def neg_expect(angles):
        psi = run(angles)
        return -float(np.real(np.vdot(psi, C * psi)))

    best = (np.inf, None)
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi / 2, size=2 * p)
        res = minimize(neg_expect, x0, method="COBYLA",
                       options={"maxiter": 250, "rhobeg": 0.3})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    psi = run(best[1])
    return np.abs(psi) ** 2


# ----------------------------------------------------------------------
# Multiplicative-weights scenario master around either sampler
# ----------------------------------------------------------------------
def mw_master(S, w, sampler, rounds=4, eta=1.0, shots=2000, p=2,
              restarts=3, seed=0):
    rng = np.random.default_rng(seed)
    ps = np.ones(len(S)) / len(S)
    portfolio = {}
    for _ in range(rounds):
        eff = {e: sum(ps[i] * wt for i, (_, ed) in enumerate(S) if e in ed)
               for e, wt in w.items()}
        eff = {e: v for e, v in eff.items() if v > 0}
        if sampler == "full":
            probs = qaoa_full(surrogate_cost_diag(eff), penalty=2.0,
                              p=p, restarts=restarts, rng=rng)
            idx = rng.choice(DIM, size=shots, p=probs)
            cands = set()
            for i in idx:
                if HAMMING[i] == B:
                    cands.add(tuple(np.flatnonzero(BITS[i]).tolist()))
        else:  # "xy"
            cfe = np.array([coverage(set(c), list(eff.keys()),
                                     eff) for c in FEAS])
            probs = qaoa_xy(cfe, p=p, restarts=restarts, rng=rng)
            idx = rng.choice(M, size=shots, p=probs)
            cands = {FEAS[i] for i in idx}
        for c in cands:
            if c not in portfolio:
                portfolio[c] = robust_score(set(c), S, w)
        best_c = max(portfolio, key=portfolio.get)
        # find the minimizing scenario of the current best; upweight it
        vals = [coverage(set(best_c), ed, w) for _, ed in S]
        ps[int(np.argmin(vals))] *= np.exp(eta)
        ps /= ps.sum()
    return portfolio


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def seed_aggregate(n_seeds=20):
    degraded = 0
    ratios = []
    for seed in range(n_seeds):
        w = make_weights(seed)
        S = scenarios(w)
        opt, _, _ = exhaustive_robust(S, w)
        g_nom = robust_score(greedy(RADIAL, w), S, w)
        ratios.append(g_nom / opt)
        if g_nom < opt - 1e-9:
            degraded += 1
    return degraded, n_seeds, min(ratios), float(np.mean(ratios))


def showcase(seed):
    w = make_weights(seed)
    S = scenarios(w)
    opt, opt_x, vals = exhaustive_robust(S, w)
    near = [c for c, v in vals.items() if v >= (1 - EPS_DIVERSITY) * opt]

    g_nom_set = greedy(RADIAL, w)
    g_nom = robust_score(g_nom_set, S, w)
    g_rob_set = robust_greedy(S, w)
    g_rob = robust_score(g_rob_set, S, w)

    results = {}
    for name in ("full", "xy"):
        port = mw_master(S, w, sampler=name, seed=seed)
        best_c = max(port, key=port.get)
        found_near = [c for c, v in port.items()
                      if v >= (1 - EPS_DIVERSITY) * opt]
        results[name] = dict(best=port[best_c], best_x=best_c,
                             n_near=len(found_near),
                             n_cands=len(port))
    return dict(w=w, S=S, opt=opt, opt_x=opt_x, n_near_total=len(near),
                g_nom=g_nom, g_nom_set=sorted(g_nom_set),
                g_rob=g_rob, g_rob_set=sorted(g_rob_set),
                qaoa=results)


if __name__ == "__main__":
    import json, platform, subprocess, datetime, scipy

    deg, tot, worst_r, mean_r = seed_aggregate()
    print(f"[aggregate over {tot} seeds, B={B}, |S|=5]")
    print(f"  nominal greedy robust-suboptimal in {deg}/{tot} seeds")
    print(f"  nominal-greedy robust ratio: mean {mean_r:.4f}, worst {worst_r:.4f}")
    print()

    # showcase: first seed where degradation occurs
    show_seed = None
    for seed in range(20):
        w = make_weights(seed)
        S = scenarios(w)
        opt, _, _ = exhaustive_robust(S, w)
        if robust_score(greedy(RADIAL, w), S, w) < opt - 1e-9:
            show_seed = seed
            break
    print(f"[showcase seed {show_seed}]")
    r = showcase(show_seed)
    opt = r["opt"]
    print(f"  robust optimum (exhaustive, 364 placements): "
          f"{opt:.4f} at buses {tuple(b+1 for b in r['opt_x'])}")
    print(f"  distinct placements within {EPS_DIVERSITY:.0%} of robust opt: "
          f"{r['n_near_total']}")
    print(f"  nominal greedy (radial): robust {r['g_nom']:.4f} "
          f"(ratio {r['g_nom']/opt:.4f}) at {tuple(b+1 for b in r['g_nom_set'])}")
    print(f"  robust greedy:           robust {r['g_rob']:.4f} "
          f"(ratio {r['g_rob']/opt:.4f}) at {tuple(b+1 for b in r['g_rob_set'])}")
    for name, lab in (("full", "QAOA-A (X mixer + penalty, p=2)"),
                      ("xy",   "QAOA-B (XY subspace mixer, p=2)")):
        q = r["qaoa"][name]
        print(f"  {lab}: best robust {q['best']:.4f} "
              f"(ratio {q['best']/opt:.4f}) at "
              f"{tuple(b+1 for b in q['best_x'])}; "
              f"near-optimal found {q['n_near']}/{r['n_near_total']}; "
              f"candidates evaluated {q['n_cands']}")

    # ---- provenance-stamped results file (benchmarking protocol) ----
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "UNCOMMITTED"
    out = {
        "script": "robust_qaoa_demo.py",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": commit,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "instance": {
            "n_buses": N, "budget": B, "radial_lines": len(RADIAL),
            "tie_lines": len(TIES), "n_scenarios": 5,
            "weight_seeds": list(range(20)),
            "eps_diversity": EPS_DIVERSITY,
        },
        "qaoa": {"depth_p": 2, "restarts": 3, "shots_per_round": 2000,
                 "mw_rounds": 4, "mw_eta": 1.0,
                 "encodings": ["A: full space, X mixer, quadratic penalty",
                               "B: subspace, complete XY mixer, Dicke init"]},
        "aggregate": {"nominal_greedy_suboptimal_seeds": deg,
                      "n_seeds": tot,
                      "nominal_greedy_mean_ratio": round(mean_r, 6),
                      "nominal_greedy_worst_ratio": round(worst_r, 6)},
        "showcase_seed": show_seed,
        "showcase": {
            "robust_optimum": round(r["opt"], 6),
            "optimum_buses_1idx": [b + 1 for b in r["opt_x"]],
            "n_near_optimal_total": r["n_near_total"],
            "nominal_greedy_ratio": round(r["g_nom"] / opt, 6),
            "robust_greedy_ratio": round(r["g_rob"] / opt, 6),
            "qaoa": {k: {"best_ratio": round(v["best"] / opt, 6),
                         "near_found": v["n_near"],
                         "candidates": v["n_cands"]}
                     for k, v in r["qaoa"].items()},
        },
    }
    with open("robust_qaoa_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nresults written to robust_qaoa_results.json "
          f"(commit: {commit})")
