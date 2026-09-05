"""
EuroGrid sensor placement -- structural analysis.

The client's stated objective is NOT full observability (minimum dominating set).
Read the survey: "the number of most critical LINES to be monitored", with
"line prioritization" as input and a "PMU budget" as the constraint.

That is BUDGETED MAXIMUM WEIGHTED COVERAGE:  a PMU at bus i measures the current
phasor on every line incident to i, so choosing <= B buses to maximise the weight
of covered lines is Max k-Vertex Cover.
"""
import numpy as np, networkx as nx, itertools, time
from networkx.algorithms.approximation import treewidth_min_degree


# ---------------- realistic topologies ----------------
def radial_feeder(n_bus=60, max_deg=4, seed=0):
    """A distribution feeder: radial (a tree), low degree, planar."""
    rng = np.random.default_rng(seed)
    G = nx.Graph(); G.add_node(0)                       # substation
    for v in range(1, n_bus):
        cand = [u for u in G.nodes() if G.degree(u) < max_deg]
        G.add_edge(int(rng.choice(cand)), v)
    return G

def weakly_meshed(n_bus=60, n_ties=5, max_deg=4, seed=0):
    """Radial feeder plus a few normally-open tie switches -> a few cycles."""
    G = radial_feeder(n_bus, max_deg, seed)
    rng = np.random.default_rng(seed + 100)
    added = 0
    while added < n_ties:
        u, v = rng.integers(0, n_bus, 2)
        if u != v and not G.has_edge(u, v) and G.degree(u) < max_deg and G.degree(v) < max_deg:
            G.add_edge(int(u), int(v)); added += 1
    return G

def meshed_transmission(n_bus=118, seed=0):
    """Transmission-like: denser, still sparse and near-planar."""
    G = nx.connected_watts_strogatz_graph(n_bus, 4, 0.15, seed=seed)
    return G


# ---------------- the objective ----------------
def line_weights(G, seed=0):
    """Line prioritisation: criticality weight per line."""
    rng = np.random.default_rng(seed)
    return {e: float(round(rng.uniform(1, 10), 2)) for e in G.edges()}

def covered_weight(G, w, S):
    S = set(S)
    return sum(wt for (u, v), wt in w.items() if u in S or v in S)


# ---------------- QUBO:  no ancillas needed ----------------
def coverage_qubo(G, w):
    """cover(e) = x_u + x_v - x_u x_v  is ALREADY quadratic.
    So   max sum_e w_e cover(e)  =  max sum_i d_w(i) x_i - sum_e w_e x_u x_v.
    Diagonal = weighted degree; off-diagonal = -w_e ON THE NETWORK EDGES ONLY.
    => the QUBO interaction graph IS the grid itself, one variable per bus."""
    nodes = sorted(G.nodes()); idx = {v: k for k, v in enumerate(nodes)}
    n = len(nodes); Q = np.zeros((n, n))
    for (u, v), wt in w.items():
        Q[idx[u], idx[u]] += wt
        Q[idx[v], idx[v]] += wt
        Q[idx[u], idx[v]] -= wt
    return Q, nodes

def qubo_value(Q, x):
    return float(x @ np.triu(Q) @ x) if False else float(x @ Q @ x)


# ---------------- solvers ----------------
def greedy(G, w, B):
    """Nemhauser-Wolsey-Fisher greedy: monotone submodular + cardinality
    => guaranteed >= (1 - 1/e) ~ 0.632 of optimum."""
    S, remaining = [], dict(w)
    for _ in range(B):
        best, gain = None, -1
        for v in G.nodes():
            if v in S: continue
            g = sum(wt for (a, b), wt in remaining.items() if a == v or b == v)
            if g > gain: best, gain = v, g
        if best is None or gain <= 0: break
        S.append(best)
        remaining = {e: wt for e, wt in remaining.items() if best not in e}
    return S

def exact(G, w, B, cap=3_000_000):
    import math
    nodes = list(G.nodes())
    if math.comb(len(nodes), B) > cap: return None, None
    best, bv = None, -1
    for S in itertools.combinations(nodes, B):
        v = covered_weight(G, w, S)
        if v > bv: best, bv = S, v
    return list(best), bv

def tree_dp_note(G):
    tw, _ = treewidth_min_degree(G)
    return tw


if __name__ == "__main__":
    print("%-22s %5s %5s %7s %7s %6s | %-28s" %
          ("network","|V|","|E|","avgdeg","maxdeg","tw~","budgeted max coverage"))
    for name, G in [("radial feeder (tree)", radial_feeder(28, 4, 0)),
                    ("weakly meshed (5 ties)", weakly_meshed(28, 4, 4, 0)),
                    ("meshed transmission", meshed_transmission(28, 0))]:
        w = line_weights(G, 1); tw = tree_dp_note(G)
        B = 5
        t0 = time.time(); S_g = greedy(G, w, B); tg = time.time()-t0
        t0 = time.time(); S_e, v_e = exact(G, w, B); te = time.time()-t0
        v_g = covered_weight(G, w, S_g); W = sum(w.values())
        line = "greedy %.4f of opt in %.3fs" % (v_g/v_e, tg) if v_e else "exact too big"
        print("%-22s %5d %5d %7.2f %7d %6d | %s" %
              (name, G.number_of_nodes(), G.number_of_edges(),
               2*G.number_of_edges()/G.number_of_nodes(),
               max(dict(G.degree()).values()), tw, line))
        if v_e:
            print("%22s   greedy covers %.1f%% of total line weight;  exact %.1f%%  (brute force %.1fs)"
                  % ("", 100*v_g/W, 100*v_e/W, te))
