"""Budgeted max weighted coverage on a real-calibrated German MV network.

Run this LOCALLY (needs pandapower; simbench optional):

    pip install pandapower simbench    # simbench optional, see fallback
    python simbench_coverage.py

Networks tried in order:
  1. SimBench '1-MV-rural--0-sw' (realistic German rural MV network)  [needs simbench]
  2. pandapower's mv_oberrhein (real DSO network, ~180 buses)          [pandapower only]

Two operating configurations per network:
  RADIAL: in-service lines whose line switches are all closed (normal operation)
  MESHED: all in-service lines (every tie closed)

Solvers:
  greedy          -- the (1-1/e) submodular greedy
  exact tree DP   -- O(n * B^2 * 2) DP on forests; EXACT when the config is a forest
                     (self-tested against enumeration on random trees at startup)
For meshed configs the DP does not apply; greedy is reported alone unless
gurobipy or PuLP is importable, in which case an exact ILP runs too.

Paste the full stdout back into the chat.
"""
import itertools, random, sys
from collections import defaultdict

# ----------------------------------------------------------------------
# generic coverage machinery
# ----------------------------------------------------------------------
def coverage(placed, edges, w):
    return sum(w[i] for i, (u, v) in enumerate(edges) if u in placed or v in placed)

def greedy(buses, B, edges, w):
    placed, val = set(), 0.0
    for _ in range(min(B, len(buses))):
        best, gain = None, -1.0
        for b in buses:
            if b in placed:
                continue
            g = coverage(placed | {b}, edges, w) - val
            if g > gain:
                best, gain = b, g
        placed.add(best); val += gain
    return placed, val

def exact_enum(buses, B, edges, w):
    best, bset = -1.0, None
    for c in itertools.combinations(sorted(buses), B):
        v = coverage(set(c), edges, w)
        if v > best:
            best, bset = v, set(c)
    return bset, best

# ----------------------------------------------------------------------
# exact DP on a forest (treewidth 1). State: dp[v][b][s] = best covered
# weight in subtree(v) using b PMUs inside it, s = 1 iff v hosts a PMU.
# Edge (v, child) weight counts iff s or s_child.
# ----------------------------------------------------------------------
NEG = float("-inf")

def tree_dp_forest(buses, B, edges, w):
    adj = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, w[i])); adj[v].append((u, w[i]))

    seen, total = set(), 0.0
    best_global = [0.0]
    # process each tree of the forest, then knapsack-merge the per-tree tables
    per_tree = []
    for root in sorted(buses):
        if root in seen:
            continue
        # iterative post-order
        order, parent, pw = [], {root: None}, {root: 0.0}
        stack = [root]
        while stack:
            v = stack.pop(); order.append(v); seen.add(v)
            for (c, ew) in adj[v]:
                if c != parent[v]:
                    parent[c] = v; pw[c] = ew; stack.append(c)
        dp = {}
        for v in reversed(order):
            base = [[NEG] * 2 for _ in range(B + 1)]
            base[0][0] = 0.0
            if B >= 1:
                base[1][1] = 0.0
            cur = base
            for (c, ew) in adj[v]:
                if c == parent[v]:
                    continue
                nxt = [[NEG] * 2 for _ in range(B + 1)]
                dpc = dp.pop(c)
                for b1 in range(B + 1):
                    for s in (0, 1):
                        if cur[b1][s] == NEG:
                            continue
                        for b2 in range(B + 1 - b1):
                            for sc in (0, 1):
                                if dpc[b2][sc] == NEG:
                                    continue
                                gain = ew if (s or sc) else 0.0
                                val = cur[b1][s] + dpc[b2][sc] + gain
                                if val > nxt[b1 + b2][s]:
                                    nxt[b1 + b2][s] = val
                cur = nxt
            dp[v] = cur
        per_tree.append(dp[root])
    # merge trees under the shared budget
    merged = [NEG] * (B + 1); merged[0] = 0.0
    for t in per_tree:
        best_t = [max(t[b][0], t[b][1]) for b in range(B + 1)]
        nxt = [NEG] * (B + 1)
        for b1 in range(B + 1):
            if merged[b1] == NEG:
                continue
            for b2 in range(B + 1 - b1):
                if best_t[b2] == NEG:
                    continue
                nxt[b1 + b2] = max(nxt[b1 + b2], merged[b1] + best_t[b2])
        merged = nxt
    return max(v for v in merged if v != NEG)

def selftest_dp(trials=200):
    rng = random.Random(0)
    for t in range(trials):
        n = rng.randint(2, 12)
        buses = list(range(n))
        edges = [(rng.randrange(i), i) for i in range(1, n)]  # random tree
        w = [rng.uniform(1, 10) for _ in edges]
        B = rng.randint(1, min(4, n))
        _, ev = exact_enum(buses, B, edges, w)
        dv = tree_dp_forest(buses, B, edges, w)
        assert abs(ev - dv) < 1e-6, f"DP mismatch on trial {t}: enum {ev} dp {dv}"
    print(f"[selftest] tree DP == enumeration on {trials} random trees  OK")

# ----------------------------------------------------------------------
# optional exact ILP for meshed configs
# ----------------------------------------------------------------------
def exact_ilp(buses, B, edges, w):
    try:
        import pulp
    except ImportError:
        return None
    prob = pulp.LpProblem("maxcov", pulp.LpMaximize)
    x = {b: pulp.LpVariable(f"x{b}", cat="Binary") for b in buses}
    y = {i: pulp.LpVariable(f"y{i}", cat="Binary") for i in range(len(edges))}
    prob += pulp.lpSum(w[i] * y[i] for i in y)
    for i, (u, v) in enumerate(edges):
        prob += y[i] <= x[u] + x[v]
    prob += pulp.lpSum(x.values()) <= B
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sum(w[i] * y[i].value() for i in y)

# ----------------------------------------------------------------------
# network loading
# ----------------------------------------------------------------------
def load_network():
    try:
        import simbench as sb
        code = "1-MV-rural--0-sw"
        net = sb.get_simbench_net(code)
        return net, f"SimBench {code}"
    except Exception as e:
        print(f"[info] simbench unavailable ({type(e).__name__}); "
              f"falling back to pandapower mv_oberrhein")
        import pandapower.networks as pn
        return pn.mv_oberrhein(), "pandapower mv_oberrhein"

def extract_configs(net):
    line = net.line
    open_line_ids = set()
    sw = net.switch
    for _, s in sw.iterrows():
        if s.et == "l" and not s.closed:
            open_line_ids.add(int(s.element))
    def edges_of(ids):
        out = []
        for lid in ids:
            r = line.loc[lid]
            out.append((int(r.from_bus), int(r.to_bus)))
        return out
    in_service = [int(i) for i in line.index if bool(line.at[i, "in_service"])]
    meshed_ids = in_service
    radial_ids = [i for i in in_service if i not in open_line_ids]
    return edges_of(radial_ids), edges_of(meshed_ids)

def is_forest(buses, edges):
    parent = {b: b for b in buses}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True

# ----------------------------------------------------------------------
def run_config(name, edges, budgets=(5, 10, 15), seeds=25):
    buses = sorted({b for e in edges for b in e})
    forest = is_forest(buses, edges)
    print(f"\n=== {name}: {len(buses)} buses, {len(edges)} lines, "
          f"{'forest (radial)' if forest else 'meshed'} ===")
    for B in budgets:
        w = [1.0] * len(edges)
        _, gv = greedy(buses, B, edges, w)
        line_out = f"B={B:>3} uniform: greedy {gv:.0f}/{len(edges)}"
        if forest:
            ev = tree_dp_forest(buses, B, edges, w)
            line_out += f"  exactDP {ev:.0f}  ratio {gv/ev:.4f}"
        else:
            iv = exact_ilp(buses, B, edges, w)
            if iv is not None:
                line_out += f"  exactILP {iv:.0f}  ratio {gv/iv:.4f}"
        print(line_out)
        ratios = []
        for seed in range(seeds):
            rng = random.Random(seed)
            w = [rng.uniform(1, 10) for _ in edges]
            _, gv = greedy(buses, B, edges, w)
            if forest:
                ev = tree_dp_forest(buses, B, edges, w)
                ratios.append(gv / ev)
            else:
                iv = exact_ilp(buses, B, edges, w)
                if iv:
                    ratios.append(gv / iv)
        if ratios:
            print(f"        {len(ratios)} random-priority seeds: "
                  f"mean {sum(ratios)/len(ratios):.4f}, worst {min(ratios):.4f}")

if __name__ == "__main__":
    selftest_dp()
    net, label = load_network()
    radial, meshed = extract_configs(net)
    run_config(f"{label} — radial operation", radial)
    run_config(f"{label} — all ties closed", meshed)
