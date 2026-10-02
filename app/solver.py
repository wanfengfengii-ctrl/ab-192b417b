"""De novo short-peptide sequencing core.

Mass-conservative enumeration + constrained combinatorial optimization.

Pipeline
--------
1. Enumerate every residue sequence (length within the requested bounds)
   whose residue masses sum *exactly* to the precursor mass.  Sequences are
   produced in lexicographic order; total mass is conserved by construction
   (no greedy per-peak nearest residue is ever joined into an
   inconsistent chain).
2. For a candidate of length L, every internal cleavage k = 1..L-1 gives a
   complementary ion pair with integer theoretical masses
   ``prefix_k`` and ``precursor - prefix_k``.
3. An observed peak may be assigned to at most one ion (and an ion to at
   most one peak).  The primary objective is the number D of cleavages
   supported by *both* complementary ions; the secondary objective, taken
   only among assignments attaining maximum D, is the sum of absolute
   mass errors over all assigned peaks.
4. The number of cleavages with no supporting ion and the number of
   unassigned (spurious) peaks are bounded by the request.

Because a peptide is short (at most 12 residues by API validation, so at
most 11 cleavages and 22 ions against at most 28 peaks), the per-candidate
optimization is done exactly:

* stage 1 -- branch-and-bound maximizing D, pruned with the size of a
  maximum bipartite matching between the still-free ions and peaks;
* stage 2 -- branch-and-bound over the doubly-supported cleavages (which
  cleavage, which ordered distinct peak pair), pruning on a non-negative
  error lower bound; at each leaf the optimal single-ion completion is a
  small min-cost flow (successive shortest paths with Dijkstra/potentials)
  whose two-tier cleavage nodes maximize the number of additionally
  covered cleavages and then minimize the total absolute error.

No heuristic approximation is involved; ties at the optimum are exact.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional


class SequenceError(Exception):
    """User-facing failure with a locatable (path-style) source."""

    def __init__(self, code: str, message: str, location: str = "request"):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location


# ---------------------------------------------------------------------------
# Min-cost flow (integer costs, successive shortest paths + potentials)
# ---------------------------------------------------------------------------


class MinCost:
    """Min-cost max-flow.

    Edge record layout: [to, reverse_index, residual_capacity, cost, is_forward].
    """

    def __init__(self, n: int):
        self.n = n
        self.g: list[list[list]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        fwd = [v, len(self.g[v]), cap, cost, True]
        rev = [u, len(self.g[u]), 0, -cost, False]
        self.g[u].append(fwd)
        self.g[v].append(rev)

    def flow(self, s: int, t: int, want: int) -> tuple[int, int]:
        """Send up to ``want`` units; return (units_sent, total_cost)."""
        n = self.n
        INF = 10**30
        pot = [0] * n
        g = self.g
        sent = cost_sum = 0
        while sent < want:
            dist = [INF] * n
            pv = [-1] * n
            pe = [-1] * n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                for ei, e in enumerate(g[u]):
                    if e[2] <= 0:
                        continue
                    nd = d + e[3] + pot[u] - pot[e[0]]
                    if nd < dist[e[0]]:
                        dist[e[0]] = nd
                        pv[e[0]] = u
                        pe[e[0]] = ei
                        heapq.heappush(pq, (nd, e[0]))
            if pv[t] < 0:
                break
            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]
            add = want - sent
            v = t
            while v != s:
                add = min(add, g[pv[v]][pe[v]][2])
                v = pv[v]
            v = t
            path_cost = 0
            while v != s:
                u, ei = pv[v], pe[v]
                e = g[u][ei]
                e[2] -= add
                g[v][e[1]][2] += add
                path_cost += e[3]
                v = u
            sent += add
            cost_sum += add * path_cost
        return sent, cost_sum

    def is_saturated_forward(self, u: int, v: int) -> bool:
        """True iff the forward edge u->v currently carries flow."""
        for e in self.g[u]:
            if e[0] == v and e[4]:
                return e[2] == 0
        return False

    def flow_on_forward(self, u: int) -> list[int]:
        """Destinations v of forward edges out of u currently carrying flow."""
        out = []
        for e in self.g[u]:
            if e[4] and self.g[e[0]][e[1]][2] > 0:
                out.append(e[0])
        return out


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    cleavage: int
    ion: str  # "prefix" | "suffix"
    theoretical_mass: int
    peak_id: str
    observed_mass: int
    error: int  # observed - theoretical


@dataclass
class CandidateResult:
    sequence: str
    length: int
    cleavage_theory: dict  # k -> {"prefix": int, "suffix": int}
    assignments: list[Assignment]
    total_abs_error: int
    double_supported_cleavages: int
    supported_cleavages: list[int]
    missing_cleavages: int
    unassigned_peaks: list[str]
    spurious_peaks: int


@dataclass
class _Opts:
    labels: list[str]
    weights: list[int]
    peaks: list[tuple[str, int]]
    precursor: int
    tol: int
    max_missing: int
    max_spurious: int
    min_len: int
    max_len: int


@dataclass
class Solution:
    status: str
    primary: Optional[CandidateResult] = None
    witnesses: list[CandidateResult] = field(default_factory=list)
    ambiguous: bool = False
    reason: Optional[dict] = None
    candidates_examined: int = 0


# ---------------------------------------------------------------------------
# Bipartite matching (Kuhn) -- used for branch-and-bound upper bounds
# ---------------------------------------------------------------------------


def _max_matching_size(ions: list[int], free_peaks: set[int],
                       ion_edges: list[list[int]]) -> int:
    """Maximum matching between the given ions and the given peak subset."""
    match_p: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for p in ion_edges[i]:
            if p not in free_peaks or p in seen:
                continue
            seen.add(p)
            if p not in match_p or augment(match_p[p], seen):
                match_p[p] = i
                return True
        return False

    size = 0
    for i in ions:
        if augment(i, set()):
            size += 1
    return size


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------


def _evaluate(masses: list[int], seq_str: str, o: _Opts,
              node_budget: list[int],
              global_best: Optional[list] = None) -> Optional[CandidateResult]:
    L = len(masses)
    m = L - 1
    n = len(o.peaks)
    g_d = global_best[0] if global_best else -1
    g_e = global_best[1] if global_best else None

    # ---- complementary theoretical ions ----------------------------------
    prefix = []
    acc = 0
    for k in range(m):
        acc += masses[k]
        prefix.append(acc)
    theory = prefix + [o.precursor - x for x in prefix]
    # ion indices: 0..m-1 prefix@(k=i+1); m..2m-1 suffix@(k=i-m+1)
    cleavage_of = list(range(1, L)) + list(range(1, L))
    ion_kind = ["prefix"] * m + ["suffix"] * m

    # ---- feasible ion -> peak edges (within uniform tolerance) -----------
    ion_edges: list[list[int]] = [[] for _ in range(2 * m)]
    edge_cost: dict[tuple[int, int], int] = {}
    for i in range(2 * m):
        tm = theory[i]
        for p, (_, pm) in enumerate(o.peaks):
            d = abs(tm - pm)
            if d <= o.tol:
                ion_edges[i].append(p)
                edge_cost[(i, p)] = d

    def pair_options(k: int, used: set[int]):
        """Feasible (cost, pa, pb) for making cleavage k doubly supported,
        given peaks already consumed; sorted by cost."""
        ia, ib = k - 1, m + k - 1
        opts = []
        for pa in ion_edges[ia]:
            if pa in used:
                continue
            ca = edge_cost[(ia, pa)]
            for pb in ion_edges[ib]:
                if pb == pa or pb in used:
                    continue
                opts.append((ca + edge_cost[(ib, pb)], pa, pb))
        opts.sort()
        return opts

    all_peaks = set(range(n))

    # A cleavage can be covered at all only if at least one ion has an edge.
    coverable = {
        k for k in range(1, L)
        if ion_edges[k - 1] or ion_edges[m + k - 1]
    }
    if m - len(coverable) > o.max_missing:
        return None  # structurally more missing cleavages than allowed

    # Static MRV order: cleavages with fewest pair choices decided first.
    pair_count = []
    for k in range(1, L):
        ia, ib = k - 1, m + k - 1
        # count distinct feasible pairs on the full peak set
        cnt = 0
        for pa in ion_edges[ia]:
            cnt += sum(1 for pb in ion_edges[ib] if pb != pa)
        pair_count.append((cnt, k))
    pair_count.sort()
    order = [k for _, k in pair_count]

    def tick() -> None:
        node_budget[0] -= 1
        if node_budget[0] <= 0:
            raise SequenceError(
                "SEARCH_BUDGET_EXCEEDED",
                "the assignment search exhausted its node budget on this "
                "spectrum (very dense tolerance/peak configuration); reduce "
                "request.tolerance or narrow the alphabet/length range",
                "request.tolerance",
            )

    # ---- stage 1: maximize D (doubly supported cleavages) ----------------
    # MRV order (fewest feasible peak pairs first); the first descent finds a
    # good D, then the matching upper bound prunes siblings aggressively.
    best_d = 0

    def bb1(idx: int, used: set[int], got: int) -> None:
        nonlocal best_d
        tick()
        if idx == m:
            if got > best_d:
                best_d = got
            return
        remain = m - idx
        if got + remain <= best_d:
            return
        remaining_ions: list[int] = []
        for j in range(idx, m):
            k = order[j]
            if any(p not in used for p in ion_edges[k - 1]):
                remaining_ions.append(k - 1)
            if any(p not in used for p in ion_edges[m + k - 1]):
                remaining_ions.append(m + k - 1)
        # each double cleavage needs two distinct matched ions/peaks
        um = _max_matching_size(remaining_ions, all_peaks - used, ion_edges)
        if got + min(remain, um // 2) <= best_d:
            return
        k = order[idx]
        for _, pa, pb in pair_options(k, used):
            used.add(pa)
            used.add(pb)
            bb1(idx + 1, used, got + 1)
            used.discard(pa)
            used.discard(pb)
        bb1(idx + 1, used, got)  # cleavage k left not doubly supported

    bb1(0, set(), 0)
    d_star = best_d

    BIG = 2 * m * o.tol + 1  # strictly exceeds any achievable total error

    # ---- greedy warm start ------------------------------------------------
    # One convex-cost flow (first ion of a cleavage free, the second priced
    # at BIG) assigning the minimum peak count demanded by the budgets.  It
    # yields a feasible (D, E) solution and seeds the branch-and-bound with
    # a tight error incumbent once the descending-D search reaches it.

    def completion_flow(doubles: dict[int, tuple[int, int]]):
        """Optimal single-ion completion after fixed double cleavages."""
        double_ions: set[int] = set()
        used: set[int] = set()
        dc = 0
        for k, (pa, pb) in doubles.items():
            double_ions.add(k - 1)
            double_ions.add(m + k - 1)
            used.update((pa, pb))
            dc += edge_cost[(k - 1, pa)] + edge_cost[(m + k - 1, pb)]
        free_ions = [i for i in range(2 * m) if i not in double_ions]
        by_clv: dict[int, list[int]] = {}
        for i in free_ions:
            by_clv.setdefault(cleavage_of[i], []).append(i)
        free_peaks = [p for p in range(n) if p not in used]
        fp = {p: j for j, p in enumerate(free_peaks)}
        d_now = len(doubles)
        c_need = max(0, m - o.max_missing - d_now)
        q_need = max(0, n - o.max_spurious - 2 * d_now)
        r = max(c_need, q_need)
        full: dict[int, int] = {}
        for k, (pa, pb) in doubles.items():
            full[k - 1] = pa
            full[m + k - 1] = pb
        if r == 0:
            return dc, full
        if r > min(len(free_ions), len(free_peaks)):
            return None
        nd = 1
        a_n, b_n, x_n = {}, {}, {}
        for k in by_clv:
            a_n[k], b_n[k], x_n[k] = nd, nd + 1, nd + 2
            nd += 3
        iin = {i: nd + 2 * j for j, i in enumerate(free_ions)}
        iout = {i: nd + 2 * j + 1 for j, i in enumerate(free_ions)}
        nd += 2 * len(free_ions)
        pb0 = nd
        nd += len(free_peaks)
        T = nd
        net = MinCost(T + 1)
        for k, ions in by_clv.items():
            net.add_edge(0, a_n[k], 1, 0)
            net.add_edge(0, b_n[k], 1, BIG)
            net.add_edge(a_n[k], x_n[k], 1, 0)
            net.add_edge(b_n[k], x_n[k], 1, 0)
            for i in ions:
                net.add_edge(x_n[k], iin[i], 1, 0)
        for i in free_ions:
            net.add_edge(iin[i], iout[i], 1, 0)
            for p in ion_edges[i]:
                if p in fp:
                    net.add_edge(iout[i], pb0 + fp[p], 1, edge_cost[(i, p)])
        for j in range(len(free_peaks)):
            net.add_edge(pb0 + j, T, 1, 0)
        sent, raw = net.flow(0, T, r)
        if sent != r:
            return None
        covered = sum(1 for k in by_clv if net.is_saturated_forward(0, a_n[k]))
        if covered < c_need:
            return None
        for i in free_ions:
            vs = net.flow_on_forward(iout[i])
            if vs:
                full[i] = free_peaks[vs[0] - pb0]
        err = raw - BIG * (r - covered) + dc
        return err, full

    # warm start computed lazily (only after confirming this candidate can
    # match the incumbent primary objective), below
    warm: Optional[tuple[int, int, dict]] = None

    # ---- stage 2: fix D = target, minimize total absolute error ----------
    #
    # A leaf fixes which cleavages are doubly supported and which peaks they
    # consume.  Remaining ("single") ion->peak assignments are then solved
    # by min-cost flow with strictly non-negative costs:
    #
    #   need extra covered cleavages c = max(0, m - max_missing - D)
    #   need extra used peaks        q = max(0, n - max_spurious - 2D)
    #   send r = max(c, q) units through
    #     S -> A_k(cap1, cost 0  ) -> ion -> peak -> T   (first ion of k)
    #     S -> B_k(cap1, cost BIG) -> ion -> peak -> T   (second ion of k)
    #   Minimizing BIG*(second-ion units) + sum error first maximizes the
    #   number of additionally covered cleavages and only then minimizes
    #   total absolute error (BIG exceeds every possible error total).  The
    #   solution is feasible iff r units flow and >= c cleavages are
    #   covered via A-tier edges.

    best: Optional[tuple[int, dict[int, int]]] = None  # (error, ion->peak)
    target_d = 0  # primary-objective target for the current bb2 round

    def leaf(doubles: dict[int, tuple[int, int]], cutoff: Optional[int]):
        nonlocal best
        tick()
        if len(doubles) != target_d:
            return
        res = completion_flow(doubles)
        if res is None:
            return
        err, full = res
        # defensive: the completion must not silently create extra pairs
        # (if it could, a higher target round would already have succeeded)
        per: dict[int, int] = {}
        for i in full:
            per[cleavage_of[i]] = per.get(cleavage_of[i], 0) + 1
        if sum(1 for c in per.values() if c == 2) != target_d:
            return
        if cutoff is not None and err > cutoff:
            return
        # within one candidate keep the strict minimum error; cross-candidate
        # ties are retained by the caller via the incumbent cutoff
        if best is None or err < best[0]:
            best = (err, full)

    def bb2(idx: int, used: set[int], used_ions: set[int], got: int,
            ecost: int, doubles: dict[int, tuple[int, int]],
            cutoff: Optional[int]) -> None:
        tick()
        remain = m - idx
        need = target_d - got
        if need > remain:
            return
        if got == target_d:
            leaf(doubles, cutoff)
            return

        future = set(order[idx:])
        # ions of already-decided (non-double) cleavages may still take a
        # single-ion assignment at the leaf, so they count for the budgets
        decided_single = [k for k in range(1, L)
                          if k not in future and k not in doubles]
        candidate_ions = [i for i in range(2 * m) if i not in used_ions]
        min_pair_costs: list[int] = []
        coverable_total = got
        for k in future:
            ia, ib = k - 1, m + k - 1
            ea = [p for p in ion_edges[ia] if p not in used]
            eb = [p for p in ion_edges[ib] if p not in used]
            if ea or eb:
                coverable_total += 1
            cmin = None
            for pa in ea:
                ca = edge_cost[(ia, pa)]
                for pb in eb:
                    if pb == pa:
                        continue
                    v = ca + edge_cost[(ib, pb)]
                    if cmin is None or v < cmin:
                        cmin = v
            if cmin is not None:
                min_pair_costs.append(cmin)
        for k in decided_single:
            ia, ib = k - 1, m + k - 1
            if (any(p not in used for p in ion_edges[ia]) or
                    any(p not in used for p in ion_edges[ib])):
                coverable_total += 1

        # peak-use upper bound: current doubles + max injective singles
        um = _max_matching_size(candidate_ions, all_peaks - used, ion_edges)
        if 2 * got + um < n - o.max_spurious:
            return
        # missing-cleavage upper bound
        if coverable_total < m - o.max_missing:
            return
        # upper bound on D still attainable (two distinct peaks per double)
        future_pair_ions = []
        for k in future:
            if any(p not in used for p in ion_edges[k - 1]):
                future_pair_ions.append(k - 1)
            if any(p not in used for p in ion_edges[m + k - 1]):
                future_pair_ions.append(m + k - 1)
        fm = _max_matching_size(future_pair_ions, all_peaks - used, ion_edges)
        if got + min(remain, fm // 2) < target_d:
            return
        # error lower bound: cheapest possible additional pairs, ignoring
        # cross-cleavage peak conflicts (relaxation -> valid lower bound)
        min_pair_costs.sort()
        lb_extra = sum(min_pair_costs[: max(need - 1, 0)])
        bound = best[0] if best is not None else cutoff
        if bound is not None:
            # a strict incumbent allows prune at equality; the cross-
            # candidate cutoff must keep equal-error ties as witnesses
            if ecost + lb_extra > bound or (best is not None and ecost + lb_extra >= bound):
                return

        k = order[idx]
        for c, pa, pb in pair_options(k, used):
            # per-option LB: need-2 further pairs from later cleavages with
            # the peaks now consumed removed
            bbound = best[0] if best is not None else cutoff
            if bbound is not None:
                blocked = used | {pa, pb}
                others = []
                for j in range(idx + 1, m):
                    kk = order[j]
                    ia, ib = kk - 1, m + kk - 1
                    cmin = None
                    for ppa in ion_edges[ia]:
                        if ppa in blocked:
                            continue
                        cca = edge_cost[(ia, ppa)]
                        for ppb in ion_edges[ib]:
                            if ppb == ppa or ppb in blocked:
                                continue
                            v = cca + edge_cost[(ib, ppb)]
                            if cmin is None or v < cmin:
                                cmin = v
                    if cmin is not None:
                        others.append(cmin)
                others.sort()
                lbv = ecost + c + sum(others[: max(need - 2, 0)])
                if lbv > bbound or (best is not None and lbv >= bbound):
                    continue  # cannot (strictly) improve; try next pair
            used.add(pa)
            used.add(pb)
            used_ions.add(k - 1)
            used_ions.add(m + k - 1)
            doubles[k] = (pa, pb)
            bb2(idx + 1, used, used_ions, got + 1, ecost + c, doubles, cutoff)
            doubles.pop(k)
            used_ions.discard(k - 1)
            used_ions.discard(m + k - 1)
            used.discard(pa)
            used.discard(pb)
        # k is not doubly supported
        bb2(idx + 1, used, used_ions, got, ecost, doubles, cutoff)

    # Stage 1's d_star ignores the missing/spurious budgets: the peak pairs
    # that maximize D can starve other cleavages of all coverage.  Try
    # targets d_star, d_star-1, ...; the first target with a budget-feasible
    # completion is the true primary optimum; its min-error leaf is the
    # secondary optimum.  The convex-flow warm start seeds the error prune
    # at the first round whose target it attains.
    #
    # A global incumbent (best D,E found in earlier candidates) prunes the
    # whole search: candidates unable to reach the incumbent D stop after
    # stage 1, and candidates at the incumbent D only explore assignments
    # whose error ties or beats the incumbent error (ties are kept, they
    # provide the lexicographic ambiguity witnesses).
    if d_star < g_d:
        return None

    # warm start: one convex flow from the empty double set; only worthwhile
    # for candidates that can reach the incumbent primary objective
    wf = completion_flow({})
    if wf is not None:
        w_err, w_map = wf
        per_clv: dict[int, int] = {}
        for i in w_map:
            per_clv[cleavage_of[i]] = per_clv.get(cleavage_of[i], 0) + 1
        warm = (sum(1 for c in per_clv.values() if c == 2), w_err, w_map)

    for target_d in range(d_star, -1, -1):
        if target_d < g_d:
            break
        cutoff = g_e if target_d == g_d else None
        if warm is not None and warm[0] == target_d and (
                cutoff is None or warm[1] <= cutoff):
            best = (warm[1], warm[2])
        else:
            best = None
        bb2(0, set(), set(), 0, 0, {}, cutoff)
        if best is not None:
            break
    if best is None:
        return None

    err_total, ion_to_peak = best

    # ---- assemble canonical result ---------------------------------------
    assignments: list[Assignment] = []
    for i, p in ion_to_peak.items():
        pid, pm = o.peaks[p]
        assignments.append(Assignment(
            cleavage=cleavage_of[i],
            ion=ion_kind[i],
            theoretical_mass=theory[i],
            peak_id=pid,
            observed_mass=pm,
            error=pm - theory[i],
        ))
    assignments.sort(key=lambda a: (a.cleavage, 0 if a.ion == "prefix" else 1,
                                    a.peak_id))
    used_peak_ids = {a.peak_id for a in assignments}
    support: dict[int, set[str]] = {}
    for a in assignments:
        support.setdefault(a.cleavage, set()).add(a.ion)
    unassigned = [pid for pid, _ in o.peaks if pid not in used_peak_ids]

    cleavage_theory = {
        k: {"prefix": theory[k - 1], "suffix": theory[m + k - 1]}
        for k in range(1, L)
    }
    return CandidateResult(
        sequence=seq_str,
        length=L,
        cleavage_theory=cleavage_theory,
        assignments=assignments,
        total_abs_error=err_total,
        double_supported_cleavages=sum(1 for v in support.values() if len(v) == 2),
        supported_cleavages=sorted(support.keys()),
        missing_cleavages=m - len(support),
        unassigned_peaks=unassigned,
        spurious_peaks=len(unassigned),
    )


# ---------------------------------------------------------------------------
# Mass-conservative sequence enumeration
# ---------------------------------------------------------------------------


def _enumerate(o: _Opts):
    """Yield (sequence_string, masses) in residue-label lexicographic order.

    Every yielded composition sums exactly to the precursor mass and has a
    length within [min_len, max_len].  Positive masses are assumed (enforced
    at the API boundary).
    """
    labels, w = o.labels, o.weights
    minw, maxw = min(w), max(w)
    target = o.precursor

    def rec(pos: int, remain: int, masses: list[int], labs: list[str]):
        if pos >= o.min_len and remain == 0:
            yield "".join(labs), masses[:]
            return
        if pos >= o.max_len or remain < minw:
            return
        after_min = max(o.min_len - pos - 1, 0)
        after_max = o.max_len - pos - 1
        for lab, wi in zip(labels, w):
            rem = remain - wi
            if rem < after_min * minw or rem > after_max * maxw:
                continue
            masses.append(wi)
            labs.append(lab)
            yield from rec(pos + 1, rem, masses, labs)
            masses.pop()
            labs.pop()

    yield from rec(0, target, [], [])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve(peaks: list[tuple[str, int]],
          precursor: int,
          residues: list[tuple[str, int]],
          length_range: tuple[int, int],
          tol: int,
          max_missing: int,
          max_spurious: int,
          candidate_limit: int = 500_000,
          node_budget: int = 500_000) -> Solution:
    residues = sorted(residues, key=lambda x: x[0])
    o = _Opts(
        labels=[lab for lab, _ in residues],
        weights=[m for _, m in residues],
        peaks=peaks,
        precursor=precursor,
        tol=tol,
        max_missing=max_missing,
        max_spurious=max_spurious,
        min_len=length_range[0],
        max_len=length_range[1],
    )

    # quick reachability check -> locatable contradiction
    if not (o.min_len * min(o.weights) <= precursor <= o.max_len * max(o.weights)):
        return Solution(
            status="no_solution",
            reason={
                "code": "MASS_UNREACHABLE",
                "message": (
                    f"precursor mass {precursor} cannot be the sum of "
                    f"{o.min_len}-{o.max_len} residue masses from the given "
                    f"alphabet (reachable range with that length interval: "
                    f"{o.min_len * min(o.weights)}.."
                    f"{o.max_len * max(o.weights)})"
                ),
                "location": "request.precursor_mass",
            },
        )

    best_key: Optional[tuple[int, int]] = None  # (D max, -E min)
    witnesses: list[CandidateResult] = []
    examined = 0
    budget = [node_budget]
    # mutable incumbent shared with the per-candidate search: [D, E]
    incumbent: Optional[list] = None
    for seq_str, masses in _enumerate(o):
        examined += 1
        if examined > candidate_limit:
            raise SequenceError(
                "CANDIDATE_LIMIT_EXCEEDED",
                f"enumeration exceeded {candidate_limit} mass-conserving "
                "candidates; narrow length_range or the residue alphabet",
                "request.length_range",
            )
        result = _evaluate(masses, seq_str, o, budget, incumbent)
        if result is None:
            continue
        key = (result.double_supported_cleavages, -result.total_abs_error)
        if best_key is None or key > best_key:
            best_key = key
            incumbent = [result.double_supported_cleavages,
                         result.total_abs_error]
            witnesses = [result]
        elif key == best_key and len(witnesses) < 2:
            witnesses.append(result)  # enumeration is lexicographic

    if best_key is None:
        return Solution(
            status="no_solution",
            reason={
                "code": "NO_FEASIBLE_EXPLANATION",
                "message": (
                    "no mass-conserving residue sequence admits a peak "
                    "assignment within the uniform tolerance that also "
                    "satisfies the missing-cleavage and spurious-peak limits"
                ),
                "location": "request",
            },
            candidates_examined=examined,
        )

    return Solution(
        status="success",
        primary=witnesses[0],
        witnesses=witnesses,
        ambiguous=len(witnesses) > 1,
        candidates_examined=examined,
    )
