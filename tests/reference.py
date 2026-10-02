"""Brute-force reference solver used only by tests.

Enumerates mass-conserving sequences independently and then explores *every*
peak assignment (each ion unmatched or matched injectively to one peak),
checking the same budgets and objectives.  No shared optimization logic with
app/solver.py, so agreement gives real correctness evidence.
"""

from __future__ import annotations


def _sequences(labels, weights, target, lmin, lmax):
    out = []

    def rec(pos, remain, masses, labs):
        if pos >= lmin and remain == 0:
            out.append(("".join(labs), tuple(masses)))
            return
        if pos >= lmax or remain <= 0:
            return
        for lab, w in zip(labels, weights):
            rec(pos + 1, remain - w, masses + [w], labs + [lab])

    rec(0, target, [], [])
    return out


def reference_solve(peaks, precursor, residues, length_range, tol,
                    max_missing, max_spurious):
    residues = sorted(residues)
    labels = [a for a, _ in residues]
    weights = [b for _, b in residues]
    lmin, lmax = length_range
    n = len(peaks)

    best_key = None  # (D, -E)
    best = {}  # sequence -> (D, E)

    for seq, masses in _sequences(labels, weights, precursor, lmin, lmax):
        L = len(masses)
        m = L - 1
        pref = []
        acc = 0
        for k in range(m):
            acc += masses[k]
            pref.append(acc)
        theory = pref + [precursor - x for x in pref]
        clv = list(range(1, L)) + list(range(1, L))

        edges = [[] for _ in range(2 * m)]
        for i in range(2 * m):
            for p, (_, pm) in enumerate(peaks):
                if abs(theory[i] - pm) <= tol:
                    edges[i].append((p, abs(theory[i] - pm)))

        best_local = None  # (D, -E)

        def dfs(i, used, assign, err):
            nonlocal best_local
            if i == 2 * m:
                a = len(assign)
                if n - a > max_spurious:
                    return
                support = {}
                for ion in assign:
                    support.setdefault(clv[ion], set()).add(
                        "p" if ion < m else "s")
                missing = m - len(support)
                if missing > max_missing:
                    return
                d = sum(1 for v in support.values() if len(v) == 2)
                key = (d, -err)
                if best_local is None or key > best_local:
                    best_local = key
                return
            # ion i unmatched
            dfs(i + 1, used, assign, err)
            for p, c in edges[i]:
                if p not in used:
                    used.add(p)
                    assign.add(i)
                    dfs(i + 1, used, assign, err + c)
                    assign.discard(i)
                    used.discard(p)

        dfs(0, set(), set(), 0)
        if best_local is None:
            continue
        if best_key is None or best_local > best_key:
            best_key = best_local
            best = {seq: best_local}
        elif best_local == best_key:
            best[seq] = best_local

    if best_key is None:
        return None
    order = sorted(best)
    return best_key[0], -best_key[1], order
