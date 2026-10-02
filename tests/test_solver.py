"""Correctness tests for the solver core.

Includes randomized cross-validation against an independent brute-force
reference (tests/reference.py).
"""

from __future__ import annotations

import random

import pytest

from app.solver import solve

from tests.reference import reference_solve


def run(peaks, precursor, residues, lrange, tol=0, mm=2, ms=4):
    return solve(
        peaks=[(p[0], p[1]) for p in peaks],
        precursor=precursor,
        residues=residues,
        length_range=lrange,
        tol=tol,
        max_missing=mm,
        max_spurious=ms,
    )


# --------------------------------------------------------------------------
# Hand-built deterministic cases
# --------------------------------------------------------------------------


def test_exact_fragment_ladder():
    # residues A=10 B=20 C=30 D=40 ; peptide "ABC" = 60, length 3
    # cleavages: k1 prefix 10 / suffix 50, k2 prefix 30 / suffix 30
    # only one peak at mass 30, so k2 gets single-side support; peak p4@31
    # is outside tolerance 0 and stays unassigned.
    peaks = [
        ("p1", 10), ("p2", 50), ("p3", 30), ("p4", 31),
        ("p5", 11), ("p6", 51), ("p7", 9), ("p8", 777),
    ]
    residues = [("A", 10), ("B", 20), ("C", 30), ("D", 40)]
    sol = run(peaks, 60, residues, (3, 3), tol=0, mm=0, ms=5)
    assert sol.status == "success"
    c = sol.primary
    assert c.sequence == "ABC"
    assert c.double_supported_cleavages == 1
    assert c.total_abs_error == 0
    assigned = {a.peak_id for a in c.assignments}
    assert assigned == {"p1", "p2", "p3"}
    assert c.missing_cleavages == 0
    assert c.spurious_peaks == 5
    assert len(assigned) == len(c.assignments)  # each peak used at most once


def test_peak_used_once_distinguishes_from_greedy():
    # two different ions both closest to the SAME peak must not both claim
    # it; sequence "AB" = 30, ions 10 and 20; a single peak at 15 is
    # equidistant (tol 5) to both, peak at 20 supports suffix.
    peaks = [("q1", 15), ("q2", 20), ("q3", 1), ("q4", 2),
             ("q5", 3), ("q6", 4), ("q7", 5), ("q8", 6)]
    residues = [("A", 10), ("B", 20), ("C", 13), ("D", 17)]
    sol = run(peaks, 30, residues, (2, 2), tol=5, mm=0, ms=6)
    assert sol.status == "success"
    assert sol.primary.sequence == "AB"
    ids = [a.peak_id for a in sol.primary.assignments]
    assert len(ids) == len(set(ids))
    # both ions supported with distinct peaks: suffix(20)->q2 err0,
    # prefix(10)->q8@6 err4; q1@15 stays unassigned (peak used once)
    assert sol.primary.double_supported_cleavages == 1
    assert sol.primary.total_abs_error == 4


def test_mass_conservation_no_greedy_chain():
    # greedy nearest-residue per peak could chain nonsense; only A+B sums 30
    peaks = [(f"x{i}", v) for i, v in enumerate(
        [10, 20, 11, 19, 9, 21, 100, 200])]
    residues = [("A", 10), ("B", 20), ("C", 45), ("D", 47)]
    sol = run(peaks, 30, residues, (2, 2), tol=1, mm=0, ms=6)
    assert sol.status == "success"
    assert sol.primary.sequence == "AB"
    assert sol.primary.double_supported_cleavages == 1


def test_no_solution_missing_budget():
    # peptide "AB"=30 ions 10/20; give no peaks near either ion
    peaks = [(f"n{i}", v) for i, v in enumerate(
        [100, 101, 102, 103, 104, 105, 106, 107])]
    residues = [("A", 10), ("B", 20), ("C", 30), ("D", 40)]
    sol = run(peaks, 30, residues, (2, 2), tol=0, mm=0, ms=8)
    assert sol.status == "no_solution"
    assert sol.reason["code"] == "NO_FEASIBLE_EXPLANATION"
    assert "location" in sol.reason


def test_mass_unreachable_is_locatable():
    peaks = [(f"n{i}", 100 + i) for i in range(8)]
    residues = [("A", 10), ("B", 20), ("C", 30), ("D", 40)]
    sol = run(peaks, 9999, residues, (2, 3), tol=0, mm=2, ms=8)
    assert sol.status == "no_solution"
    assert sol.reason["code"] == "MASS_UNREACHABLE"
    assert sol.reason["location"] == "request.precursor_mass"


def test_ambiguity_returns_two_lexicographic_witnesses():
    # Length-2 sequences AB and BA have identical ion fingerprints
    # ({10, 20}); peaks at 11/19 (tol 1) support both with total error 2.
    # The API must surface the ambiguity and return the lexicographically
    # first two distinct sequences.  Cross-composition ambiguity is covered
    # by the randomized brute-force cross-validation below.
    peaks = [("a", 11), ("b", 19), ("c", 100), ("d", 101),
             ("e", 102), ("f", 103), ("g", 104), ("h", 105)]
    residues = [("A", 10), ("B", 20), ("C", 12), ("D", 18)]
    sol = run(peaks, 30, residues, (2, 2), tol=1, mm=0, ms=6)
    assert sol.status == "success"
    assert sol.ambiguous is True
    seqs = [w.sequence for w in sol.witnesses]
    assert seqs == ["AB", "BA"]
    for w in sol.witnesses:
        assert w.double_supported_cleavages == 1
        assert w.total_abs_error == 2


def test_spurious_peak_limit_forces_assignment():
    # same spectrum but demand almost all peaks assigned -> impossible
    peaks = [("a", 11), ("b", 19), ("c", 100), ("d", 101),
             ("e", 102), ("f", 103), ("g", 104), ("h", 105)]
    residues = [("A", 10), ("B", 20), ("C", 12), ("D", 18)]
    sol = run(peaks, 30, residues, (2, 2), tol=1, mm=0, ms=2)
    assert sol.status == "no_solution"


def test_tolerance_error_minimization():
    # candidate ions 10/20; peaks 10(exact), 19(err1), 20(exact)
    # optimum error 0 with both exact peaks; must not pick 19
    peaks = [("a", 10), ("b", 20), ("c", 19), ("d", 100),
             ("e", 101), ("f", 102), ("g", 103), ("h", 104)]
    residues = [("A", 10), ("B", 20), ("C", 33), ("D", 37)]
    sol = run(peaks, 30, residues, (2, 2), tol=1, mm=0, ms=6)
    assert sol.primary.sequence == "AB"
    assert sol.primary.total_abs_error == 0
    ids = {a.peak_id: a for a in sol.primary.assignments}
    assert set(ids) == {"a", "b"}


def test_primary_objective_beats_error():
    # Two feasible explanations compete:
    #   ABC [10,20,30] ions {10, 30, 50, 30} -> D=2 with peaks 29/31, E=2
    #   AAD [10,10,40] ions {10, 20, 50, 40} -> D=1 using exact peaks, E=0
    # Maximizing doubly-supported cleavages must dominate; ABC wins even
    # though its total error is larger.
    peaks = [("t1", 10), ("t2", 50), ("t3", 29), ("t4", 31),
             ("u1", 90), ("u2", 91), ("u3", 92), ("u4", 93)]
    residues = [("A", 10), ("B", 20), ("C", 30), ("D", 40)]
    sol = run(peaks, 60, residues, (3, 3), tol=1, mm=1, ms=6)
    assert sol.status == "success"
    assert sol.primary.sequence == "ABC"
    assert sol.primary.double_supported_cleavages == 2
    assert sol.primary.total_abs_error == 2


# --------------------------------------------------------------------------
# Randomized cross-validation vs independent brute force
# --------------------------------------------------------------------------


def _gen_case(rng):
    nres = rng.randint(4, 5)
    base = sorted(rng.sample(range(5, 40), nres))
    labels = [chr(ord("A") + i) for i in range(nres)]
    residues = list(zip(labels, base))
    L = rng.randint(2, 5)
    seq_idx = [rng.randrange(nres) for _ in range(L)]
    masses = [base[i] for i in seq_idx]
    target = sum(masses)
    pref = []
    acc = 0
    for m_ in masses[:-1]:
        acc += m_
        pref.append(acc)
    ions = pref + [target - x for x in pref]

    npeaks = rng.randint(8, 12)
    peaks = []
    # include some true ions (with jitter), some spurious
    shuffled = ions[:]
    rng.shuffle(shuffled)
    n_true = rng.randint(1, min(len(ions), npeaks - 2))
    vals = []
    for tm in shuffled[:n_true]:
        vals.append(tm + rng.choice([-1, 0, 0, 1]))
    while len(vals) < npeaks:
        v = rng.randint(1, max(target + 10, 50))
        vals.append(v)
    rng.shuffle(vals)
    for i, v in enumerate(vals):
        pid = f"pk{i}"
        peaks.append((pid, v))
    tol = rng.choice([0, 1, 2])
    mm = rng.randint(0, 3)
    ms = rng.randint(0, npeaks)
    return peaks, target, residues, (L, L), tol, mm, ms


@pytest.mark.parametrize("seed", range(120))
def test_matches_brute_force(seed):
    rng = random.Random(seed)
    args = _gen_case(rng)
    ref = reference_solve(*args)
    sol = solve(peaks=args[0], precursor=args[1], residues=args[2],
                length_range=args[3], tol=args[4],
                max_missing=args[5], max_spurious=args[6])
    if ref is None:
        assert sol.status == "no_solution", (seed, sol)
        return
    d, e, order = ref
    assert sol.status == "success", (seed, sol.reason)
    assert sol.primary.double_supported_cleavages == d, seed
    assert sol.primary.total_abs_error == e, (seed, sol.primary.total_abs_error, e)
    assert sol.primary.sequence == order[0], (seed, sol.primary.sequence, order)
    if len(order) > 1:
        assert sol.ambiguous is True, seed
        assert [w.sequence for w in sol.witnesses] == order[:2], seed
    else:
        assert sol.ambiguous is False, seed


def test_assignment_invariants_random():
    # on random solvable cases every response must be structurally valid
    rng = random.Random(4242)
    checked = 0
    attempts = 0
    while checked < 25 and attempts < 400:
        attempts += 1
        args = _gen_case(rng)
        sol = solve(peaks=args[0], precursor=args[1], residues=args[2],
                    length_range=args[3], tol=args[4],
                    max_missing=args[5], max_spurious=args[6])
        if sol.status != "success":
            continue
        for w in sol.witnesses:
            ids = [a.peak_id for a in w.assignments]
            assert len(ids) == len(set(ids))  # peak used at most once
            for a in w.assignments:
                assert abs(a.error) <= args[4]
                assert a.observed_mass - a.theoretical_mass == a.error
            assert w.missing_cleavages <= args[5]
            assert w.spurious_peaks <= args[6]
            # complementary masses really are complementary
            for k, cl in w.cleavage_theory.items():
                assert cl["prefix"] + cl["suffix"] == args[1]
        checked += 1
    assert checked == 25
