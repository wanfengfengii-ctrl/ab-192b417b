"""End-to-end tests for POST /api/spectra/sequence."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**over):
    p = {
        "peaks": [
            {"id": "p1", "mass": 10}, {"id": "p2", "mass": 50},
            {"id": "p3", "mass": 30}, {"id": "p4", "mass": 31},
            {"id": "p5", "mass": 11}, {"id": "p6", "mass": 51},
            {"id": "p7", "mass": 9}, {"id": "p8", "mass": 777},
        ],
        "precursor_mass": 60,
        "residues": [
            {"label": "A", "mass": 10}, {"label": "B", "mass": 20},
            {"label": "C", "mass": 30}, {"label": "D", "mass": 40},
        ],
        "length_range": {"min": 3, "max": 3},
        "tolerance": 0,
        "max_missing_cleavages": 0,
        "max_spurious_peaks": 5,
    }
    p.update(over)
    return p


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_success_shape():
    r = client.post("/api/spectra/sequence", json=base_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["canonical_sequence"] == "ABC"
    assert body["optimal_objective"] == {
        "double_supported_cleavages": 1, "total_abs_error": 0}
    w = body["witnesses"][0]
    # per-cleavage complementary theoretical masses
    for cl in w["cleavages"]:
        assert cl["prefix_mass"] + cl["suffix_mass"] == 60
    # assignments well-formed and peaks unique
    ids = [a["peak_id"] for a in w["peak_assignments"]]
    assert len(ids) == len(set(ids))
    for a in w["peak_assignments"]:
        assert a["observed_mass"] - a["theoretical_mass"] == a["error"]
        assert a["abs_error"] == abs(a["error"])
    assert set(["cleavage", "ion", "theoretical_mass", "peak_id",
                "observed_mass", "error"]).issubset(
                    set(w["peak_assignments"][0].keys()))


def test_ambiguous_returns_two_witnesses():
    payload = base_payload(
        peaks=[{"id": c, "mass": v} for c, v in zip(
            "abcdefgh", [11, 19, 100, 101, 102, 103, 104, 105])],
        precursor_mass=30,
        length_range={"min": 2, "max": 2},
        tolerance=1,
        max_spurious_peaks=6,
    )
    r = client.post("/api/spectra/sequence", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["ambiguous"] is True
    assert len(body["witnesses"]) == 2
    seqs = [w["sequence"] for w in body["witnesses"]]
    assert seqs == ["AB", "BA"]


def test_no_solution_is_locatable_422():
    payload = base_payload(
        peaks=[{"id": f"n{i}", "mass": 100 + i} for i in range(8)])
    r = client.post("/api/spectra/sequence", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "no_solution"
    err = body["error"]
    assert err["code"] == "NO_FEASIBLE_EXPLANATION"
    assert err["location"]


def test_mass_unreachable_422():
    payload = base_payload(precursor_mass=9999)
    r = client.post("/api/spectra/sequence", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MASS_UNREACHABLE"
    assert "precursor_mass" in r.json()["error"]["location"]


def test_validation_errors_are_locatable():
    # duplicate peak ids
    p = base_payload()
    p["peaks"][1]["id"] = "p1"
    r = client.post("/api/spectra/sequence", json=p)
    assert r.status_code == 400
    errs = r.json()["errors"]
    assert any("peaks" in e["location"] for e in errs)

    # residue masses not distinct
    p = base_payload()
    p["residues"][1]["mass"] = 10
    r = client.post("/api/spectra/sequence", json=p)
    assert r.status_code == 400

    # structural contradiction: not enough ion slots
    p = base_payload(max_spurious_peaks=0)
    r = client.post("/api/spectra/sequence", json=p)
    assert r.status_code == 400
    assert any("contradiction" in e["message"] for e in r.json()["errors"])


def test_bad_counts_rejected():
    p = base_payload()
    p["peaks"] = p["peaks"][:7]  # below 8
    r = client.post("/api/spectra/sequence", json=p)
    assert r.status_code == 400
    p2 = base_payload()
    p2["residues"] = p2["residues"][:3]  # below 4
    r2 = client.post("/api/spectra/sequence", json=p2)
    assert r2.status_code == 400
