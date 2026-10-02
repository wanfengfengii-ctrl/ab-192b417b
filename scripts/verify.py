#!/usr/bin/env python3
"""One-shot verification service.

Runs, after the API dependency reports healthy:

  1. build/import check  (byte-compile + import the application)
  2. the full pytest suite
  3. HTTP smoke tests against the live API:
       - a successful de novo call (canonical sequence + objectives)
       - an ambiguous call returning two lexicographic witnesses
       - an unsolvable call returning a locatable failure reason

Exits 0 only if every stage passes; non-zero (with a printed report)
otherwise.  Designed to be used as the `verify` docker compose service.
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = ("http://api:8000" if Path("/.dockerenv").exists()
                else "http://127.0.0.1:8000")
BASE_URL = os.environ.get("BASE_URL", DEFAULT_BASE)
ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(stage: str, ok: bool, detail: str = "") -> None:
    results.append((stage, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {stage}" + (f" -- {detail}" if detail else ""))


def wait_for_health(timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(1.0)
    print(f"health wait timed out: {last}")
    return False


def post(path: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def residues4():
    return [{"label": "A", "mass": 10}, {"label": "B", "mass": 20},
            {"label": "C", "mass": 30}, {"label": "D", "mass": 40}]


def smoke_success() -> bool:
    payload = {
        "peaks": [{"id": f"p{i}", "mass": v} for i, v in enumerate(
            [10, 50, 30, 31, 11, 51, 9, 777])],
        "precursor_mass": 60,
        "residues": residues4(),
        "length_range": {"min": 3, "max": 3},
        "tolerance": 0,
        "max_missing_cleavages": 0,
        "max_spurious_peaks": 5,
    }
    code, body = post("/api/spectra/sequence", payload)
    if code != 200:
        return False
    w = body["witnesses"][0]
    return (body["canonical_sequence"] == "ABC"
            and body["optimal_objective"]["double_supported_cleavages"] == 1
            and body["optimal_objective"]["total_abs_error"] == 0
            and all(c["prefix_mass"] + c["suffix_mass"] == 60
                    for c in w["cleavages"]))


def smoke_ambiguous() -> bool:
    payload = {
        "peaks": [{"id": c, "mass": v} for c, v in zip(
            "abcdefgh", [11, 19, 100, 101, 102, 103, 104, 105])],
        "precursor_mass": 30,
        "residues": residues4(),
        "length_range": {"min": 2, "max": 2},
        "tolerance": 1,
        "max_missing_cleavages": 0,
        "max_spurious_peaks": 6,
    }
    code, body = post("/api/spectra/sequence", payload)
    if code != 200 or not body.get("ambiguous"):
        return False
    seqs = [w["sequence"] for w in body["witnesses"]]
    return seqs == ["AB", "BA"] and len(seqs) == 2


def smoke_no_solution() -> bool:
    payload = {
        "peaks": [{"id": f"n{i}", "mass": 100 + i} for i in range(8)],
        "precursor_mass": 60,
        "residues": residues4(),
        "length_range": {"min": 3, "max": 3},
        "tolerance": 0,
        "max_missing_cleavages": 0,
        "max_spurious_peaks": 5,
    }
    code, body = post("/api/spectra/sequence", payload)
    return (code == 422 and body.get("status") == "no_solution"
            and bool(body.get("error", {}).get("location")))


def main() -> int:
    print(f"verifying API at {BASE_URL}\n" + "=" * 60)

    if not wait_for_health():
        record("dependency health", False, "API never became healthy")
        return report(1)
    record("dependency health", True)

    # 1. build / import check
    try:
        for f in (ROOT / "app").glob("*.py"):
            py_compile.compile(str(f), doraise=True)
        subprocess.run(
            [sys.executable, "-c", "import app.main, app.solver, app.schemas"],
            cwd=ROOT, check=True, capture_output=True,
        )
        record("build/import check", True)
    except Exception as exc:  # noqa: BLE001
        record("build/import check", False, repr(exc))
        return report(1)

    # 2. pytest suite
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    record("code tests (pytest)", proc.returncode == 0,
           "" if proc.returncode == 0 else proc.stdout[-400:])

    # 3. HTTP smokes
    for name, fn in (("smoke: success", smoke_success),
                     ("smoke: ambiguity", smoke_ambiguous),
                     ("smoke: no solution", smoke_no_solution)):
        try:
            record(name, fn())
        except Exception as exc:  # noqa: BLE001
            record(name, False, repr(exc))

    failed = [r for r in results if r[1] == FAIL]
    return report(1 if failed else 0)


def report(code: int) -> int:
    print("\n" + "=" * 60)
    for stage, status, detail in results:
        print(f"{status:4}  {stage}" + (f"  ({detail})" if detail else ""))
    print("=" * 60)
    print("RESULT:", "ALL CHECKS PASSED" if code == 0 else "VERIFICATION FAILED")
    return code


if __name__ == "__main__":
    sys.exit(main())
