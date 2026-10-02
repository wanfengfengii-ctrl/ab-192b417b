"""POST /api/spectra/sequence -- de novo short-peptide sequencing API."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .schemas import SequenceRequest
from .solver import SequenceError, solve

app = FastAPI(
    title="Peptide De Novo Sequencing",
    version="1.0.0",
    description="Recover short peptide sequences from tandem MS spectra "
                "with missing peaks and spurious peaks under strict mass "
                "conservation and peak-use-once assignment.",
)


def _result_payload(sol):
    def witness(c):
        return {
            "sequence": c.sequence,
            "length": c.length,
            "objective": {
                "double_supported_cleavages": c.double_supported_cleavages,
                "total_abs_error": c.total_abs_error,
            },
            "cleavages": [
                {
                    "k": k,
                    "prefix_mass": c.cleavage_theory[k]["prefix"],
                    "suffix_mass": c.cleavage_theory[k]["suffix"],
                }
                for k in sorted(c.cleavage_theory)
            ],
            "peak_assignments": [
                {
                    "peak_id": a.peak_id,
                    "observed_mass": a.observed_mass,
                    "cleavage": a.cleavage,
                    "ion": a.ion,
                    "theoretical_mass": a.theoretical_mass,
                    "error": a.error,
                    "abs_error": abs(a.error),
                }
                for a in c.assignments
            ],
            "unassigned_peaks": c.unassigned_peaks,
            "missing_cleavages": c.missing_cleavages,
            "spurious_peaks": c.spurious_peaks,
        }

    return {
        "status": "success",
        "ambiguous": sol.ambiguous,
        "candidates_examined": sol.candidates_examined,
        "optimal_objective": sol.primary and {
            "double_supported_cleavages": sol.primary.double_supported_cleavages,
            "total_abs_error": sol.primary.total_abs_error,
        },
        "canonical_sequence": sol.primary.sequence,
        "witnesses": [witness(c) for c in sol.witnesses],
    }


@app.post("/api/spectra/sequence")
def sequence_spectrum(payload: SequenceRequest):
    peaks = [(p.id, p.mass) for p in payload.peaks]
    residues = [(r.label, r.mass) for r in payload.residues]
    sol = solve(
        peaks=peaks,
        precursor=payload.precursor_mass,
        residues=residues,
        length_range=(payload.length_range.min, payload.length_range.max),
        tol=payload.tolerance,
        max_missing=payload.max_missing_cleavages,
        max_spurious=payload.max_spurious_peaks,
    )
    if sol.status != "success":
        return JSONResponse(status_code=422, content={
            "status": "no_solution",
            "error": sol.reason,
            "candidates_examined": sol.candidates_examined,
        })
    return _result_payload(sol)


@app.exception_handler(SequenceError)
async def sequence_error_handler(request: Request, exc: SequenceError):
    return JSONResponse(status_code=422, content={
        "status": "error",
        "error": {
            "code": exc.code,
            "message": exc.message,
            "location": exc.location,
        },
    })


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        parts = [str(x) for x in e["loc"]]
        if parts and parts[0] in ("body", "query", "path"):
            parts = parts[1:]
        loc = "request." + ".".join(parts) if parts else "request"
        # cross-field validators attach an explicit locatable location
        ctx_err = e.get("ctx", {}).get("error")
        if ctx_err is not None and hasattr(ctx_err, "location"):
            loc = ctx_err.location
        errors.append({
            "code": e["type"],
            "message": e["msg"].removeprefix("Value error, "),
            "location": loc,
        })
    return JSONResponse(status_code=400, content={
        "status": "invalid_request",
        "errors": errors,
    })


@app.get("/health")
def health():
    return {"status": "ok"}
