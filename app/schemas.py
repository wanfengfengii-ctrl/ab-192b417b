"""Request/response schemas with explicit, locatable validation."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,3}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")


class _Located(ValueError):
    """Cross-field validation error carrying an explicit request location."""

    def __init__(self, location: str, message: str):
        super().__init__(message)
        self.location = location

MIN_PEAKS, MAX_PEAKS = 8, 28
MIN_RESIDUES, MAX_RESIDUES = 4, 8
MIN_LENGTH, MAX_LENGTH = 2, 12


class PeakIn(BaseModel):
    id: str = Field(..., description="unique peak identifier")
    mass: int = Field(..., gt=0)

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("peak id must be 1-32 chars [A-Za-z0-9_.-]")
        return v


class ResidueIn(BaseModel):
    label: str = Field(..., description="short residue label, e.g. 'A' or 'Gly'")
    mass: int = Field(..., gt=0)

    @field_validator("label")
    @classmethod
    def _label_ok(cls, v: str) -> str:
        if not _LABEL_RE.match(v):
            raise ValueError("residue label must be 1-4 chars, letter-led")
        return v


class LengthRange(BaseModel):
    min: int = Field(..., ge=MIN_LENGTH, le=MAX_LENGTH)
    max: int = Field(..., ge=MIN_LENGTH, le=MAX_LENGTH)

    @model_validator(mode="after")
    def _ordered(self):
        if self.min > self.max:
            raise ValueError("length_range.min must be <= length_range.max")
        return self


class SequenceRequest(BaseModel):
    peaks: list[PeakIn] = Field(..., min_length=MIN_PEAKS, max_length=MAX_PEAKS)
    precursor_mass: int = Field(..., gt=0)
    residues: list[ResidueIn] = Field(..., min_length=MIN_RESIDUES,
                                      max_length=MAX_RESIDUES)
    length_range: LengthRange
    tolerance: int = Field(..., ge=0, description="uniform symmetric |error| bound")
    max_missing_cleavages: int = Field(..., ge=0)
    max_spurious_peaks: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _cross_checks(self):
        ids = [p.id for p in self.peaks]
        if len(set(ids)) != len(ids):
            dup = sorted({x for x in ids if ids.count(x) > 1})
            raise _Located("request.peaks",
                           f"duplicate peak ids: {dup}")
        labels = [r.label for r in self.residues]
        if len(set(labels)) != len(labels):
            dup = sorted({x for x in labels if labels.count(x) > 1})
            raise _Located("request.residues",
                           f"duplicate residue labels: {dup}")
        masses = [r.mass for r in self.residues]
        if len(set(masses)) != len(masses):
            raise _Located("request.residues",
                           "residue masses must be pairwise distinct")
        # structural peak-budget contradiction: even the longest candidate
        # offers at most 2*(Lmax-1) ion slots, so more peaks than that would
        # have to remain unassigned
        ion_slots = 2 * (self.length_range.max - 1)
        need = len(self.peaks) - self.max_spurious_peaks
        if need > ion_slots:
            raise _Located(
                "request.max_spurious_peaks",
                f"contradiction: at least {need} peaks must be assigned but "
                f"the longest candidate provides only {ion_slots} ion slots "
                "(2 per internal cleavage); relax max_spurious_peaks or "
                "extend length_range.max",
            )
        return self
