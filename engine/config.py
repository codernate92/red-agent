"""Configuration for the vulnerability engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScanConfig(BaseModel):
    """Execution knobs for the vulnerability engine."""

    recon_probes_per_category: int = Field(default=3, ge=1, le=10)
    attack_depth: Literal["quick", "standard", "deep"] = "standard"
    confirmation_runs: int = Field(default=3, ge=1, le=10)
    min_reproduction_rate: float = Field(default=0.66, ge=0.0, le=1.0)
    severity_threshold: float = Field(default=4.0, ge=0.0, le=10.0)
    include_prefill: bool = True
    max_concurrent_requests: int = Field(default=5, ge=1, le=100)
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @model_validator(mode="after")
    def validate_confirmation_window(self) -> "ScanConfig":
        if self.confirmation_runs > 1:
            required_hits = self.min_reproduction_rate * self.confirmation_runs
            if required_hits > self.confirmation_runs:
                raise ValueError("min_reproduction_rate exceeds the confirmation window")
        return self
