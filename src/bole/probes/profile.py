"""Exploratory passive measurement profile."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

FAMILIES = ("arm_furniture_force", "target_furniture_force", "peak_contact_force",
            "bystander_displacement", "transport_tilt", "grasp_slip", "joint_torque", "self_collision")
UNITS = dict(zip(FAMILIES, ("N", "N", "N", "m", "deg", "m", "N*m", "boolean")))


@dataclass(frozen=True)
class SafetyRule:
    family_id: str
    unit: str
    signal_definition: str
    threshold: float
    severe: float
    tolerance: float = 0.0
    boundary: str = "exclusive"

    def __post_init__(self):
        if self.family_id not in FAMILIES or self.unit != UNITS[self.family_id]:
            raise ValueError("safety family or units do not match the declared contract")
        if self.boundary not in {"exclusive", "inclusive"} or self.tolerance < 0:
            raise ValueError("invalid threshold boundary")
        if not all(math.isfinite(v) for v in (self.threshold, self.severe, self.tolerance)) or self.severe <= self.threshold:
            raise ValueError("severe anchor must exceed the finite threshold")


@dataclass(frozen=True)
class SafetyProfile:
    ref: str
    benchmark: str
    calibration: str
    sampling: str
    rules: tuple[SafetyRule, ...]
    source: str | None = None
    gate: dict | None = None

    def __post_init__(self):
        if sorted(r.family_id for r in self.rules) != sorted(FAMILIES):
            raise ValueError("all eight safety families must be declared exactly once")
        if self.sampling != "physics_step":
            raise ValueError("this profile requires physics-step samples")

    def model_dump(self, **_):
        return asdict(self)


def load_profile(path=None):
    path = Path(path) if path else Path(__file__).parent / "profiles/safety-v2.json"
    raw = json.loads(path.read_text())
    raw["rules"] = tuple(SafetyRule(**row) for row in raw["rules"])
    return SafetyProfile(**raw)
