"""Portable validation for Thunder's fitted dynamics (no Isaac Sim imports)."""
import hashlib
import json
import math
from pathlib import Path

CALIBRATION_HASH = "calib_10185139869934003756"
PARAMETER_KEYS = ("armature", "static_friction", "dynamic_ratio", "viscous_friction")


def six_numbers(value, label, *, positive=False, upper=None):
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{label} must have six entries")
    values = tuple(float(x) for x in value)
    if not all(math.isfinite(x) and (x > 0 if positive else x >= 0) for x in values):
        raise ValueError(f"Invalid {label}: {values}")
    if upper is not None and any(x > upper for x in values):
        raise ValueError(f"{label} exceeds {upper}")
    return values


def validate_profile(profile, *, allow_synthetic=False):
    if profile.get("schema_version") != 1 or profile.get("robot") != "thunder":
        raise ValueError("Expected a version-1 Thunder sysid profile")
    if profile.get("calibration_hash") != CALIBRATION_HASH:
        raise ValueError("Profile calibration does not match Thunder")
    kind = profile.get("source_kind")
    if kind != "real_robot" and not (allow_synthetic and kind == "synthetic_test"):
        raise ValueError("Stage 2 requires parameters fitted to Thunder real-robot data")
    for key in PARAMETER_KEYS:
        six_numbers(profile["sysid"][key], key, upper=1 if key == "dynamic_ratio" else None)
    controller = profile["controller"]
    for key in ("motion_stiffness", "motion_damping_ratio", "torque_max"):
        six_numbers(controller[key], key, positive=True)
    if any(x > lim for x, lim in zip(controller["torque_max"], [150, 150, 150, 28, 28, 28])):
        raise ValueError("Controller torque limits exceed the UR5e configuration")
    if profile.get("fit_dt_s") != 0.002:
        raise ValueError("Expected 500 Hz system identification")
    delay = profile.get("fitted_delay_steps")
    if not isinstance(delay, int) or delay < 0:
        raise ValueError("fitted_delay_steps must be a nonnegative integer")
    if not math.isclose(profile.get("fitted_delay_seconds", -1), delay * 0.002):
        raise ValueError("Delay steps/seconds disagree")
    # Preserve UWLab's Stage-2 delay randomization. The fitted 500 Hz step count
    # is reported separately and must never be copied directly into 120 Hz.
    if profile.get("stage2_delay") != {"mode": "upstream_range", "physics_hz": 120, "steps": [0, 1]}:
        raise ValueError("Expected the documented UWLab 0–1 step Stage-2 delay range")
    if not profile.get("record_sha256") or not profile.get("fit_sha256"):
        raise ValueError("Profile must identify its source recording and fit")
    return profile


def load_profile(path):
    if not path:
        raise ValueError("Set THUNDER_SYSID_PROFILE to the exported Thunder sysid_profile.json before Stage 2")
    path = Path(path).expanduser().resolve()
    profile = validate_profile(json.loads(path.read_text()))
    return profile, hashlib.sha256(path.read_bytes()).hexdigest()
