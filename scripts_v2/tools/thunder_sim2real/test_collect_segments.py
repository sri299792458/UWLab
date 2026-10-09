"""CPU checks for segmented excitations in collect_thunder.py (R231): the single pinned chirp is unchanged, segments run
back to back from the start pose, and the safety guards (<= 3 Hz, within the recorded full-amplitude chirp) hold."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workstation"))
import collect_thunder  # noqa: E402

FULL = json.loads((HERE / "workstation/collection.remapped_full_amplitude_swe_candidate.json").read_text())


def pinned_chirp(duration, f0, f1, amps):
    """The pinned UW construction, written out independently of collect_thunder."""
    dt = 0.002
    count = int(duration / dt)
    t = np.linspace(0, duration, count)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    env = np.ones(count)
    env[:int(2. / dt)] = np.linspace(0, 1, int(2. / dt))
    env[-int(3. / dt):] = np.linspace(1, 0, int(3. / dt))
    return np.stack([amps[i] * env * np.sin(phase + i * np.pi / 3) for i in range(6)], axis=1)


class SegmentTests(unittest.TestCase):
    def test_single_chirp_is_unchanged(self):
        offsets = collect_thunder.make_plan(FULL)
        np.testing.assert_array_equal(offsets, pinned_chirp(8.0, 0.1, 3.0, np.asarray(FULL["amplitudes_m_rad"])))

    def test_segments_run_back_to_back_from_the_start_pose(self):
        for name in ("collection.policy_regime_chirp.json",):
            config = json.loads((HERE / "workstation" / name).read_text())
            offsets = collect_thunder.make_plan(config)
            expected = np.concatenate([pinned_chirp(s["duration_s"], s["f0_hz"], s["f1_hz"], np.asarray(s["amplitudes_m_rad"]))
                                       for s in config["segments"]])
            np.testing.assert_array_equal(offsets, expected)
            n0 = int(config["segments"][0]["duration_s"] / 0.002)
            np.testing.assert_allclose(offsets[[0, n0 - 1, n0, -1]], 0.0, atol=1e-12)   # each segment starts/ends at 0

    def test_guards(self):
        base = json.loads((HERE / "workstation/collection.policy_regime_chirp.json").read_text())
        too_fast = copy.deepcopy(base); too_fast["segments"][0]["f1_hz"] = 3.5
        too_big = copy.deepcopy(base); too_big["segments"][1]["amplitudes_m_rad"][4] = 0.6
        mixed = copy.deepcopy(base); mixed["duration_s"] = 8.0
        for config in (too_fast, too_big, mixed):
            with self.assertRaises(ValueError):
                collect_thunder.make_plan(config)


if __name__ == "__main__":
    unittest.main()
