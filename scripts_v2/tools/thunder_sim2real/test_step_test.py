"""CPU checks for the step-response test: plan validation, robot-side friction scales, full mocked execution, analysis.

No RTDE sockets: the robot interfaces, scheduler and clock are test doubles (same approach as test_contract.py). The fake
robot never moves, so every step should analyze as stop_short == commanded.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workstation"))
import collect_thunder  # noqa: E402
import step_test_thunder  # noqa: E402


def base_config(**changes):
    config = json.loads((HERE / "workstation/collection.step_test.json").read_text())
    config.update(robot_ip="test-double-only", polyscope_version="test-double-only")
    # Short schedule for the mocked run: one translation and one rotation axis, one size each, 0.5 s holds.
    config.update(step_axes=["y", "rz"], translation_steps_m=[0.01], rotation_steps_rad=[0.0698132],
                  initial_hold_s=0.5, step_hold_s=0.5, return_hold_s=0.5)
    config.update(changes)
    return config


class StepTestTests(unittest.TestCase):
    def test_plan_rejects_large_steps_and_unknown_friction(self):
        with self.assertRaises(ValueError):
            step_test_thunder.make_plan(base_config(translation_steps_m=[0.08]), "off")
        with self.assertRaises(ValueError):
            step_test_thunder.make_plan(base_config(), "on")
        with self.assertRaises(ValueError):
            collect_thunder.friction_scales({"direct_torque_params": {"viscous_scale": [1.2] * 6, "coulomb_scale": [0.5] * 6}})

    def test_plan_schedule_and_guard(self):
        config, offsets = step_test_thunder.make_plan(base_config(), "off")
        schedule = config["step_test"]["schedule"]
        self.assertEqual([(s["axis"], round(s["value"], 6)) for s in schedule],
                         [("y", 0.01), ("y", -0.01), ("rz", 0.069813), ("rz", -0.069813)])
        self.assertEqual(len(offsets), 250 + 4 * (250 + 250))
        first = schedule[0]
        self.assertTrue(np.allclose(offsets[first["start_sample"]:first["return_start_sample"]], [0, 0.01, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(offsets[first["return_start_sample"]:first["return_start_sample"] + 250], 0))
        guard = np.asarray(config["joint_excursion_limit_rad"])
        self.assertTrue(np.all(guard >= np.deg2rad(5.0)))
        self.assertNotIn("direct_torque_params", config)          # off = fitted zeros

    def run_mock(self, friction):
        config, offsets = step_test_thunder.make_plan(base_config(), friction)
        expected = collect_thunder.friction_scales(config)
        q0 = np.array(config["start_joint_positions_rad"])
        clock = [0.0]
        cycle = lambda: int(np.floor(clock[0] / 0.002 + 1e-9))
        receive = Mock()
        receive.getActualQ.side_effect = lambda: q0.copy()
        receive.getActualQd.side_effect = lambda: np.zeros(6)
        receive.getTimestamp.side_effect = lambda: (clock.__setitem__(0, clock[0] + 1e-5), 100.0 + cycle() * 0.002)[1]
        control = Mock(); control.setPayload.return_value = True
        seen = []

        def direct_torque(torque, *, viscous_scale, coulomb_scale):
            seen.append((list(viscous_scale), list(coulomb_scale)))
            return True
        control.directTorque.side_effect = direct_torque
        factory = Mock(return_value=control); factory.FLAG_VERBOSE = 1; factory.FLAG_UPLOAD_SCRIPT = 2
        scheduler = {"policy": 0, "priority": 0}
        saved = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receive))}))
            stack.enter_context(patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.5"))
            stack.enter_context(patch.object(collect_thunder.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(collect_thunder.time, "sleep", side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)))
            stack.enter_context(patch.object(collect_thunder.os, "sched_setscheduler",
                                             side_effect=lambda tid, policy, param: scheduler.update(policy=policy, priority=param.sched_priority)))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getscheduler", side_effect=lambda tid: scheduler["policy"]))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getparam",
                                             side_effect=lambda tid: collect_thunder.os.sched_param(scheduler["priority"])))
            stack.enter_context(patch.object(collect_thunder, "thread_schedule_snapshot", return_value={}))
            stack.enter_context(patch.object(collect_thunder, "require_new_fifo_thread", return_value={}))
            stack.enter_context(patch.object(torch, "save", side_effect=lambda record, _: saved.append(copy.deepcopy(record))))
            stack.enter_context(patch("builtins.print"))
            collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt")
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(seen), len(offsets) + 1)                 # every cycle, then zero torque
        self.assertTrue(all(s == (expected["viscous_scale"], expected["coulomb_scale"]) for s in seen))
        record = saved[0]
        self.assertTrue(record["completed"])
        self.assertEqual(record["direct_torque_params"], expected)
        self.assertEqual(record["collection_config"]["step_test"]["friction_mode"], friction)
        return record

    def test_mocked_execution_off_and_ur_default(self):
        off = self.run_mock("off")
        self.assertEqual(off["direct_torque_params"]["coulomb_scale"], [0.0] * 6)
        on = self.run_mock("ur_default")
        self.assertEqual(on["direct_torque_params"]["coulomb_scale"], [0.8, 0.8, 0.7, 0.8, 0.8, 0.8])
        # The fitting contract must keep rejecting records with robot-side compensation.
        from records import validate_record
        with self.assertRaises(ValueError):
            validate_record(on)

    def test_analysis_of_mocked_record(self):
        record = self.run_mock("off")
        import analyze_step_test
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.pt"
            torch.save(record, path)
            with patch.object(sys, "argv", ["analyze_step_test.py", str(path)]), patch("builtins.print"):
                analyze_step_test.main()
            summary = json.loads(path.with_suffix(".steps.json").read_text())
        self.assertEqual(len(summary["steps"]), 4)
        for step in summary["steps"]:                                   # the fake robot never moves
            self.assertAlmostEqual(step["reached"], 0.0, places=6)
            self.assertAlmostEqual(step["stop_short"], step["commanded"], places=3)
            self.assertIsNone(step["onset_s"])


if __name__ == "__main__":
    unittest.main()
