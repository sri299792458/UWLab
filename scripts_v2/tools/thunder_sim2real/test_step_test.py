"""CPU checks for the step-response test: plan validation, robot-side friction scales, full mocked execution, analysis.

No RTDE sockets: the robot interfaces, scheduler and clock are test doubles (same approach as test_contract.py). The fake
robot never moves, so every step should analyze as stop_short == commanded.
"""
from contextlib import ExitStack, nullcontext
import copy
import gc
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
    config["policy_test"].update(axes=["y", "rz"], single_translation_m=[0.01], single_rotation_rad=[0.0698132],
                                 constant_translation_m=[[0.01, 2]], constant_rotation_rad=[[0.0698132, 2]],
                                 policy_steps_per_trial=3, initial_hold_s=0.5, return_hold_s=0.5)
    config.update(changes)
    return config


class StepTestTests(unittest.TestCase):
    def test_policy_audit_uses_current_plan_api_without_any_control_connection(self):
        import audit_step_timing
        factory = Mock(side_effect=RuntimeError("test only: no robot connection"))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "audit.json"
            stack.enter_context(patch.dict(sys.modules, {"rtde_receive": SimpleNamespace(RTDEReceiveInterface=factory)}))
            stack.enter_context(patch.object(audit_step_timing.ct, "check_realtime_permissions"))
            stack.enter_context(patch.object(audit_step_timing.ct, "fifo_thread_creation", side_effect=lambda _: nullcontext()))
            stack.enter_context(patch.object(audit_step_timing.os, "sched_setscheduler"))
            stack.enter_context(patch.object(sys, "argv", ["audit_step_timing.py", "--config",
                                str(HERE / "workstation/collection.step_test.json"), "--mode", "policy",
                                "--verbose-receive", "--repeats", "3", "--output", str(output)]))
            stack.enter_context(patch("builtins.print"))
            with self.assertRaises(SystemExit):
                audit_step_timing.main()
            result = json.loads(output.read_text())
        self.assertEqual(result["requested_samples"], 3 * 99500)
        self.assertEqual(result["sequence_samples"], 99500)
        self.assertEqual(result["repeats"], 3)
        self.assertEqual(result["robot_control_commands_sent"], 0)
        self.assertFalse(result["continuous_500hz"])
        self.assertTrue(result["verbose_receive"])
        self.assertIn("test only: no robot connection", result["failure"])
        factory.assert_called_once()
        self.assertTrue(factory.call_args.kwargs["verbose"])
        self.assertEqual(factory.call_args.kwargs["variables"], ["timestamp", "actual_q", "actual_qd"])

    def test_plan_rejects_large_steps_and_unknown_friction(self):
        with self.assertRaises(ValueError):
            step_test_thunder.make_plan(base_config(translation_steps_m=[0.08]), "off")
        with self.assertRaises(ValueError):
            step_test_thunder.make_plan(base_config(), "on")
        with self.assertRaises(ValueError):
            collect_thunder.friction_scales({"direct_torque_params": {"viscous_scale": [1.2] * 6, "coulomb_scale": [0.5] * 6}})

    def test_gap_tolerance_is_refused_for_sysid_before_robot_connection(self):
        receive_factory, control_factory = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=receive_factory),
                "rtde_control": SimpleNamespace(RTDEControlInterface=control_factory)}), \
                patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.5"):
            with self.assertRaisesRegex(ValueError, "only for step-response comparisons"):
                collect_thunder.collect(base_config(), np.zeros((6, 6)), Path(directory) / "never.pt", allow_small_gaps=True)
        receive_factory.assert_not_called()
        control_factory.assert_not_called()

    def test_plan_schedule_and_guard(self):
        config, offsets, _ = step_test_thunder.make_plan(base_config(), "off")
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

    def run_mock(self, friction, mode="held"):
        config, offsets, anchors = step_test_thunder.make_plan(base_config(), friction, mode)
        expected = collect_thunder.friction_scales(config)
        q0 = np.array(config["start_joint_positions_rad"])
        clock = [0.0]
        cycle = lambda: int(np.floor(clock[0] / 0.002 + 1e-9))
        receive = Mock()
        receive.getActualQ.side_effect = lambda: q0.copy()
        receive.getActualQd.side_effect = lambda: np.zeros(6)
        receive.getTimestamp.side_effect = lambda: (clock.__setitem__(0, clock[0] + 1e-5), 100.0 + cycle() * 0.002)[1]
        control = Mock(); control.setPayload.return_value = True
        control.isProgramRunning.side_effect = [False, False, True] + [True] * 10   # re-uploaded script starts after 2 polls
        seen = []

        def direct_torque(torque, *, viscous_scale, coulomb_scale):
            self.assertFalse(gc.isenabled())
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
            collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt", anchors=anchors)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(seen), len(offsets) + 1)                 # every cycle, then zero torque
        self.assertTrue(all(s == (expected["viscous_scale"], expected["coulomb_scale"]) for s in seen))
        record = saved[0]
        self.assertTrue(record["completed"])
        self.assertEqual(record["direct_torque_params"], expected)
        self.assertEqual(record["collection_config"]["step_test"]["friction_mode"], friction)
        self.assertEqual(record["recording_storage"]["kind"], "preallocated_numpy")
        self.assertTrue(record["recording_storage"]["memory_locked_during_collection"])
        self.assertEqual(record["rt_receive_variables"], ["timestamp", "actual_q", "actual_qd"])
        self.assertTrue(torch.all(record["host_record_end_times_s"] >= record["host_command_return_times_s"]))
        if friction == "off":
            control.setCustomScriptFile.assert_not_called()
            self.assertEqual(record["control_script"], {"source": "ur_rtde compiled-in"})
        else:
            control.setCustomScriptFile.assert_called_once_with(str(collect_thunder.FIXED_CONTROL_SCRIPT))
            self.assertEqual(record["control_script"]["source"], "vendor fixed script")
            names = [c[0] for c in control.method_calls]
            self.assertLess(names.index("setCustomScriptFile"), names.index("setPayload"))   # payload set on the new script
        self.record_offsets, self.record_anchors = offsets, anchors
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

    def test_fixed_control_script_differs_only_in_the_two_scale_lines(self):
        import difflib, hashlib
        vendor = HERE / "workstation/vendor/ur_rtde_1_6_5"
        orig = (vendor / "rtde_control.script").read_text().splitlines()
        fixed = (vendor / "rtde_control_fixed.script").read_text().splitlines()
        self.assertEqual((fixed[0], fixed[-1]), ("def rtde_control():", "end"))      # ur_rtde's wrapping of its built-in script
        changed = [l for l in difflib.unified_diff(orig, fixed[1:-1], lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        self.assertEqual(changed, ["-$5.26         viscous_scale = q_from_input_float_registers(6)",
                                   "-$5.26         couloumb_scale = q_from_input_float_registers(12)",
                                   "+$5.26         viscous_scaling = q_from_input_float_registers(6)",
                                   "+$5.26         coulomb_scaling = q_from_input_float_registers(12)"])
        prov = json.loads((vendor / "PROVENANCE.json").read_text())
        for key in ("original_script", "fixed_script"):
            self.assertEqual(hashlib.sha256((vendor / prov[key]["path"]).read_bytes()).hexdigest(), prov[key]["sha256"])

    def test_policy_plan_anchor_codes_and_cap(self):
        config, offsets, anchors = step_test_thunder.make_plan(base_config(), "off", "policy")
        self.assertEqual(anchors[0], 2)
        for st in config["step_test"]["schedule"]:
            starts = [st["start_sample"] + k * 50 for k in range(st["policy_steps"])]
            self.assertTrue(all(anchors[i] == 1 for i in starts))                      # re-anchor every 0.1 s
            for k, i in enumerate(starts):
                ax = step_test_thunder.AXES.index(st["axis"])
                self.assertAlmostEqual(offsets[i, ax], st["value"] if k < st["repeats"] else 0.0)
            self.assertEqual(anchors[st["return_start_sample"]], 2)                    # back to the initial center
        with self.assertRaises(ValueError):
            bad = base_config(); bad["policy_test"]["constant_translation_m"] = [[0.02, 3]]   # 60 mm > 40 mm cap
            step_test_thunder.make_plan(bad, "off", "policy")

    def test_mocked_policy_execution_targets_follow_measured_pose(self):
        from vendor import ur5e_kinematics as kin
        for friction in ("off", "ur_default"):
            record = self.run_mock(friction, "policy")
            self.assertEqual(record["target_mode"], "anchored")
            self.assertTrue(torch.equal(record["target_anchors"], torch.tensor(self.record_anchors)))
            pos0, _ = kin.get_ee_pose(record["joint_positions"][0].numpy())               # the fake robot never moves
            for i in np.nonzero(self.record_anchors == 1)[0][:6]:
                expected = pos0 + self.record_offsets[i, :3]
                self.assertTrue(np.allclose(record["waypoint_target_pos"][i].numpy(), expected, atol=1e-12))

    def test_analysis_of_mocked_policy_record(self):
        record = self.run_mock("off", "policy")
        import analyze_step_test
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.pt"
            torch.save(record, path)
            with patch.object(sys, "argv", ["analyze_step_test.py", str(path)]), patch("builtins.print"):
                analyze_step_test.main()
            summary = json.loads(path.with_suffix(".steps.json").read_text())
        self.assertEqual(summary["mode"], "policy")
        self.assertEqual(len(summary["trials"]), 8)
        self.assertTrue(all(t["after_trial"] == 0.0 and len(t["step_ends"]) == 3 for t in summary["trials"]))

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

    def test_held_analysis_uses_elapsed_robot_time_including_gap(self):
        import analyze_step_test as analysis
        times = np.arange(80) * 0.002
        times[2:] += 0.004  # Two missing cycles; timestamps remain unmodified.
        q = np.zeros((80, 6))
        q[3:, 0] = 0.001
        q[50:, 0] = 0.002
        schedule = [{"axis": "x", "value": 0.01, "start_sample": 1, "hold_samples": 60,
                     "return_start_sample": 61, "return_samples": 10}]
        with patch.object(analysis, "pose_offsets", side_effect=lambda qs, *_: qs.copy()):
            row = analysis.analyze_held(q, schedule, times, np.zeros(3), np.array([1, 0, 0, 0]))[0]
        self.assertEqual(row["onset_s"], 0.008)
        self.assertEqual(row["moved_0p1s"], 1.0)  # Index 49 is at 0.1 s; index 50 is later.
        self.assertAlmostEqual(row["hold_elapsed_s"], 0.124)
        self.assertEqual(row["gaps_during_hold"], 1)
        self.assertEqual(analysis.timing_review(times)["missing_cycles"], 2)
        self.assertFalse(analysis.timing_review(times)["continuous_500hz"])

    def test_policy_analysis_flags_stretched_interval_and_partial_return(self):
        import analyze_step_test as analysis
        q = np.zeros((12, 6))
        times = np.arange(12) * 0.002
        times[3:] += 0.002
        schedule = [{"kind": "single", "axis": "x", "value": 0.01, "start_sample": 1,
                     "policy_steps": 2, "step_samples": 4, "repeats": 1,
                     "return_start_sample": 9, "return_samples": 4}]
        with patch.object(analysis.kin, "get_ee_pose", return_value=(np.zeros(3), np.array([1, 0, 0, 0]))):
            row = analysis.analyze_policy(q, schedule, times)[0]
        self.assertEqual(row["policy_step_durations_s"], [0.01, 0.008])
        self.assertEqual(row["gaps_during_trial"], 1)
        self.assertFalse(row["return_complete"])


if __name__ == "__main__":
    unittest.main()
