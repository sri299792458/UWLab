"""CPU checks for data provenance, timing, and controller calibration."""
import copy
from contextlib import ExitStack
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from records import HERE, JOINT_NAMES, profile_api, sha256, validate_record


def synthetic_record(n=200):
    times = torch.arange(n, dtype=torch.float64) * 0.002
    q0 = torch.tensor([0.0, -1.57, 1.57, -1.57, -1.57, -1.57], dtype=torch.float64)
    return {
        "schema_version": 1, "robot": "thunder", "source_kind": "synthetic_test",
        "sample_phase": "pre_command", "pose_frame": "base_link", "quaternion_order": "wxyz",
        "completed": True, "failure": None, "cleanup_errors": [], "control_freq": 500, "dt": 0.002,
        "joint_names": JOINT_NAMES, "joint_positions": q0.repeat(n, 1),
        "joint_velocities": torch.zeros(n, 6, dtype=torch.float64),
        "joint_torques": torch.zeros(n, 6, dtype=torch.float64),
        "initial_joint_pos": q0, "initial_joint_vel": torch.zeros(6, dtype=torch.float64),
        "waypoint_step_indices": torch.arange(n), "num_waypoints": n,
        "waypoint_target_pos": torch.zeros(n, 3, dtype=torch.float64),
        "waypoint_target_quat": torch.tensor([1., 0., 0., 0.], dtype=torch.float64).repeat(n, 1),
        "host_sample_times_s": times, "host_command_times_s": times + 0.0005,
        "robot_sample_times_s": times, "robot_after_read_times_s": times.clone(),
        "osc_params": {"motion_stiffness": [1000]*3 + [50]*3,
                       "motion_damping_ratio": [1]*6, "torque_max": [150]*3 + [28]*3},
        "collection_config": {"gripper_configuration": "open_empty"},
        "calibration": json.loads((HERE / "workstation/thunder_calibration.json").read_text()),
        "calibration_sha256": sha256(HERE / "workstation/thunder_calibration.json"),
        "controller_provenance": json.loads((HERE / "workstation/vendor/PROVENANCE.json").read_text()),
        "ur_rtde_version": "1.6.5",
        "direct_torque_params": {"viscous_scale": [0.0]*6, "coulomb_scale": [0.0]*6},
    }


def synthetic_profile():
    return {"schema_version": 1, "robot": "thunder", "source_kind": "synthetic_test",
            "calibration_hash": profile_api().CALIBRATION_HASH,
            "sysid": {"armature": [0.2]*6, "static_friction": [0.5]*6,
                      "dynamic_ratio": [0.8]*6, "viscous_friction": [0.4]*6},
            "controller": synthetic_record()["osc_params"], "fit_dt_s": 0.002,
            "fitted_delay_steps": 2, "fitted_delay_seconds": 0.004,
            "stage2_delay": {"mode": "upstream_range", "physics_hz": 120, "steps": [0, 1]},
            "record_sha256": "synthetic-test-only", "fit_sha256": "synthetic-test-only"}


class ContractTests(unittest.TestCase):
    def test_preallocated_log_keeps_earlier_samples_when_source_arrays_change(self):
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        buffer = collect_thunder.RecordingBuffer(2)
        q, qd, torque, pos, quat = np.arange(6.), np.zeros(6), np.ones(6), np.ones(3), np.array([1., 0., 0., 0.])
        buffer.lock()
        try:
            with patch.object(collect_thunder.time, "monotonic", return_value=2.):
                buffer.append(q, qd, torque, pos, quat, 1., 1.1, 100., 100., 1.2)
            q[:] = 99
            torque[:] = -99
            with patch.object(collect_thunder.time, "monotonic", return_value=3.):
                buffer.append(q, qd, torque, pos, quat, 2., 2.1, 100.002, 100.002, 2.2)
            np.testing.assert_array_equal(buffer.column(0)[0], np.arange(6.))
            np.testing.assert_array_equal(buffer.column(2)[0], np.ones(6))
            np.testing.assert_array_equal(buffer.column(7), [100., 100.002])
            np.testing.assert_array_equal(buffer.column(10), [2., 3.])
            self.assertEqual(len(buffer), 2)
        finally:
            buffer.unlock()
        self.assertFalse(buffer.locked)

    def test_memlock_failure_prevents_all_robot_connections(self):
        import collect_thunder
        receive_factory, control_factory = Mock(), Mock()
        config = json.loads((HERE / "workstation/collection.example.json").read_text())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "test-only.pt"
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=control_factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=receive_factory)}))
            stack.enter_context(patch.object(collect_thunder, "check_realtime_permissions"))
            stack.enter_context(patch.object(collect_thunder.os, "sched_setscheduler"))
            stack.enter_context(patch.object(collect_thunder.RecordingBuffer, "lock", side_effect=RuntimeError("test memlock failure")))
            with self.assertRaisesRegex(RuntimeError, "test memlock failure"):
                collect_thunder.collect(config, np.zeros((6, 6)), output)
            self.assertFalse(output.exists())
        receive_factory.assert_not_called()
        control_factory.assert_not_called()

    def test_synthetic_cannot_be_used_as_robot_data(self):
        with self.assertRaises(ValueError):
            validate_record(synthetic_record())
        self.assertEqual(validate_record(synthetic_record(), allow_synthetic=True)["samples"], 200)

    def test_bad_timing_is_rejected(self):
        for key in ("robot_sample_times_s", "robot_after_read_times_s", "host_command_times_s"):
            record = synthetic_record()
            record[key][10] = -1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(record, allow_synthetic=True)

    def test_rtde165_record_requires_disabled_friction_compensation(self):
        for key in ("viscous_scale", "coulomb_scale"):
            record = synthetic_record()
            record["direct_torque_params"][key][2] = 0.8
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(record, allow_synthetic=True)

    def test_wrong_geometry_phase_or_incomplete_capture_is_rejected(self):
        for key, bad in (("completed", False), ("sample_phase", "post_step"),
                         ("pose_frame", "tool0"), ("calibration_sha256", "wrong"),
                         ("joint_names", list(reversed(JOINT_NAMES)))):
            record = synthetic_record(); record[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(record, allow_synthetic=True)

    def test_nonfinite_joint_sample_is_rejected(self):
        record = synthetic_record(); record["joint_positions"][4, 1] = float("nan")
        with self.assertRaises(ValueError):
            validate_record(record, allow_synthetic=True)

    def test_profile_is_explicit_and_delay_has_units(self):
        api = profile_api(); profile = synthetic_profile()
        with self.assertRaises(ValueError):
            api.validate_profile(profile)
        api.validate_profile(profile, allow_synthetic=True)
        profile["fitted_delay_seconds"] = 2 / 120
        with self.assertRaises(ValueError):
            api.validate_profile(profile, allow_synthetic=True)
        with self.assertRaises(ValueError):
            api.load_profile("")

    def test_controller_is_unmodified_and_thunder_jacobian_matches_finite_difference(self):
        provenance = json.loads((HERE / "workstation/vendor/PROVENANCE.json").read_text())
        self.assertEqual(sha256(HERE / "workstation/vendor/ur5e_kinematics.py"), provenance["sha256"])
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        collect_thunder.install_calibration()
        kin = collect_thunder.kin
        q = np.array([0.1, -1.3, 1.5, -1.1, -1.4, 0.2])
        jac = kin.compute_jacobian_calibrated(q)
        eps = 1e-6
        for j in range(6):
            delta = np.zeros(6); delta[j] = eps
            numeric = (kin.get_ee_pose(q+delta)[0] - kin.get_ee_pose(q-delta)[0]) / (2*eps)
            np.testing.assert_allclose(jac[:3, j], numeric, atol=1e-7, rtol=1e-5)

    def test_motion_template_cannot_execute_with_missing_fields(self):
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        config = json.loads((HERE / "workstation/collection.example.json").read_text())
        with self.assertRaises(ValueError):
            collect_thunder.make_plan(config)

    def run_mock_collection(self, *, wrong_start=False, invalid_velocity=False, fail_command=False,
                            robot_period_s=0.002, midread_every=0, samples=6, fail_rt_interface=None,
                            pause_after_command_s=0., compute_pause_s=0., initial_host_time_s=0.,
                            first_command_pause_s=0., allow_small_gaps=False,
                            repeated_command_pause_s=0., expected_failure=None):
        """No RTDE sockets or on-disk robot records: interfaces, clock and save are mocked.

        A fake host clock drives a mock robot that publishes a new 2 ms-stamped state every robot_period_s host
        seconds (a value other than 0.002 models host/robot clock drift). midread_every > 0 makes every Nth
        getActualQ call land just after a new publish, so the snapshot must be re-read.
        """
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        config = json.loads((HERE / "workstation/collection.simulation_candidate.json").read_text())
        config.update(robot_ip="test-double-only", polyscope_version="test-double-only",
                      payload_mass_kg=1., payload_cog_m=[0., 0., 0.], joint_excursion_limit_rad=[1.]*6)
        if allow_small_gaps:
            config["step_test"] = {"mode": "held"}
        q0 = np.array(config["start_joint_positions_rad"])
        clock = [initial_host_time_s]
        cycle = lambda: int(np.floor(clock[0] / robot_period_s + 1e-9))
        calls = [0]

        def advance(dt):
            clock[0] += dt

        def actual_q():
            calls[0] += 1
            if midread_every and calls[0] % midread_every == 0:
                clock[0] = (cycle() + 1) * robot_period_s      # a new robot state is published mid-read
            return q0 + (0.1 if wrong_start else cycle()*0.001)
        receive = Mock()
        receive.getActualQ.side_effect = actual_q
        receive.getActualQd.side_effect = lambda: np.full(6, np.nan) if invalid_velocity else np.zeros(6)
        receive.getTimestamp.side_effect = lambda: (advance(1e-5), 100. + cycle()*0.002)[1]
        control = Mock()
        control.setPayload.return_value = True
        torque_calls = [0]
        scheduler = {"policy": 0, "priority": 0}
        original_compute = collect_thunder.kin.OperationalSpaceController.compute

        def compute_with_pause(controller, *args, **kwargs):
            result = original_compute(controller, *args, **kwargs)
            if torque_calls[0] == 1:
                advance(compute_pause_s)
            return result

        def set_scheduler(tid, policy, param):
            scheduler.update(policy=policy, priority=param.sched_priority)

        def direct_torque(torque, *, viscous_scale, coulomb_scale):
            # Enforce the actual 1.6.5 call signature, including cleanup. Mock
            # objects alone would accept the removed friction_comp keyword.
            self.assertEqual(len(torque), 6)
            self.assertEqual(viscous_scale, [0.0]*6)
            self.assertEqual(coulomb_scale, [0.0]*6)
            self.assertEqual(scheduler, {"policy": collect_thunder.os.SCHED_FIFO, "priority": 80})
            self.assertFalse(gc.isenabled())
            torque_calls[0] += 1
            if torque_calls[0] == 1:
                advance(first_command_pause_s)
            if torque_calls[0] == 2:
                advance(pause_after_command_s)
            if torque_calls[0] >= 2:
                advance(repeated_command_pause_s)
            return not (fail_command and torque_calls[0] == 2)

        control.directTorque.side_effect = direct_torque
        def construct_control(*args, **kwargs):
            self.assertEqual(scheduler, {"policy": collect_thunder.os.SCHED_FIFO, "priority": 85})
            return control
        def construct_receive(*args, **kwargs):
            self.assertEqual(scheduler, {"policy": collect_thunder.os.SCHED_FIFO, "priority": 90})
            self.assertEqual(kwargs["variables"], ["timestamp", "actual_q", "actual_qd"])
            return receive
        def verify_worker(before, priority, interface):
            if interface == fail_rt_interface:
                raise RuntimeError(f'{interface}: mocked worker stayed at normal priority')
            return {priority: {"policy": collect_thunder.os.SCHED_FIFO, "priority": priority}}
        factory = Mock(side_effect=construct_control)
        factory.FLAG_VERBOSE = 1; factory.FLAG_UPLOAD_SCRIPT = 2
        saved = []
        expected_gc_state = gc.isenabled()

        def save_record(record, _):
            self.assertEqual(gc.isenabled(), expected_gc_state)
            saved.append(record)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(side_effect=construct_receive)),
            }))
            stack.enter_context(patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.5"))
            stack.enter_context(patch.object(collect_thunder.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(collect_thunder.time, "sleep", side_effect=advance))
            stack.enter_context(patch.object(collect_thunder.kin.OperationalSpaceController, "compute",
                                             new=compute_with_pause))
            stack.enter_context(patch.object(collect_thunder.os, "sched_setscheduler", side_effect=set_scheduler))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getscheduler", side_effect=lambda tid: scheduler["policy"]))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getparam", side_effect=lambda tid: collect_thunder.os.sched_param(scheduler["priority"])))
            stack.enter_context(patch.object(collect_thunder, "thread_schedule_snapshot", return_value={}))
            stack.enter_context(patch.object(collect_thunder, "require_new_fifo_thread", side_effect=verify_worker))
            stack.enter_context(patch.object(torch, "save", side_effect=save_record))
            stack.enter_context(patch("builtins.print"))
            offsets = collect_thunder.make_plan(config)[:samples]
            should_fail = (wrong_start or invalid_velocity or fail_command or fail_rt_interface
                           or pause_after_command_s or compute_pause_s or midread_every)
            if expected_failure is not None:
                should_fail = expected_failure
            if should_fail:
                with self.assertRaises(RuntimeError):
                    collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt", allow_small_gaps=allow_small_gaps)
            else:
                collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt", allow_small_gaps=allow_small_gaps)
            self.assertFalse((Path(directory) / "mock-only.pt").exists())
        receive.disconnect.assert_called_once()
        self.assertEqual(scheduler, {"policy": 0, "priority": 0})
        self.assertEqual(gc.isenabled(), expected_gc_state)
        return saved, control, factory

    def test_late_startup_frame_and_slow_first_sdk_call_still_capture_fresh_cycles(self):
        saved, control, _ = self.run_mock_collection(initial_host_time_s=0.0019,
                                                     first_command_pause_s=0.0022)
        record = saved[0]
        self.assertTrue(record["completed"])
        self.assertEqual(validate_record(record)["samples"], 6)
        self.assertAlmostEqual(float(record["robot_sample_times_s"][0]), 100.002)
        self.assertTrue(record["startup"]["first_sample_waited_for_new_frame"])
        self.assertEqual(control.directTorque.call_count, 7)  # warmup sends no torque commands
        self.assertAlmostEqual(float(record["host_command_return_times_s"][0]
                                     - record["host_command_times_s"][0]), 0.0022)

    def test_collector_restores_cyclic_gc_on_success_and_failure(self):
        original = gc.isenabled()
        try:
            for enabled in (True, False):
                for fail_command in (False, True):
                    with self.subTest(gc_enabled=enabled, fail_command=fail_command):
                        if enabled:
                            gc.enable()
                        else:
                            gc.disable()
                        saved, _, _ = self.run_mock_collection(fail_command=fail_command)
                        self.assertEqual(saved[0]["gc_control"],
                                         {"automatic_gc_suspended": True, "enabled_on_entry": enabled})
                        self.assertEqual(gc.isenabled(), enabled)
        finally:
            if original:
                gc.enable()
            else:
                gc.disable()

    def test_collector_locks_to_robot_cycles_under_clock_drift(self):
        for period in (0.0019, 0.002, 0.0021):
            with self.subTest(robot_period_s=period):
                saved, control, factory = self.run_mock_collection(robot_period_s=period, samples=200)
                record = saved[0]
                self.assertEqual(validate_record(record)["samples"], 200)   # strict timestamp contract passes
                self.assertEqual(record["timing_summary"]["duplicate_intervals"], 0)
                self.assertEqual(record["timing_summary"]["intervals_over_2p4ms"], 0)
                self.assertIsNone(record["timing_failure"])
                self.assertEqual(record["timing_mode"], "robot_state_locked")
                self.assertEqual(record["rt_priorities"]["loop"], "SCHED_FIFO 80")
                self.assertEqual(factory.call_args.kwargs["rt_priority"], 85)

    def test_skipped_robot_cycles_abort_and_preserve_partial_record(self):
        for pause in (0.004, 0.038):
            with self.subTest(pause_s=pause):
                saved, control, _ = self.run_mock_collection(pause_after_command_s=pause)
                record = saved[0]
                self.assertFalse(record["completed"])
                self.assertGreater(record["timing_failure"]["interval_s"], 0.0024)
                self.assertIn("last_record_end_s", record["timing_failure"])
                self.assertIn("Fixed-step collection cannot skip robot cycles", record["failure"])
                self.assertEqual(record["num_waypoints"], 2)
                self.assertEqual(record["timing_summary"]["intervals_over_2p4ms"], 0)
                # Two excitation commands, then cleanup; no third excitation after the gap.
                self.assertEqual(control.directTorque.call_count, 3)
                self.assertEqual(control.directTorque.call_args.args[0], [0.0]*6)
                control.servoJ.assert_called_once()
                control.servoStop.assert_called_once()
                control.stopScript.assert_called_once()
                control.disconnect.assert_called_once()

    def test_long_compute_pause_rejects_stale_torque_command(self):
        saved, control, _ = self.run_mock_collection(compute_pause_s=0.038)
        record = saved[0]
        self.assertFalse(record["completed"])
        self.assertIn("more than 20 ms old before torque command", record["failure"])
        self.assertEqual(record["num_waypoints"], 1)
        # The delayed second excitation is never sent; only the first command and cleanup run.
        self.assertEqual(control.directTorque.call_count, 2)
        self.assertEqual(control.directTorque.call_args.args[0], [0.0]*6)
        control.servoJ.assert_called_once()
        control.servoStop.assert_called_once()
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()

    def test_step_comparison_keeps_isolated_gaps_but_fitter_rejects_them(self):
        for pause, missing in ((0.004, 1), (0.006, 2)):
            with self.subTest(pause_s=pause):
                saved, control, _ = self.run_mock_collection(pause_after_command_s=pause,
                                                             allow_small_gaps=True, expected_failure=False)
                record = saved[0]
                self.assertTrue(record["completed"])
                self.assertEqual(record["num_waypoints"], 6)
                self.assertEqual(record["gap_events"][0]["sample_index"], 2)
                self.assertEqual(record["gap_events"][0]["missing_cycles"], missing)
                self.assertEqual(record["timing_summary"]["intervals_over_2p4ms"], 1)
                self.assertEqual(record["state_gap_policy"]["mode"], "step_response_bounded")
                with self.assertRaisesRegex(ValueError, "duplicate/missing samples"):
                    validate_record(record)
                self.assertEqual(control.directTorque.call_count, 7)

    def test_step_comparison_stops_before_command_after_large_gap_or_exhausted_budget(self):
        cases = ({"pause_after_command_s": 0.008},
                 {"repeated_command_pause_s": 0.004, "samples": 12})
        for case in cases:
            with self.subTest(case=case):
                saved, control, _ = self.run_mock_collection(allow_small_gaps=True, expected_failure=True, **case)
                record = saved[0]
                self.assertFalse(record["completed"])
                self.assertTrue("at most 6 ms" in record["failure"] or "budget exceeded" in record["failure"])
                self.assertEqual(control.directTorque.call_count, record["num_waypoints"] + 1)
                self.assertEqual(control.directTorque.call_args.args[0], [0.0] * 6)
                self.assertLessEqual(sum(g["missing_cycles"] for g in record["gap_events"]), 5)

    def test_step_gap_tolerance_keeps_stale_command_guard(self):
        saved, control, _ = self.run_mock_collection(allow_small_gaps=True, compute_pause_s=0.038)
        self.assertFalse(saved[0]["completed"])
        self.assertIn("more than 20 ms old", saved[0]["failure"])
        self.assertEqual(control.directTorque.call_count, 2)

    def test_step_gap_tolerance_rejects_nonintegral_robot_cycles(self):
        import collect_thunder
        receive = Mock()
        receive.getTimestamp.return_value = 100.003
        receive.getActualQ.return_value = np.zeros(6)
        receive.getActualQd.return_value = np.zeros(6)
        with self.assertRaisesRegex(RuntimeError, "whole robot cycles"):
            collect_thunder.read_new_state(receive, 100.0, max_interval_s=0.006)

    def test_rt_permission_failure_prevents_robot_connections(self):
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        config = json.loads((HERE / "workstation/collection.remapped_full_amplitude_swe_candidate.json").read_text())
        receive_factory, control_factory = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=control_factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=receive_factory),
                }), patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.5"), \
                patch.object(collect_thunder.os, "sched_setscheduler", side_effect=[PermissionError("test double"), None]):
            with self.assertRaisesRegex(RuntimeError, "Real-time scheduling required"):
                collect_thunder.collect(config, collect_thunder.make_plan(config)[:6], Path(directory)/'never.pt')
        receive_factory.assert_not_called()
        control_factory.assert_not_called()

    def test_actual_worker_priority_failure_prevents_torque(self):
        for interface in ('RTDEReceiveInterface', 'RTDEControlInterface'):
            with self.subTest(interface=interface):
                saved, control, factory = self.run_mock_collection(fail_rt_interface=interface)
                self.assertFalse(saved)
                control.directTorque.assert_not_called()
                control.servoJ.assert_not_called()
                if interface == 'RTDEReceiveInterface':
                    factory.assert_not_called()
                else:
                    control.stopScript.assert_called_once()
                    control.disconnect.assert_called_once()

    def test_collector_rejects_old_sdk_before_robot_connection(self):
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        config = json.loads((HERE / "workstation/collection.remapped_full_amplitude_swe_candidate.json").read_text())
        control_factory, receive_factory = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=control_factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=receive_factory),
                }), patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.2"):
            with self.assertRaisesRegex(RuntimeError, "1.6.5"):
                collect_thunder.collect(config, collect_thunder.make_plan(config)[:6], Path(directory)/"never.pt")
        control_factory.assert_not_called()
        receive_factory.assert_not_called()

    def test_state_stream_timeout(self):
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        clock = [0.0]
        receive = Mock(); receive.getTimestamp.return_value = 100.0
        def advance(dt):
            clock[0] += dt
        with patch.object(collect_thunder.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(collect_thunder.time, "sleep", side_effect=advance):
            with self.assertRaisesRegex(RuntimeError, "20 ms"):
                collect_thunder.read_new_state(receive, 100.0)
        receive.getActualQ.assert_not_called()

    def test_read_only_audit_can_measure_a_gap_without_relaxing_collection_default(self):
        import collect_thunder
        receive = Mock()
        receive.getTimestamp.return_value = 100.004
        receive.getActualQ.return_value = np.zeros(6)
        receive.getActualQd.return_value = np.zeros(6)
        sample = collect_thunder.read_new_state(receive, 100.0, require_continuity=False)
        self.assertEqual(sample[1], 100.004)
        with self.assertRaisesRegex(RuntimeError, 'Robot state interval 4.000 ms'):
            collect_thunder.read_new_state(receive, 100.0)

    def test_collector_rereads_a_state_updated_mid_read_and_stops_on_gap(self):
        saved, _, _ = self.run_mock_collection(midread_every=50, samples=200)
        record = saved[0]
        self.assertTrue(torch.equal(record["robot_sample_times_s"], record["robot_after_read_times_s"]))  # consistent Q/Qd
        self.assertFalse(record["completed"])
        self.assertIn("Robot state interval 4.000 ms", record["failure"])
        self.assertGreater(record["timing_failure"]["crossed_reads_during_wait"], 0)
        self.assertEqual(record["timing_summary"]["duplicate_intervals"], 0)
        self.assertEqual(record["timing_summary"]["intervals_over_2p4ms"], 0)   # no command after the skipped cycle
        with self.assertRaises(ValueError):
            validate_record(record)                                                # incomplete record remains ineligible

    def test_collector_success_produces_valid_precommand_samples_and_cleans_up(self):
        saved, control, factory = self.run_mock_collection()
        self.assertEqual(len(saved), 1)
        self.assertEqual(validate_record(saved[0])["samples"], 6)
        self.assertEqual(control.directTorque.call_count, 7)  # six samples, then zero torque
        control.servoStop.assert_called_once()
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()

    def test_collector_command_failure_keeps_partial_record_ineligible(self):
        saved, control, _ = self.run_mock_collection(fail_command=True)
        self.assertEqual(len(saved), 1)
        self.assertFalse(saved[0]["completed"])
        with self.assertRaises(ValueError):
            validate_record(saved[0])
        control.servoStop.assert_called_once()
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()

    def test_collector_rejects_wrong_start_or_invalid_velocity_before_control(self):
        for kwargs in ({"wrong_start": True}, {"invalid_velocity": True}):
            with self.subTest(kwargs=kwargs):
                saved, control, factory = self.run_mock_collection(**kwargs)
                self.assertFalse(saved)
                factory.assert_not_called()
                control.directTorque.assert_not_called()

    def test_prepare_finetune_writes_quoted_launch_without_launching(self):
        import prepare_finetune
        import subprocess
        profile = synthetic_profile()
        profile_api().validate_profile(profile, allow_synthetic=True)
        with tempfile.TemporaryDirectory(prefix="thunder launch test ") as directory:
            folder = Path(directory)
            profile_file, checkpoint = folder / "profile.json", folder / "model.pt"
            profile_file.write_text(json.dumps(profile))
            torch.save({"model_state_dict": {"policy.weight": torch.zeros(2, 3)}}, checkpoint)
            args = ["prepare_finetune.py", "--checkpoint", str(checkpoint), "--profile", str(profile_file),
                    "--output", str(folder / "prepared"), "--gpus", "0,1", "--num_envs", "8", "--iterations", "3"]
            api = Mock(); api.load_profile.return_value = (profile, sha256(profile_file))
            with patch.object(sys, "argv", args), patch.object(prepare_finetune, "profile_api", return_value=api), patch("builtins.print"):
                prepare_finetune.main()
            metadata = json.loads((folder / "prepared/launch_metadata.json").read_text())
            self.assertEqual(metadata["status"], "prepared_not_launched")
            self.assertEqual(metadata["total_envs"], 16)
            self.assertIn("--distributed", metadata["command"])
            self.assertIn(str(checkpoint), metadata["command"])
            self.assertEqual(json.loads((folder / "prepared/sysid_profile.json").read_text())["source_kind"], "synthetic_test")
            subprocess.run(["bash", "-n", str(folder / "prepared/launch.sh")], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
