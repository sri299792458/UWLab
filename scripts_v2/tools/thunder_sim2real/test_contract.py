"""CPU checks for data provenance, timing, and controller calibration."""
import copy
from contextlib import ExitStack
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
        "ur_rtde_version": "1.6.2",
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

    def run_mock_collection(self, *, wrong_start=False, invalid_velocity=False, fail_command=False):
        """No RTDE sockets or on-disk robot records: interfaces and save are mocked."""
        sys.path.insert(0, str(HERE / "workstation"))
        import collect_thunder
        config = json.loads((HERE / "workstation/collection.simulation_candidate.json").read_text())
        config.update(robot_ip="test-double-only", polyscope_version="test-double-only",
                      payload_mass_kg=1., payload_cog_m=[0., 0., 0.], joint_excursion_limit_rad=[1.]*6)
        q0 = np.array(config["start_joint_positions_rad"])
        tick = [0]
        receive = Mock()
        receive.getActualQ.side_effect = lambda: q0 + (0.1 if wrong_start else tick[0]*0.001)
        receive.getActualQd.side_effect = lambda: np.full(6, np.nan) if invalid_velocity else np.zeros(6)
        receive.getTimestamp.side_effect = lambda: 100. + tick[0]*0.002
        control = Mock()
        control.setPayload.return_value = True
        control.directTorque.side_effect = [True, False, True] if fail_command else None
        control.directTorque.return_value = True
        control.waitPeriod.side_effect = lambda _: tick.__setitem__(0, tick[0]+1)
        factory = Mock(return_value=control)
        factory.FLAG_VERBOSE = 1; factory.FLAG_UPLOAD_SCRIPT = 2
        saved = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receive)),
            }))
            stack.enter_context(patch.object(collect_thunder.importlib.metadata, "version", return_value="1.6.2"))
            stack.enter_context(patch.object(torch, "save", side_effect=lambda record, _: saved.append(record)))
            stack.enter_context(patch("builtins.print"))
            offsets = collect_thunder.make_plan(config)[:6]
            if wrong_start or invalid_velocity or fail_command:
                with self.assertRaises(RuntimeError):
                    collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt")
            else:
                collect_thunder.collect(config, offsets, Path(directory) / "mock-only.pt")
            self.assertFalse((Path(directory) / "mock-only.pt").exists())
        receive.disconnect.assert_called_once()
        return saved, control, factory

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
