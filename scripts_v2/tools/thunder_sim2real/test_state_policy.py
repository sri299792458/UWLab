"""CPU checks for the R214 state-policy runner (workstation/run_state_policy.py) without a robot, camera or gripper.

Fixture: test_data/r214_contract_fixture.npz, 4 sim episodes (one per reset family, first 40 policy steps) recorded in R214's
native training env (R226): raw poses in base_link, the env's actor observation and the actor output.
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
import torch  # noqa: F401  (import before patch.dict(sys.modules) so a mocked run cannot unload it)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workstation"))
import collect_thunder  # noqa: E402
import run_state_policy as rsp  # noqa: E402
from state_policy import frames as F  # noqa: E402
from state_policy import obs as O  # noqa: E402
from state_policy import gripper as G  # noqa: E402
from vendor import ur5e_kinematics as kin  # noqa: E402

FIX = np.load(HERE / "test_data/r214_contract_fixture.npz")
START = np.array([0.778399, -2.568071, -1.872467, -1.927923, -2.296463, 2.327399])   # configured start (R217 row 5692)
UP_Z_TO_BASE_Y = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # cube +Z -> base_link +y (world up)


def cube_matrix(pos):
    T = np.eye(4); T[:3, :3] = UP_Z_TO_BASE_Y; T[:3, 3] = pos
    return T


class ObservationContractTests(unittest.TestCase):
    def test_rebuilds_recorded_sim_observations(self):
        for e in range(4):
            h = O.ObservationHistory()
            for t in range(FIX["obs"].shape[1]):
                terms = O.frame_terms(FIX["last_action"][e, t], FIX["joint_pos"][e, t], FIX["wrist_pos"][e, t],
                                      FIX["wrist_quat"][e, t], FIX["ins_pos"][e, t], FIX["ins_quat"][e, t],
                                      FIX["rec_pos"][e, t], FIX["rec_quat"][e, t])
                np.testing.assert_allclose(h.push(terms), FIX["obs"][e, t], atol=2e-6)

    def test_exported_policy_reproduces_sim_actions(self):
        m = rsp.load_manifest()
        act = rsp.load_policy(m)
        for e in range(4):
            for t in range(0, 40, 7):
                np.testing.assert_allclose(act(FIX["obs"][e, t]), FIX["action"][e, t], atol=2e-4)

    def test_numpy_policy_equals_torchscript_export(self):
        m = rsp.load_manifest()
        act = rsp.load_policy(m)
        ref_path = Path(m["policy"]["torchscript_reference"]["path"])
        if not ref_path.exists():
            self.skipTest("TorchScript reference lives with the R226 records (not shipped)")
        jit = torch.jit.load(str(ref_path))
        obs = FIX["obs"].reshape(-1, O.OBS_DIM)
        with torch.inference_mode():
            ref = jit(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        np.testing.assert_allclose(np.stack([act(o) for o in obs]), ref, atol=5e-5)

    def test_kinematics_match_sim_wrist_pose(self):
        collect_thunder.install_calibration()
        for e in range(4):
            pos, quat = kin.get_ee_pose(FIX["joint_pos"][e, 0, :6])
            np.testing.assert_allclose(pos, FIX["wrist_pos"][e, 0], atol=1e-5)
            self.assertLess(1 - abs(float(quat @ FIX["wrist_quat"][e, 0])), 1e-6)

    def test_action_to_target_matches_training_rule(self):
        collect_thunder.install_calibration()
        pos, quat = kin.get_ee_pose(START)
        scale = np.array([0.02, 0.02, 0.02, 0.02, 0.2, 0.02])
        a = np.array([1.5, -2.0, 0.5, 3.0, -0.7, 0.2, -1.0])
        tp, tq = rsp.action_to_target(pos, quat, a, scale)
        rot = a[3:6] * scale[3:]
        angle = np.linalg.norm(rot)
        delta = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * rot / angle])   # RelCartesianOSCAction
        np.testing.assert_allclose(tp, pos + a[:3] * scale[:3])
        np.testing.assert_allclose(tq, O.quat_mul(delta, quat), atol=1e-12)


class GripperAndFrameTests(unittest.TestCase):
    def test_gripper_map_anchors_and_linkage(self):
        gm = G.GripperMap(rsp.load_manifest())
        self.assertAlmostEqual(gm.finger_joint(3), 0.0)
        self.assertAlmostEqual(gm.finger_joint(92), 0.2884)
        self.assertAlmostEqual(gm.finger_joint(226), 0.8194)
        self.assertAlmostEqual(gm.finger_joint(255), 0.8194)                      # clamped
        held = FIX["joint_pos"][2, 0, 6:]                                         # Grasped episode: cube in hand
        np.testing.assert_allclose(gm.sim_joints(np.interp(held[0], gm.finger, gm.real)), held, atol=0.02)
        np.testing.assert_allclose(gm.sim_joints(3), np.zeros(6), atol=0.01)

    def test_camera_transform_frames(self):
        T = np.eye(4); T[:3, 3] = [0.3, -0.2, 0.9]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "T.json"
            path.write_text(json.dumps({"T_base_camera": T.tolist()}))
            np.testing.assert_allclose(F.load_camera_transform(path, "base_link"), T)
            converted = F.load_camera_transform(path, "ur_base")                  # UR Base -> base_link: x, y negated
            np.testing.assert_allclose(converted[:3, 3], [-0.3, 0.2, 0.9])
            np.testing.assert_allclose(converted[:3, :3], np.diag([-1.0, -1.0, 1.0]))
            path.write_text(json.dumps({"T_base_camera": (2 * T).tolist()}))
            with self.assertRaises(ValueError):
                F.load_camera_transform(path, "base_link")
            with self.assertRaises(ValueError):
                F.load_camera_transform(path, "base")
        pos, quat = F.matrix_to_pos_quat(cube_matrix([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(F.pos_quat_to_matrix(pos, quat), cube_matrix([0.1, 0.2, 0.3]), atol=1e-12)


class FakeTracker:
    stale_after = None

    def __init__(self, T_base_camera, configs, **kwargs):
        self.calls = 0

    def start(self):
        pass

    def check(self):
        pass

    def stop(self):
        pass

    def latest(self):
        import time
        self.calls += 1
        now = time.monotonic()
        capture = now - 0.04 if FakeTracker.stale_after is None or self.calls < FakeTracker.stale_after else now - 5.0
        base = dict(valid=True, capture_t=capture, reproj_px=0.5, n_tags=2, predicted=False, arrival_t=now, global_time=True)
        rec = F.matrix_to_pos_quat(cube_matrix([-0.45, -0.5935, 0.05]))
        ins = F.matrix_to_pos_quat(cube_matrix([-0.40, -0.5935, -0.10]))
        return {"receptive": dict(base, pos=rec[0], quat=rec[1]), "insertive": dict(base, pos=ins[0], quat=ins[1])}


class FakeGripper:
    instances = []

    def __init__(self, robot_ip, speed, force, *, allow_motion):
        self.allow_motion, self.speed, self.force, self.commands = allow_motion, speed, force, []
        FakeGripper.instances.append(self)

    def start(self):
        pass

    def check(self):
        pass

    def stop(self):
        pass

    def command(self, close):
        self.commands.append(close)

    def latest(self):
        import time
        return 226.0, 3, time.monotonic()


class MockedRunTests(unittest.TestCase):
    def run_mode(self, mode, q0=START, stale_after=None, seconds=1.0):
        cfg = json.loads((HERE / "workstation/collection.state_policy.json").read_text())
        cfg["limits"]["max_episode_s"] = seconds
        FakeTracker.stale_after = stale_after
        FakeGripper.instances = []
        clock = [0.0]
        cycle = lambda: int(np.floor(clock[0] / 0.002 + 1e-9))
        receive = Mock()
        receive.getActualQ.side_effect = lambda: np.array(q0, dtype=float)
        receive.getActualQd.side_effect = lambda: np.zeros(6)
        receive.getTimestamp.side_effect = lambda: (clock.__setitem__(0, clock[0] + 1e-5), 100.0 + cycle() * 0.002)[1]
        control = Mock(); control.setPayload.return_value = True; control.directTorque.return_value = True
        factory = Mock(return_value=control); factory.FLAG_VERBOSE = 1; factory.FLAG_UPLOAD_SCRIPT = 2
        scheduler = {"policy": 0, "priority": 0}
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            cfg_path = Path(d) / "cfg.json"; cfg_path.write_text(json.dumps(cfg))
            cam = Path(d) / "cam.npy"; np.save(cam, np.eye(4))
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_control": SimpleNamespace(RTDEControlInterface=factory),
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receive))}))
            stack.enter_context(patch("importlib.metadata.version", return_value="1.6.5"))
            stack.enter_context(patch.object(collect_thunder.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(collect_thunder.time, "sleep", side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)))
            stack.enter_context(patch.object(collect_thunder.os, "sched_setscheduler",
                                             side_effect=lambda tid, policy, param: scheduler.update(policy=policy, priority=param.sched_priority)))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getscheduler", side_effect=lambda tid: scheduler["policy"]))
            stack.enter_context(patch.object(collect_thunder.os, "sched_getparam",
                                             side_effect=lambda tid: collect_thunder.os.sched_param(scheduler["priority"])))
            stack.enter_context(patch.object(collect_thunder, "thread_schedule_snapshot", return_value={}))
            stack.enter_context(patch.object(collect_thunder, "require_new_fifo_thread", return_value={}))
            stack.enter_context(patch("state_policy.tracker.CubeTracker", FakeTracker))
            stack.enter_context(patch("state_policy.gripper.GripperProcess", FakeGripper))
            stack.enter_context(patch("builtins.input", return_value="GO"))
            stack.enter_context(patch("builtins.print"))
            out = Path(d) / "run"
            rsp.run(SimpleNamespace(mode=mode, config=cfg_path, camera_transform=cam, camera_transform_frame="base_link",
                                    output=out, seconds=1.0))
            summary = json.loads((out / "summary.json").read_text())
            log = dict(np.load(out / "log.npz"))
        return summary, log, control

    def test_dry_run_sends_nothing_and_steps_policy_at_10_hz(self):
        summary, log, control = self.run_mode("dry-run")
        self.assertEqual(summary["stop_reason"], "max_episode_s")
        self.assertTrue(summary["start_pose"]["at_start"])
        self.assertEqual(summary["policy_steps"], 10)
        self.assertTrue(np.array_equal(log["tick_cycle"], np.arange(10) * 50))
        control.directTorque.assert_not_called()
        self.assertEqual(FakeGripper.instances[0].commands, [])
        self.assertFalse(FakeGripper.instances[0].allow_motion)
        # The first observation is the builder's output for the logged state (history filled with the first frame).
        collect_thunder.install_calibration()
        wpos, wquat = kin.get_ee_pose(START)
        terms = O.frame_terms(np.zeros(7), np.concatenate([START, G.GripperMap(rsp.load_manifest()).sim_joints(226)]),
                              wpos, wquat, log["tick_cubes"][0, 1, :3], log["tick_cubes"][0, 1, 3:],
                              log["tick_cubes"][0, 0, :3], log["tick_cubes"][0, 0, 3:])
        np.testing.assert_allclose(log["tick_obs"][0], O.ObservationHistory().push(terms))
        np.testing.assert_allclose(log["tick_obs"][1][O.HISTORY * 6:O.HISTORY * 6 + 7 * 4], 0.0)   # 4 older prev-action slots
        np.testing.assert_allclose(log["tick_obs"][1][O.HISTORY * 6 + 7 * 4:O.HISTORY * 6 + 7 * 5], log["tick_action"][0])

    def test_execute_commands_torque_gripper_and_hands_off(self):
        summary, log, control = self.run_mode("execute")
        self.assertEqual(summary["stop_reason"], "max_episode_s")
        self.assertEqual(control.directTorque.call_count, 500 + 1)                 # every cycle, then zero torque
        control.servoJ.assert_called_once()
        control.servoStop.assert_called_once()
        g = FakeGripper.instances[0]
        self.assertTrue(g.allow_motion)
        self.assertEqual((g.speed, g.force), (128, 0))
        self.assertGreaterEqual(len(g.commands), 1)
        self.assertTrue(np.all(np.isfinite(log["torque"])))
        # The 500 Hz target is held between policy steps and equals the action rule at each step.
        collect_thunder.install_calibration()
        wpos, wquat = kin.get_ee_pose(START)
        scale = np.array(rsp.load_manifest()["training"]["arm_action_scale"])
        tp, tq = rsp.action_to_target(wpos, wquat, log["tick_action"][3], scale)
        np.testing.assert_allclose(log["target_pos"][151:201], np.tile(tp, (50, 1)))   # used from the next cycle

    def test_stale_cube_stops_and_start_outside_training_refused(self):
        summary, _, control = self.run_mode("execute", stale_after=4)
        self.assertIn("SafetyStop", summary["stop_reason"])
        self.assertIn("old", summary["stop_reason"])
        control.servoJ.assert_called_once()                                        # handoff after torque started
        off = START + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.03])                  # 1.7 deg off on wrist 3
        summary, _, control = self.run_mode("execute", q0=off)
        self.assertIn("not at the start pose", summary["stop_reason"])
        control.directTorque.assert_not_called()


class TrackerProcessLogicTests(unittest.TestCase):
    """Runs the tracker's process function in-process with fake pyrealsense2 / aprilcube modules."""

    def test_camera_poses_mapped_to_base_and_last_measurement_kept(self):
        import multiprocessing as mp
        from state_policy import tracker as TR
        T_cam_cube = cube_matrix([0.05, -0.02, 0.60])                       # cube 0.6 m in front of the camera
        T_base_cam = F.pos_quat_to_matrix([0.2, -0.1, 0.4], [0.9238795, 0.3826834, 0.0, 0.0])
        frames_seen = []

        class Det:
            def __init__(self, cfg):
                self.ok = "20-25" not in cfg                                     # the insertive cube is never seen
            def process_frame(self, image, timestamp=None):
                frames_seen.append(timestamp)
                if self.ok and len(frames_seen) <= 4:                           # receptive seen in the first 2 frames only
                    return {"success": True, "T": T_cam_cube, "reproj_error": 0.3, "n_tags": 2, "predicted": False}
                return {"success": False, "T": None, "reproj_error": float("inf"), "n_tags": 0, "predicted": True}

        stop = mp.Value("i", 0, lock=False)

        class Frame:
            def get_frame_timestamp_domain(self):
                return "global"
            def get_timestamp(self):
                return 1.0e6
            def get_data(self):
                return np.zeros((4, 4, 3), np.uint8)
            def __bool__(self):
                return True

        class Pipeline:
            n = 0
            def start(self, config):
                intr = SimpleNamespace(fx=600.0, fy=601.0, ppx=320.0, ppy=240.0, coeffs=[0.1, 0, 0, 0, 0])
                stream = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))
                sensor = SimpleNamespace(supports=lambda o: True, set_option=lambda o, v: None)
                return SimpleNamespace(get_device=lambda: SimpleNamespace(first_color_sensor=lambda: sensor),
                                       get_stream=lambda s: stream)
            def wait_for_frames(self, timeout):
                Pipeline.n += 1
                if Pipeline.n >= 5:
                    stop.value = 1
                return SimpleNamespace(get_color_frame=lambda: Frame())
            def stop(self):
                pass

        rs = SimpleNamespace(pipeline=Pipeline, config=lambda: SimpleNamespace(enable_stream=lambda *a: None,
                                                                            enable_device=lambda s: None),
                             stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8"),
                             option=SimpleNamespace(global_time_enabled="g"),
                             timestamp_domain=SimpleNamespace(global_time="global"))
        made = []
        aprilcube = SimpleNamespace(detector=lambda cfg, cam, dist_coeffs=None, enable_filter=True:
                                    (made.append((cfg, cam, list(dist_coeffs), enable_filter)), Det(cfg))[1])
        data = mp.Array("d", TR.SLOT * 2, lock=False); seq = mp.Value("q", 0, lock=False)
        nframes = mp.Value("q", 0, lock=False); intr = mp.Array("d", 12, lock=False); err = mp.Array("c", 512, lock=False)
        with patch.dict(sys.modules, {"pyrealsense2": rs, "aprilcube": aprilcube}):
            TR._tracker_main(T_base_cam.tolist(), {"receptive": "ids10-15.json", "insertive": "ids20-25.json"}, 640, 480,
                             30, None, True, data, seq, nframes, intr, err, stop)
        self.assertEqual(err.value, b"")
        self.assertEqual(nframes.value, 5)
        self.assertEqual(made[0][1], {"fx": 600.0, "fy": 601.0, "cx": 320.0, "cy": 240.0})
        tracker = TR.CubeTracker.__new__(TR.CubeTracker)
        tracker.data, tracker.seq = data, seq
        out = tracker.latest()
        self.assertTrue(out["receptive"]["valid"])
        self.assertFalse(out["insertive"]["valid"])
        np.testing.assert_allclose(F.pos_quat_to_matrix(out["receptive"]["pos"], out["receptive"]["quat"]),
                                   T_base_cam @ T_cam_cube, atol=1e-12)
        # Kept from frame 2 although frames 3-5 had no measurement: its capture time stays the frame-2 time.
        self.assertEqual(out["receptive"]["capture_t"], frames_seen[2])


if __name__ == "__main__":
    unittest.main()
