"""Robotiq 2F-85 for the state policy: binary open/close at the deployment settings, and the reported position mapped to
the 6 simulated gripper joints the policy observes.

The real gripper reports one position (0-255). The sim observation holds finger_joint plus 5 linkage joints. Mapping:
  real position -> finger_joint: piecewise linear through measured anchors (open 3 / 60 mm cube ~92 / empty closed 226 at
  speed 128, force 0) and the sim finger_joint at the same states (0 / 0.2884 / 0.8194 rad); an approximation between them.
  finger_joint -> linkage joints: median sim relation (R226; p99 error <= 0.013 rad).
Socket I/O runs in a separate process so it never holds the control loop's interpreter lock.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

GRIPPER_PORT = 63352
OPEN_COMMAND, CLOSE_COMMAND = 0, 255
POLL_HZ = 50


class GripperMap:
    def __init__(self, manifest):
        g = manifest["gripper"]
        a = g["position_to_finger_joint"]
        self.real = np.asarray(a["real_position"], dtype=float)
        self.finger = np.asarray(a["finger_joint_rad"], dtype=float)
        self.linkage = [np.asarray(g["linkage_from_finger_joint"][name], dtype=float)
                        for name in manifest["observation"]["sim_joint_names"][6:]]

    def finger_joint(self, position):
        return float(np.interp(position, self.real, self.finger))   # clamps outside the anchors

    def sim_joints(self, position):
        """6 gripper joint positions in sim order (finger_joint first) for a reported real position."""
        fj = self.finger_joint(position)
        return np.array([fj] + [float(np.interp(fj, t[:, 0], t[:, 1])) for t in self.linkage[1:]])


class GripperProcess:
    """Shared state: desired (-1 none, 0 open, 1 close), position, object status, sample time (monotonic), error."""

    def __init__(self, robot_ip, speed, force, *, allow_motion):
        self.args = (robot_ip, int(speed), int(force), bool(allow_motion))
        self.desired = mp.Value("i", -1, lock=False)
        self.position = mp.Value("d", np.nan, lock=False)
        self.object_status = mp.Value("i", -1, lock=False)
        self.sample_time = mp.Value("d", np.nan, lock=False)
        self.commands_sent = mp.Value("i", 0, lock=False)
        self.error = mp.Array("c", 512, lock=False)
        self.stop_flag = mp.Value("i", 0, lock=False)
        self.process = None

    def start(self, timeout_s=5.0):
        ctx = mp.get_context("spawn")
        self.process = ctx.Process(target=_gripper_main, daemon=True,
                                   args=(*self.args, self.desired, self.position, self.object_status, self.sample_time,
                                         self.commands_sent, self.error, self.stop_flag))
        self.process.start()
        deadline = time.monotonic() + timeout_s
        while not np.isfinite(self.sample_time.value):
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Gripper process produced no position within 5 s")
            time.sleep(0.01)

    def check(self):
        if self.error.value:
            raise RuntimeError("Gripper process: " + self.error.value.decode(errors="replace"))
        if self.process is not None and not self.process.is_alive():
            raise RuntimeError("Gripper process exited")

    def command(self, close):
        self.desired.value = 1 if close else 0

    def latest(self):
        return float(self.position.value), int(self.object_status.value), float(self.sample_time.value)

    def stop(self):
        self.stop_flag.value = 1
        if self.process is not None:
            self.process.join(2.0)
            if self.process.is_alive():
                self.process.terminate()


def _gripper_main(robot_ip, speed, force, allow_motion, desired, position, object_status, sample_time, commands_sent,
                  error, stop_flag):
    try:
        from state_policy.vendor.robotiq_gripper import RobotiqGripper
        gripper = RobotiqGripper()
        gripper.connect(robot_ip, GRIPPER_PORT)
        if not gripper.is_active():
            raise RuntimeError("gripper is not activated; activate it on the pendant first (no activation motion is sent)")
        sent = -1
        period = 1.0 / POLL_HZ
        while not stop_flag.value:
            started = time.monotonic()
            want = desired.value
            if allow_motion and want in (0, 1) and want != sent:
                ok, _ = gripper.move(CLOSE_COMMAND if want == 1 else OPEN_COMMAND, speed, force)
                if not ok:
                    raise RuntimeError("gripper move command was not acknowledged")
                sent = want
                commands_sent.value += 1
            pos = gripper.get_current_position()
            obj = gripper._get_var(gripper.OBJ)
            position.value, object_status.value, sample_time.value = float(pos), int(obj), time.monotonic()
            time.sleep(max(0.0, period - (time.monotonic() - started)))
        gripper.disconnect()
    except BaseException as exc:   # surfaced to the control loop through check()
        error.value = f"{type(exc).__name__}: {exc}".encode()[:511]
