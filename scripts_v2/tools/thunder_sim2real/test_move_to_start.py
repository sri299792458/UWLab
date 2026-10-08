"""CPU-only checks for bounded start-pose moves; no RTDE sockets."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workstation"))
import move_to_start as move


class MoveToStartTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "workstation/collection.step_test.json").read_text())
        self.config["robot_ip"] = "OFFLINE-TEST-DOUBLE"
        self.target = np.array(self.config["start_joint_positions_rad"])
        self.q = self.target + np.deg2rad(2.0)
        self.receive = Mock()
        self.receive.isConnected.return_value = True
        self.receive.getSafetyStatusBits.return_value = 1
        self.receive.getRobotMode.return_value = 7
        self.receive.getActualQ.side_effect = lambda: self.q.copy()
        self.receive.getActualQd.return_value = np.zeros(6)
        self.control = Mock()
        self.control.isSteady.return_value = True

        def arrive(*args):
            self.q = self.target.copy()
            return True
        self.control.moveJ.side_effect = arrive
        self.receive_factory = Mock(return_value=self.receive)
        self.control_factory = Mock(return_value=self.control)

    def run_move(self):
        with patch("builtins.print"):
            move.move_to_start(self.config, self.receive_factory, self.control_factory)

    def test_nearby_move_is_slow_async_and_verified(self):
        self.run_move()
        self.control.moveJ.assert_called_once_with(self.target.tolist(), 0.05, 0.1, True)
        self.control.stopJ.assert_not_called()
        self.control.stopScript.assert_called_once()
        self.control.disconnect.assert_called_once()
        self.receive.disconnect.assert_called_once()

    def test_already_at_target_does_not_start_a_control_script(self):
        self.q = self.target.copy()
        self.run_move()
        self.control_factory.assert_not_called()
        self.receive.disconnect.assert_called_once()

    def test_normal_and_reduced_modes_allow_3pe_input_active(self):
        for safety in (1, 2, 2049, 2050):
            with self.subTest(safety_bits=safety):
                self.q = self.target + np.deg2rad(2.0)
                self.control.reset_mock()
                self.receive.getSafetyStatusBits.return_value = safety
                self.run_move()
                self.control.moveJ.assert_called_once_with(self.target.tolist(), 0.05, 0.1, True)

    def test_stop_flags_still_block_motion_with_3pe_input_active(self):
        for bit in range(2, 11):
            self.receive.getSafetyStatusBits.return_value = 2049 | (1 << bit)
            with self.subTest(stop_bit=bit), self.assertRaises(RuntimeError):
                self.run_move()
        self.control_factory.assert_not_called()

    def test_unknown_flags_still_block_motion(self):
        self.receive.getSafetyStatusBits.return_value = 1 | (1 << 12)
        with self.assertRaises(RuntimeError):
            self.run_move()
        self.control_factory.assert_not_called()

    def test_3pe_input_without_ready_mode_still_blocks_motion(self):
        self.receive.getSafetyStatusBits.return_value = 2048
        with self.assertRaises(RuntimeError):
            self.run_move()
        self.control_factory.assert_not_called()

    def test_far_or_different_revolution_is_rejected_before_control(self):
        for delta in (np.deg2rad(5.01), 2*np.pi):
            self.q = self.target.copy()
            self.q[0] += delta
            with self.subTest(delta=delta), self.assertRaisesRegex(RuntimeError, "more than 5 degrees"):
                self.run_move()
        self.control_factory.assert_not_called()

    def test_moving_or_safety_stopped_robot_is_rejected(self):
        for mode in ("moving", "safety_stop", "not_ready"):
            self.receive.getActualQd.return_value = np.ones(6)*0.02 if mode == "moving" else np.zeros(6)
            self.receive.getSafetyStatusBits.return_value = 4 if mode == "safety_stop" else 1
            self.receive.getRobotMode.return_value = 5 if mode == "not_ready" else 7
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                self.run_move()
        self.control_factory.assert_not_called()

    def test_position_is_checked_again_after_connecting_control(self):
        def changed_pose(*args):
            self.q[0] += np.deg2rad(10.0)
            return self.control
        self.control_factory.side_effect = changed_pose
        with self.assertRaisesRegex(RuntimeError, "more than 5 degrees"):
            self.run_move()
        self.control.moveJ.assert_not_called()
        self.control.stopScript.assert_called_once()
        self.receive.disconnect.assert_called_once()

    def test_timeout_stops_motion_and_disconnects(self):
        self.control.moveJ.side_effect = None
        self.control.moveJ.return_value = True
        clock = [0.0]
        def tick():
            clock[0] += 10.0
            return clock[0]
        with patch.object(move.time, "monotonic", side_effect=tick), patch.object(move.time, "sleep"), \
                self.assertRaises(TimeoutError):
            self.run_move()
        self.control.stopJ.assert_called_once_with(0.5)
        self.control.stopScript.assert_called_once()
        self.control.disconnect.assert_called_once()
        self.receive.disconnect.assert_called_once()

    def test_interrupt_stops_motion_and_disconnects(self):
        self.control.moveJ.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_move()
        self.control.stopJ.assert_called_once_with(0.5)
        self.control.stopScript.assert_called_once()
        self.control.disconnect.assert_called_once()
        self.receive.disconnect.assert_called_once()

    def test_invalid_target_is_rejected_before_connections(self):
        self.config["start_joint_positions_rad"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.run_move()
        self.receive_factory.assert_not_called()
        self.control_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
