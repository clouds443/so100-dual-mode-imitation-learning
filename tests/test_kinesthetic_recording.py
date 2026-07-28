import ast
import inspect
import subprocess
import sys
import time
import types
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from lerobot.common.robot_devices import control_utils
from lerobot.common.robot_devices.control_utils import control_loop
from lerobot.common.robot_devices.robots.manipulator import (
    ManipulatorRobot,
    follower_gripper_angle_to_raw_position,
)


class FakeFollowerArm:
    def __init__(self, positions, motor_names=None):
        if motor_names is None:
            motor_names = ["shoulder_pan", "shoulder_lift"]
        self.motor_names = motor_names
        self.motors = {name: (idx + 1, "sts3215") for idx, name in enumerate(self.motor_names)}
        self.positions = np.array(positions, dtype=np.float32)
        self.writes = []
        self.raw_writes = []

    def read(self, data_name):
        if data_name != "Present_Position":
            raise AssertionError(f"Unexpected read: {data_name}")
        return self.positions

    def write(self, data_name, values, motor_names=None):
        self.writes.append((data_name, values, motor_names))

    def write_raw(self, data_name, values, motor_names=None):
        self.raw_writes.append((data_name, values, motor_names))

    def disconnect(self):
        pass


class FakeCamera:
    def __init__(self):
        self.logs = {"delta_timestamp_s": 0.001}

    def async_read(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def disconnect(self):
        pass


class KinestheticRecordingTest(unittest.TestCase):
    def test_gripper_open_key_press_and_release_updates_pressed_state(self):
        class FakeKey:
            space = object()
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        self.assertTrue(hasattr(control_utils, "handle_keyboard_press"))
        self.assertTrue(hasattr(control_utils, "handle_keyboard_release"))
        control_utils.handle_keyboard_press(type("KeyCode", (), {"char": "O"})(), events, FakeKeyboard)

        self.assertTrue(events["gripper_open_pressed"])
        self.assertTrue(events["gripper_open_requested"])
        self.assertFalse(events["gripper_close_pressed"])

        control_utils.handle_keyboard_release(type("KeyCode", (), {"char": "o"})(), events, FakeKeyboard)

        self.assertFalse(events["gripper_open_pressed"])
        self.assertTrue(events["gripper_open_requested"])

    def test_gripper_close_key_press_and_release_updates_pressed_state(self):
        class FakeKey:
            space = object()
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        control_utils.handle_keyboard_press(type("KeyCode", (), {"char": "c"})(), events, FakeKeyboard)

        self.assertTrue(events["gripper_close_pressed"])
        self.assertTrue(events["gripper_close_requested"])
        self.assertFalse(events["gripper_open_pressed"])

        control_utils.handle_keyboard_release(type("KeyCode", (), {"char": "C"})(), events, FakeKeyboard)

        self.assertFalse(events["gripper_close_pressed"])
        self.assertTrue(events["gripper_close_requested"])

    def test_gripper_open_key_invokes_immediate_callback_when_provided(self):
        class FakeKey:
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
            "gripper_requested_direction": None,
        }
        calls = []

        control_utils.handle_keyboard_press(
            type("KeyCode", (), {"char": "o"})(),
            events,
            FakeKeyboard,
            on_gripper_open=lambda: calls.append("open"),
        )

        self.assertEqual(calls, ["open"])
        self.assertFalse(events["gripper_open_requested"])
        self.assertIsNone(events["gripper_requested_direction"])

    def test_gripper_close_key_invokes_immediate_callback_when_provided(self):
        class FakeKey:
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
            "gripper_requested_direction": None,
        }
        calls = []

        control_utils.handle_keyboard_press(
            type("KeyCode", (), {"char": "c"})(),
            events,
            FakeKeyboard,
            on_gripper_close=lambda: calls.append("close"),
        )

        self.assertEqual(calls, ["close"])
        self.assertFalse(events["gripper_close_requested"])
        self.assertIsNone(events["gripper_requested_direction"])

    def test_gripper_keys_accept_windows_virtual_key_codes(self):
        class FakeKey:
            space = object()
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        control_utils.handle_keyboard_press(type("KeyCode", (), {"vk": ord("O")})(), events, FakeKeyboard)
        control_utils.handle_keyboard_press(type("KeyCode", (), {"vk": ord("C")})(), events, FakeKeyboard)

        self.assertTrue(events["gripper_open_pressed"])
        self.assertTrue(events["gripper_close_pressed"])
        self.assertFalse(events["gripper_open_requested"])
        self.assertTrue(events["gripper_close_requested"])
        self.assertEqual(events["gripper_requested_direction"], "close")

        control_utils.handle_keyboard_release(type("KeyCode", (), {"vk": ord("O")})(), events, FakeKeyboard)
        control_utils.handle_keyboard_release(type("KeyCode", (), {"vk": ord("C")})(), events, FakeKeyboard)

        self.assertFalse(events["gripper_open_pressed"])
        self.assertFalse(events["gripper_close_pressed"])

    def test_space_key_does_not_control_gripper_anymore(self):
        class FakeKey:
            space = object()
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        control_utils.handle_keyboard_press(FakeKey.space, events, FakeKeyboard)

        self.assertFalse(events["gripper_open_pressed"])
        self.assertFalse(events["gripper_close_pressed"])
        self.assertFalse(events["gripper_open_requested"])
        self.assertFalse(events["gripper_close_requested"])

    def test_polled_keyboard_char_requests_gripper_step(self):
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        self.assertTrue(hasattr(control_utils, "handle_polled_keyboard_char"))
        control_utils.handle_polled_keyboard_char("o", events, gripper_open_key="o", gripper_close_key="c")
        control_utils.handle_polled_keyboard_char("C", events, gripper_open_key="o", gripper_close_key="c")

        self.assertFalse(events["gripper_open_requested"])
        self.assertTrue(events["gripper_close_requested"])
        self.assertEqual(events["gripper_requested_direction"], "close")

    def test_cv2_key_code_requests_gripper_step(self):
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
        }

        self.assertTrue(hasattr(control_utils, "handle_cv2_key_code"))
        control_utils.handle_cv2_key_code(ord("o"), events, gripper_open_key="o", gripper_close_key="c")

        self.assertTrue(events["gripper_open_requested"])
        self.assertFalse(events["gripper_close_requested"])

    def test_global_keyboard_state_invokes_immediate_gripper_callback(self):
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
            "gripper_requested_direction": None,
        }
        calls = []

        self.assertTrue(hasattr(control_utils, "poll_global_keyboard_state"))
        control_utils.poll_global_keyboard_state(
            events,
            gripper_open_key="o",
            gripper_close_key="c",
            on_gripper_open=lambda: calls.append("open"),
            on_gripper_close=lambda: calls.append("close"),
            key_state_reader=lambda key: key == "o",
        )

        self.assertEqual(calls, ["open"])
        self.assertTrue(events["gripper_open_pressed"])
        self.assertFalse(events["gripper_close_pressed"])
        self.assertFalse(events["gripper_open_requested"])

    def test_global_keyboard_poll_does_not_clear_pynput_held_state(self):
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
            "gripper_requested_direction": None,
        }

        class FakeKey:
            right = object()
            left = object()
            esc = object()

        class FakeKeyboard:
            Key = FakeKey

        control_utils.handle_keyboard_press(type("KeyCode", (), {"char": "c"})(), events, FakeKeyboard)
        control_utils.poll_global_keyboard_state(events, key_state_reader=lambda key: False)

        self.assertTrue(events["gripper_close_pressed"])

    def test_windows_hotkey_id_invokes_gripper_callback(self):
        calls = []

        self.assertTrue(hasattr(control_utils, "handle_windows_hotkey_id"))
        control_utils.handle_windows_hotkey_id(
            1,
            on_gripper_open=lambda: calls.append("open"),
            on_gripper_close=lambda: calls.append("close"),
        )
        control_utils.handle_windows_hotkey_id(
            2,
            on_gripper_open=lambda: calls.append("open"),
            on_gripper_close=lambda: calls.append("close"),
        )

        self.assertEqual(calls, ["close", "open"])

    def test_gripper_command_text_invokes_callback(self):
        calls = []

        self.assertTrue(hasattr(control_utils, "handle_gripper_command_text"))
        self.assertTrue(
            control_utils.handle_gripper_command_text(
                "open",
                on_gripper_open=lambda: calls.append("open"),
                on_gripper_close=lambda: calls.append("close"),
            )
        )
        self.assertTrue(
            control_utils.handle_gripper_command_text(
                "close",
                on_gripper_open=lambda: calls.append("open"),
                on_gripper_close=lambda: calls.append("close"),
            )
        )
        self.assertFalse(
            control_utils.handle_gripper_command_text(
                "noop",
                on_gripper_open=lambda: calls.append("open"),
                on_gripper_close=lambda: calls.append("close"),
            )
        )

        self.assertEqual(calls, ["open", "close"])

    def test_gripper_command_text_ignores_utf8_bom_from_powershell(self):
        calls = []

        self.assertTrue(
            control_utils.handle_gripper_command_text(
                "\ufeffclose",
                on_gripper_open=lambda: calls.append("open"),
                on_gripper_close=lambda: calls.append("close"),
            )
        )

        self.assertEqual(calls, ["close"])

    def test_legacy_gripper_position_limits_map_to_raw_angle_domain(self):
        open_angle, closed_angle = control_utils.resolve_gripper_angle_limits(
            gripper_open_position=180,
            gripper_closed_position=0,
        )

        self.assertEqual((open_angle, closed_angle), (90, -90))

    def test_legacy_gripper_position_limits_infer_0_to_180_range_from_pair(self):
        open_angle, closed_angle = control_utils.resolve_gripper_angle_limits(
            gripper_open_position=180,
            gripper_closed_position=90,
        )

        self.assertEqual((open_angle, closed_angle), (90, 0))

    def test_legacy_gripper_position_limits_accept_explicit_range_for_ambiguous_values(self):
        open_angle, closed_angle = control_utils.resolve_gripper_angle_limits(
            gripper_open_position=90,
            gripper_closed_position=0,
            gripper_position_range=180,
        )

        self.assertEqual((open_angle, closed_angle), (0, -90))

    def test_legacy_gripper_position_limits_accept_old_0_to_100_range(self):
        open_angle, closed_angle = control_utils.resolve_gripper_angle_limits(
            gripper_open_position=100,
            gripper_closed_position=0,
        )

        self.assertEqual((open_angle, closed_angle), (90, -90))

    def test_explicit_gripper_angles_override_legacy_position_limits(self):
        open_angle, closed_angle = control_utils.resolve_gripper_angle_limits(
            gripper_open_angle=80,
            gripper_closed_angle=-70,
            gripper_open_position=180,
            gripper_closed_position=0,
        )

        self.assertEqual((open_angle, closed_angle), (80, -70))

    def test_file_command_listener_consumes_command_file(self):
        calls = []

        with TemporaryDirectory() as tmp_dir:
            command_file = Path(tmp_dir) / "gripper_command.txt"
            listener = control_utils.FileCommandListener(
                command_file,
                on_gripper_open=lambda: calls.append("open"),
                on_gripper_close=lambda: calls.append("close"),
                poll_interval_s=0.01,
            )
            listener.start()
            try:
                deadline = time.perf_counter() + 1
                command_file.write_text("close", encoding="utf-8")
                while time.perf_counter() < deadline and calls != ["close"]:
                    time.sleep(0.01)
            finally:
                listener.stop()

        self.assertEqual(calls, ["close"])

    def test_gripper_http_listener_invokes_callbacks(self):
        calls = []

        self.assertTrue(hasattr(control_utils, "GripperHttpListener"))
        listener = control_utils.GripperHttpListener(
            port=0,
            on_gripper_open=lambda: calls.append("open"),
            on_gripper_close=lambda: calls.append("close"),
        )
        listener.start()
        try:
            with urllib.request.urlopen(f"{listener.url}/close", timeout=2) as response:
                self.assertEqual(response.status, 200)
            with urllib.request.urlopen(f"{listener.url}/open", timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            listener.stop()

        self.assertEqual(calls, ["close", "open"])

    def test_set_follower_torque_writes_expected_torque_value(self):
        arm = FakeFollowerArm([1, 2])
        robot = ManipulatorRobot(robot_type="so100", follower_arms={"main": arm})

        robot.set_follower_torque(False)
        robot.set_follower_torque(True)

        self.assertEqual(arm.writes, [("Torque_Enable", 0, None), ("Torque_Enable", 1, None)])

    def test_set_follower_torque_can_target_specific_motors(self):
        arm = FakeFollowerArm(
            [1, 2, 3],
            motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        )
        robot = ManipulatorRobot(robot_type="so100", follower_arms={"main": arm})

        self.assertIn("motor_names", inspect.signature(robot.set_follower_torque).parameters)
        robot.set_follower_torque(False, motor_names=["shoulder_pan", "shoulder_lift"])
        robot.set_follower_torque(True, motor_names="gripper")

        self.assertEqual(
            arm.writes,
            [
                ("Torque_Enable", 0, ["shoulder_pan", "shoulder_lift"]),
                ("Torque_Enable", 1, "gripper"),
            ],
        )

    def test_set_follower_gripper_position_only_writes_gripper_goal(self):
        arm = FakeFollowerArm(
            [1, 2, 3],
            motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        )
        robot = ManipulatorRobot(robot_type="so100", follower_arms={"main": arm})

        self.assertTrue(hasattr(robot, "set_follower_gripper_position"))
        robot.set_follower_gripper_position(100)

        self.assertEqual(arm.writes, [("Goal_Position", 100, "gripper")])

    def test_follower_gripper_angle_to_raw_position_matches_reference_scripts(self):
        self.assertEqual(follower_gripper_angle_to_raw_position(-90), 1024)
        self.assertEqual(follower_gripper_angle_to_raw_position(0), 2048)
        self.assertEqual(follower_gripper_angle_to_raw_position(90), 3072)
        self.assertEqual(follower_gripper_angle_to_raw_position(-180), 1024)
        self.assertEqual(follower_gripper_angle_to_raw_position(180), 3072)

    def test_feetech_write_raw_skips_calibration_conversion(self):
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        added_params = []

        class FakeGroupSyncWrite:
            def __init__(self, port_handler, packet_handler, addr, num_bytes):
                self.addr = addr
                self.num_bytes = num_bytes

            def addParam(self, motor_id, data):
                added_params.append((motor_id, self.addr, self.num_bytes, data))

            def changeParam(self, motor_id, data):
                added_params.append((motor_id, self.addr, self.num_bytes, data))

            def txPacket(self):
                return 0

        fake_scs = types.SimpleNamespace(COMM_SUCCESS=0, GroupSyncWrite=FakeGroupSyncWrite)
        bus = FeetechMotorsBus("COM_FAKE", {"gripper": (6, "sts3215")}, mock=True)
        bus.is_connected = True
        bus.port_handler = types.SimpleNamespace(port_name="COM_FAKE", closePort=lambda: None)
        bus.packet_handler = types.SimpleNamespace(getTxRxResult=lambda comm: f"comm={comm}")
        bus.set_calibration(
            {
                "motor_names": ["gripper"],
                "calib_mode": ["DEGREE"],
                "drive_mode": [0],
                "homing_offset": [0],
            }
        )

        with (
            mock.patch.dict(sys.modules, {"tests.mock_scservo_sdk": fake_scs}),
            mock.patch.object(bus, "revert_calibration", side_effect=AssertionError("calibration used")),
        ):
            bus.write_raw("Goal_Position", 3072, "gripper")

        self.assertEqual(added_params, [(6, 42, 2, 3072)])

    def test_feetech_write_raw_initializes_writer_even_after_goal_position_read(self):
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, get_group_sync_key

        added_params = []

        class FakeGroupSyncWrite:
            def __init__(self, port_handler, packet_handler, addr, num_bytes):
                self.addr = addr
                self.num_bytes = num_bytes

            def addParam(self, motor_id, data):
                added_params.append(("add", motor_id, self.addr, self.num_bytes, data))

            def changeParam(self, motor_id, data):
                added_params.append(("change", motor_id, self.addr, self.num_bytes, data))

            def txPacket(self):
                return 0

        fake_scs = types.SimpleNamespace(COMM_SUCCESS=0, GroupSyncWrite=FakeGroupSyncWrite)
        bus = FeetechMotorsBus("COM_FAKE", {"gripper": (6, "sts3215")}, mock=True)
        bus.is_connected = True
        bus.port_handler = types.SimpleNamespace(port_name="COM_FAKE", closePort=lambda: None)
        bus.packet_handler = types.SimpleNamespace(getTxRxResult=lambda comm: f"comm={comm}")
        bus.group_readers[get_group_sync_key("Goal_Position", ["gripper"])] = object()

        with mock.patch.dict(sys.modules, {"tests.mock_scservo_sdk": fake_scs}):
            bus.write_raw("Goal_Position", 3072, "gripper")

        self.assertEqual(added_params, [("add", 6, 42, 2, 3072)])

    def test_set_follower_gripper_angle_writes_raw_reference_position(self):
        arm = FakeFollowerArm(
            [1, 2, 3],
            motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        )
        robot = ManipulatorRobot(robot_type="so100", follower_arms={"main": arm})

        robot.set_follower_gripper_angle(90)

        self.assertEqual(arm.raw_writes, [("Goal_Position", 3072, "gripper")])
        self.assertEqual(arm.writes, [])

    def test_step_follower_gripper_moves_incrementally_in_raw_angle_domain(self):
        arm = FakeFollowerArm(
            [1, 2, 3],
            motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        )
        robot = ManipulatorRobot(robot_type="so100", follower_arms={"main": arm})

        self.assertTrue(hasattr(robot, "step_follower_gripper_towards"))
        robot.set_follower_gripper_angle(85)
        robot.step_follower_gripper_towards(target_position=90, step_size=10, min_position=-90, max_position=90)
        robot.step_follower_gripper_towards(target_position=-90, step_size=30, min_position=-90, max_position=90)

        self.assertEqual(
            arm.raw_writes,
            [
                ("Goal_Position", follower_gripper_angle_to_raw_position(85), "gripper"),
                ("Goal_Position", follower_gripper_angle_to_raw_position(90), "gripper"),
                ("Goal_Position", follower_gripper_angle_to_raw_position(60), "gripper"),
            ],
        )

    def test_kinesthetic_step_records_follower_position_as_state_and_action(self):
        arm = FakeFollowerArm([10, 20])
        camera = FakeCamera()
        robot = ManipulatorRobot(
            robot_type="so100",
            follower_arms={"main": arm},
            cameras={"laptop": camera},
        )
        robot.is_connected = True

        observation, action = robot.kinesthetic_step(record_data=True)

        torch.testing.assert_close(observation["observation.state"], torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(action["action"], torch.tensor([10.0, 20.0]))
        self.assertEqual(observation["observation.images.laptop"].shape, torch.Size([4, 5, 3]))
        self.assertEqual(arm.writes, [])

    def test_control_loop_uses_kinesthetic_step_for_kinesthetic_recording(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.kinesthetic_calls = 0

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def teleop_step(self, record_data=False):
                raise AssertionError("teleop_step should not be used in kinesthetic mode")

            def kinesthetic_step(self, record_data=False):
                self.kinesthetic_calls += 1
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}

        control_loop(
            robot=robot,
            control_time_s=0.000001,
            teleoperate=True,
            record_control_mode="kinesthetic",
            dataset=dataset,
            events={"exit_early": False},
            fps=30,
        )

        self.assertEqual(robot.kinesthetic_calls, 1)
        torch.testing.assert_close(dataset["current_episode"]["action"][0], torch.tensor([1.0]))

    def test_control_loop_steps_gripper_open_while_open_key_is_held(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {"exit_early": False, "gripper_open_pressed": True, "gripper_close_pressed": False}

        try:
            control_loop(
                robot=robot,
                control_time_s=0.000001,
                teleoperate=True,
                record_control_mode="kinesthetic",
                dataset=dataset,
                events=events,
                fps=30,
                kinesthetic_gripper_keyboard=True,
                gripper_open_angle=90,
                gripper_closed_angle=-90,
                gripper_speed=30,
            )
        except TypeError as exc:
            self.fail(f"control_loop should accept keyboard gripper options: {exc}")

        self.assertEqual(robot.step_calls, [(90, 1, -90, 90)])

    def test_control_loop_steps_gripper_once_for_quick_open_tap(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": True,
            "gripper_close_requested": False,
        }

        control_loop(
            robot=robot,
            control_time_s=0.000001,
            teleoperate=True,
            record_control_mode="kinesthetic",
            dataset=dataset,
            events=events,
            fps=30,
            kinesthetic_gripper_keyboard=True,
            gripper_open_angle=90,
            gripper_closed_angle=-90,
            gripper_speed=30,
            gripper_tap_step=10,
        )

        self.assertEqual(robot.step_calls, [(90, 10, -90, 90)])
        self.assertFalse(events["gripper_open_requested"])

    def test_control_loop_reports_gripper_raw_state_after_step(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                return 45

            def read_follower_gripper_raw_state(self):
                return 2560, 2500

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": True,
            "gripper_close_requested": False,
        }
        output = StringIO()

        with redirect_stdout(output):
            control_loop(
                robot=robot,
                control_time_s=0.000001,
                teleoperate=True,
                record_control_mode="kinesthetic",
                dataset=dataset,
                events=events,
                fps=30,
                kinesthetic_gripper_keyboard=True,
                gripper_open_angle=90,
                gripper_closed_angle=-90,
                gripper_speed=30,
                gripper_tap_step=10,
                gripper_report_interval_s=0,
            )

        self.assertIn(
            "GRIPPER_OPEN next_angle=45.0 target_raw=3072 goal_raw=2560 present_raw=2500",
            output.getvalue(),
        )

    def test_control_loop_steps_gripper_closed_while_close_key_is_held(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {"exit_early": False, "gripper_open_pressed": False, "gripper_close_pressed": True}

        control_loop(
            robot=robot,
            control_time_s=0.000001,
            teleoperate=True,
            record_control_mode="kinesthetic",
            dataset=dataset,
            events=events,
            fps=30,
            kinesthetic_gripper_keyboard=True,
            gripper_open_angle=90,
            gripper_closed_angle=-90,
            gripper_speed=30,
        )

        self.assertEqual(robot.step_calls, [(-90, 1, -90, 90)])

    def test_control_loop_steps_gripper_once_for_quick_close_tap(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": True,
        }

        control_loop(
            robot=robot,
            control_time_s=0.000001,
            teleoperate=True,
            record_control_mode="kinesthetic",
            dataset=dataset,
            events=events,
            fps=30,
            kinesthetic_gripper_keyboard=True,
            gripper_open_angle=90,
            gripper_closed_angle=-90,
            gripper_speed=30,
            gripper_tap_step=10,
        )

        self.assertEqual(robot.step_calls, [(-90, 10, -90, 90)])
        self.assertFalse(events["gripper_close_requested"])

    def test_control_loop_does_not_move_gripper_when_open_and_close_are_both_held(self):
        class FakeRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.leader_arms = {}
                self.follower_arms = {"main": object()}
                self.cameras = {}
                self.logs = {}
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

        robot = FakeRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        events = {"exit_early": False, "gripper_open_pressed": True, "gripper_close_pressed": True}

        control_loop(
            robot=robot,
            control_time_s=0.000001,
            teleoperate=True,
            record_control_mode="kinesthetic",
            dataset=dataset,
            events=events,
            fps=30,
            kinesthetic_gripper_keyboard=True,
            gripper_open_angle=90,
            gripper_closed_angle=-90,
            gripper_speed=30,
        )

        self.assertEqual(robot.step_calls, [])

    def test_record_with_keyboard_gripper_keeps_only_gripper_torque_enabled(self):
        from lerobot.scripts import control_robot as control_robot_script

        class FakeRecordRobot:
            def __init__(self):
                self.is_connected = True
                self.robot_type = "so100"
                self.has_camera = False
                self.num_cameras = 0
                self.follower_arms = {
                    "main": FakeFollowerArm(
                        [1, 2, 3],
                        motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
                    )
                }
                self.torque_calls = []
                self.gripper_angle_calls = []
                self.step_calls = []

            def connect(self):
                raise AssertionError("Robot should already be connected")

            def disconnect(self):
                self.is_connected = False

            def set_follower_torque(self, enabled, motor_names=None):
                self.torque_calls.append((enabled, motor_names))

            def set_follower_gripper_angle(self, angle):
                self.gripper_angle_calls.append(angle)

            def step_follower_gripper_towards(self, target_position, step_size, min_position, max_position):
                self.step_calls.append((target_position, step_size, min_position, max_position))

            def kinesthetic_step(self, record_data=False):
                return {"observation.state": torch.tensor([1.0])}, {"action": torch.tensor([1.0])}

        robot = FakeRecordRobot()
        dataset = {"fps": 30, "num_episodes": 0, "video": False, "videos_dir": Path(".")}
        listener_kwargs = {}
        global_listener_instances = []
        file_listener_instances = []
        http_listener_instances = []

        def fake_init_keyboard_listener(**kwargs):
            listener_kwargs.update(kwargs)
            events = {
                "exit_early": False,
                "gripper_open_pressed": False,
                "gripper_close_pressed": False,
                "gripper_open_requested": False,
                "gripper_close_requested": False,
                "gripper_requested_direction": None,
            }
            listener_kwargs["events"] = events
            return (
                None,
                events,
            )

        class FakeGlobalKeyboardPollListener:
            def __init__(self, events, **kwargs):
                self.events = events
                self.kwargs = kwargs
                self.started = False
                self.stopped = False
                global_listener_instances.append(self)

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        class FakeFileCommandListener(FakeGlobalKeyboardPollListener):
            def __init__(self, command_file, **kwargs):
                self.command_file = command_file
                self.kwargs = kwargs
                self.started = False
                self.stopped = False
                file_listener_instances.append(self)

        class FakeGripperHttpListener(FakeGlobalKeyboardPollListener):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = False
                self.stopped = False
                self.url = "http://127.0.0.1:8765/"
                http_listener_instances.append(self)

        with (
            mock.patch.object(control_robot_script, "init_dataset", return_value=dataset),
            mock.patch.object(control_robot_script, "init_keyboard_listener", side_effect=fake_init_keyboard_listener),
            mock.patch.object(
                control_robot_script,
                "GlobalKeyboardPollListener",
                side_effect=FakeGlobalKeyboardPollListener,
            ),
            mock.patch.object(
                control_robot_script,
                "FileCommandListener",
                side_effect=FakeFileCommandListener,
            ),
            mock.patch.object(
                control_robot_script,
                "GripperHttpListener",
                side_effect=FakeGripperHttpListener,
            ),
            mock.patch.object(control_robot_script, "warmup_record"),
            mock.patch.object(control_robot_script, "create_lerobot_dataset", return_value=object()),
        ):
            try:
                control_robot_script.record(
                    robot=robot,
                    root="data",
                    repo_id="test/so100_student",
                    record_control_mode="kinesthetic",
                    kinesthetic_gripper_keyboard=1,
                    gripper_open_angle=90,
                    gripper_closed_angle=-90,
                    gripper_speed=30,
                    gripper_tap_step=20,
                    gripper_http_port=8765,
                    fps=30,
                    num_episodes=0,
                    push_to_hub=False,
                    run_compute_stats=False,
                    play_sounds=False,
                )
            except TypeError as exc:
                self.fail(f"record should accept keyboard gripper options: {exc}")

        self.assertEqual(
            robot.torque_calls,
            [
                (False, ["shoulder_pan", "shoulder_lift"]),
                (True, "gripper"),
            ],
        )
        self.assertEqual(robot.gripper_angle_calls, [90])
        self.assertNotIn("on_gripper_open", listener_kwargs)
        self.assertNotIn("on_gripper_close", listener_kwargs)
        self.assertEqual(robot.step_calls, [])
        self.assertEqual(len(global_listener_instances), 1)
        self.assertTrue(global_listener_instances[0].started)
        self.assertEqual(global_listener_instances[0].kwargs["gripper_open_key"], "o")
        self.assertEqual(global_listener_instances[0].kwargs["gripper_close_key"], "c")
        self.assertNotIn("on_gripper_open", global_listener_instances[0].kwargs)
        self.assertNotIn("on_gripper_close", global_listener_instances[0].kwargs)
        self.assertTrue(global_listener_instances[0].stopped)
        self.assertEqual(len(file_listener_instances), 1)
        self.assertTrue(file_listener_instances[0].started)
        self.assertTrue(file_listener_instances[0].stopped)
        self.assertEqual(str(file_listener_instances[0].command_file), ".cache/gripper_command.txt")
        file_listener_instances[0].kwargs["on_gripper_close"]()
        self.assertEqual(robot.step_calls, [])
        self.assertTrue(listener_kwargs["events"]["gripper_close_requested"])
        self.assertEqual(len(http_listener_instances), 1)
        self.assertTrue(http_listener_instances[0].started)
        self.assertTrue(http_listener_instances[0].stopped)
        self.assertEqual(http_listener_instances[0].kwargs["port"], 8765)

    def test_record_help_lists_keyboard_gripper_options(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "lerobot/scripts/control_robot.py", "record", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kinesthetic-gripper-keyboard", result.stdout)
        self.assertIn("--gripper-open-angle", result.stdout)
        self.assertIn("--gripper-closed-angle", result.stdout)
        self.assertIn("--gripper-open-position", result.stdout)
        self.assertIn("--gripper-closed-position", result.stdout)
        self.assertIn("--gripper-position-range", result.stdout)
        self.assertIn("--gripper-open-key", result.stdout)
        self.assertIn("--gripper-close-key", result.stdout)
        self.assertIn("--gripper-speed", result.stdout)
        self.assertIn("--gripper-tap-step", result.stdout)
        self.assertIn("--gripper-report-interval-s", result.stdout)
        self.assertIn("--gripper-command-file", result.stdout)
        self.assertIn("--gripper-http-port", result.stdout)
        self.assertIn("--play-sounds", result.stdout)

    def test_record_defaults_use_visible_gripper_motion(self):
        repo_root = Path(__file__).resolve().parents[1]
        module = ast.parse((repo_root / "lerobot/scripts/control_robot.py").read_text(encoding="utf-8"))
        record_def = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "record")
        positional_args = record_def.args.args
        defaults = record_def.args.defaults
        defaults_by_name = {
            arg.arg: default.value
            for arg, default in zip(positional_args[-len(defaults) :], defaults, strict=True)
            if isinstance(default, ast.Constant)
        }

        self.assertEqual(defaults_by_name["gripper_speed"], 90)
        self.assertEqual(defaults_by_name["gripper_tap_step"], 45)
        self.assertEqual(defaults_by_name["gripper_http_port"], 8765)

    def test_gripper_keyboard_diagnostic_help_lists_options(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "lerobot/scripts/test_gripper_keyboard.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gripper-open-key", result.stdout)
        self.assertIn("--gripper-close-key", result.stdout)
        self.assertIn("--gripper-speed", result.stdout)
        self.assertIn("--gripper-tap-step", result.stdout)


if __name__ == "__main__":
    unittest.main()
