#!/usr/bin/env python

import argparse
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if sys.path[0] != repo_root:
    sys.path.insert(0, repo_root)

from lerobot.common.robot_devices.control_utils import (
    GlobalKeyboardPollListener,
    KeyboardListenerGroup,
    FileCommandListener,
    handle_cv2_key_code,
    init_keyboard_listener,
    poll_terminal_keyboard,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import follower_gripper_angle_to_raw_position
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config


def read_raw_gripper_state(robot):
    for arm in robot.follower_arms.values():
        if "gripper" not in arm.motor_names:
            raise ValueError("Follower arm does not have a 'gripper' motor.")

        motor_id, motor_model = arm.motors["gripper"]
        if hasattr(arm, "read_with_motor_ids"):
            present = arm.read_with_motor_ids([motor_model], [motor_id], "Present_Position")[0]
            goal = arm.read_with_motor_ids([motor_model], [motor_id], "Goal_Position")[0]
            return int(present), int(goal)

        position = arm.read("Present_Position", "gripper")[0]
        return float(position), float(position)

    raise ValueError("No follower arm is configured.")


def step_and_report(robot, direction, target_angle, step_size, lower_angle, upper_angle):
    next_angle = robot.step_follower_gripper_towards(
        target_position=target_angle,
        step_size=step_size,
        min_position=lower_angle,
        max_position=upper_angle,
    )
    present_raw, goal_raw = read_raw_gripper_state(robot)
    target_raw = follower_gripper_angle_to_raw_position(target_angle)
    print(
        f"GRIPPER_{direction.upper()} next_angle={next_angle:.1f} "
        f"target_raw={target_raw} goal_raw={goal_raw} present_raw={present_raw}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test O/C keyboard gripper control on the student follower arm without recording data."
    )
    parser.add_argument("--robot-path", default="lerobot/configs/robot/so100_student.yaml")
    parser.add_argument("--robot-overrides", type=str, nargs="*", default=None)
    parser.add_argument("--duration-s", type=float, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gripper-open-angle", type=float, default=90)
    parser.add_argument("--gripper-closed-angle", type=float, default=-90)
    parser.add_argument("--gripper-open-key", type=str, default="o")
    parser.add_argument("--gripper-close-key", type=str, default="c")
    parser.add_argument("--gripper-speed", type=float, default=90)
    parser.add_argument("--gripper-tap-step", type=float, default=45)
    parser.add_argument("--gripper-command-file", type=str, default=".cache/gripper_command.txt")
    args = parser.parse_args()

    if len(args.gripper_open_key) != 1 or len(args.gripper_close_key) != 1:
        raise ValueError("gripper open/close keys must each be a single character.")

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)
    listener = None

    lower_angle = min(args.gripper_open_angle, args.gripper_closed_angle)
    upper_angle = max(args.gripper_open_angle, args.gripper_closed_angle)

    def open_once():
        step_and_report(
            robot,
            "open",
            args.gripper_open_angle,
            abs(args.gripper_tap_step),
            lower_angle,
            upper_angle,
        )

    def close_once():
        step_and_report(
            robot,
            "close",
            args.gripper_closed_angle,
            abs(args.gripper_tap_step),
            lower_angle,
            upper_angle,
        )

    try:
        robot.connect()
        arm = next(iter(robot.follower_arms.values()))
        non_gripper_motor_names = [motor_name for motor_name in arm.motor_names if motor_name != "gripper"]
        robot.set_follower_torque(False, motor_names=non_gripper_motor_names)
        robot.set_follower_torque(True, motor_names="gripper")
        robot.set_follower_gripper_angle(args.gripper_open_angle)
        time.sleep(0.5)
        present_raw, goal_raw = read_raw_gripper_state(robot)
        print(f"READY goal_raw={goal_raw} present_raw={present_raw}", flush=True)
        print(
            f"Hold '{args.gripper_open_key.upper()}' to open, "
            f"hold '{args.gripper_close_key.upper()}' to close. Ctrl+C exits.",
            flush=True,
        )

        keyboard_listener, events = init_keyboard_listener(
            gripper_open_key=args.gripper_open_key,
            gripper_close_key=args.gripper_close_key,
            on_gripper_open=open_once,
            on_gripper_close=close_once,
        )
        global_listener = GlobalKeyboardPollListener(
            events,
            gripper_open_key=args.gripper_open_key,
            gripper_close_key=args.gripper_close_key,
            on_gripper_open=open_once,
            on_gripper_close=close_once,
        )
        command_listener = FileCommandListener(
            args.gripper_command_file,
            on_gripper_open=open_once,
            on_gripper_close=close_once,
        )
        global_listener.start()
        command_listener.start()
        listener = KeyboardListenerGroup([keyboard_listener, global_listener, command_listener])

        start_t = time.perf_counter()
        while time.perf_counter() - start_t < args.duration_s:
            loop_t = time.perf_counter()
            poll_terminal_keyboard(events, args.gripper_open_key, args.gripper_close_key)

            open_pressed = events.get("gripper_open_pressed", False)
            close_pressed = events.get("gripper_close_pressed", False)
            requested_direction = events.get("gripper_requested_direction")

            if open_pressed != close_pressed:
                target_angle = args.gripper_open_angle if open_pressed else args.gripper_closed_angle
                direction = "open" if open_pressed else "close"
                step_and_report(
                    robot,
                    direction,
                    target_angle,
                    abs(args.gripper_speed) / args.fps,
                    lower_angle,
                    upper_angle,
                )
                events["gripper_open_requested"] = False
                events["gripper_close_requested"] = False
                events["gripper_requested_direction"] = None
            elif requested_direction in ("open", "close"):
                if requested_direction == "open":
                    open_once()
                else:
                    close_once()
                events["gripper_open_requested"] = False
                events["gripper_close_requested"] = False
                events["gripper_requested_direction"] = None

            try:
                import cv2

                handle_cv2_key_code(cv2.waitKey(1), events, args.gripper_open_key, args.gripper_close_key)
            except Exception:
                pass

            busy_wait(1 / args.fps - (time.perf_counter() - loop_t))

    except KeyboardInterrupt:
        print("Stopping gripper keyboard test.", flush=True)
    finally:
        if listener is not None:
            listener.stop()
        if getattr(robot, "is_connected", False):
            robot.set_follower_torque(False, motor_names="gripper")
            robot.disconnect()


if __name__ == "__main__":
    main()
