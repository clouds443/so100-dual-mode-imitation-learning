"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py teleoperate

# Remove the cameras from the robot definition. They are not used in 'teleoperate' anyway.
python lerobot/scripts/control_robot.py teleoperate --robot-overrides '~cameras'
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

- Train on this dataset with the ACT policy:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/eval_act_koch_real \
    --num-episodes 10 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

repo_root = str(Path(__file__).resolve().parents[2])
if sys.path[0] != repo_root:
    sys.path.insert(0, repo_root)

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    delete_current_episode,
    init_dataset,
    save_current_episode,
)
from lerobot.common.robot_devices.control_utils import (
    GlobalKeyboardPollListener,
    FileCommandListener,
    GripperHttpListener,
    KeyboardListenerGroup,
    control_loop,
    has_method,
    init_keyboard_listener,
    init_policy,
    log_control_info,
    record_episode,
    request_gripper_close,
    request_gripper_open,
    reset_environment,
    resolve_gripper_angle_limits,
    sanity_check_dataset_name,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import follower_gripper_angle_to_raw_position
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, arms: list[str] | None):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    if arms is None:
        arms = robot.available_arms

    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(
    robot: Robot, fps: int | None = None, teleop_time_s: float | None = None, display_cameras: bool = False
):
    control_loop(
        robot,
        control_time_s=teleop_time_s,
        fps=fps,
        teleoperate=True,
        display_cameras=display_cameras,
    )


@safe_disconnect
def record(
    robot: Robot,
    root: str,
    repo_id: str,
    pretrained_policy_name_or_path: str | None = None,
    policy_overrides: List[str] | None = None,
    record_control_mode: str = "teleoperate",
    kinesthetic_gripper_keyboard: int = 0,
    gripper_open_angle: float | None = None,
    gripper_closed_angle: float | None = None,
    gripper_open_position: float | None = None,
    gripper_closed_position: float | None = None,
    gripper_position_range: float | None = None,
    gripper_open_key: str = "o",
    gripper_close_key: str = "c",
    gripper_speed: float = 90,
    gripper_tap_step: float = 45,
    gripper_report_interval_s: float = 0.25,
    gripper_command_file: str = ".cache/gripper_command.txt",
    gripper_http_port: int = 8765,
    fps: int | None = None,
    warmup_time_s=2,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    num_image_writer_processes=0,
    num_image_writer_threads_per_camera=4,
    force_override=False,
    display_cameras=True,
    play_sounds=True,
):
    # TODO(rcadene): Add option to record logs
    listener = None
    events = None
    policy = None
    device = None
    use_amp = None

    if record_control_mode not in ["teleoperate", "kinesthetic"]:
        raise ValueError(
            f"record_control_mode must be 'teleoperate' or 'kinesthetic', but got {record_control_mode!r}."
        )
    if len(gripper_open_key) != 1 or len(gripper_close_key) != 1:
        raise ValueError("gripper_open_key and gripper_close_key must each be a single character.")

    gripper_open_angle, gripper_closed_angle = resolve_gripper_angle_limits(
        gripper_open_angle,
        gripper_closed_angle,
        gripper_open_position,
        gripper_closed_position,
        gripper_position_range,
    )

    # Load pretrained policy
    if pretrained_policy_name_or_path is not None:
        policy, policy_fps, device, use_amp = init_policy(pretrained_policy_name_or_path, policy_overrides)

        if fps is None:
            fps = policy_fps
            logging.warning(f"No fps provided, so using the fps from policy config ({policy_fps}).")
        elif fps != policy_fps:
            logging.warning(
                f"There is a mismatch between the provided fps ({fps}) and the one from policy config ({policy_fps})."
            )

    kinesthetic_gripper_keyboard_enabled = (
        policy is None and record_control_mode == "kinesthetic" and bool(kinesthetic_gripper_keyboard)
    )
    if kinesthetic_gripper_keyboard_enabled:
        logging.warning(
            f"Kinesthetic gripper keyboard uses open_angle={gripper_open_angle} "
            f"raw={follower_gripper_angle_to_raw_position(gripper_open_angle)}, "
            f"closed_angle={gripper_closed_angle} "
            f"raw={follower_gripper_angle_to_raw_position(gripper_closed_angle)}, "
            f"speed={gripper_speed} deg/s, tap_step={gripper_tap_step} deg."
        )

    # Create empty dataset or load existing saved episodes
    sanity_check_dataset_name(repo_id, policy)
    dataset = init_dataset(
        repo_id,
        root,
        force_override,
        fps,
        video,
        write_images=robot.has_camera,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads=num_image_writer_threads_per_camera * robot.num_cameras,
    )

    if not robot.is_connected:
        robot.connect()

    if policy is None and record_control_mode == "kinesthetic":
        if not has_method(robot, "set_follower_torque") or not has_method(robot, "kinesthetic_step"):
            raise ValueError(
                f"The robot type '{robot.robot_type}' does not support kinesthetic recording."
            )
        if kinesthetic_gripper_keyboard_enabled and (
            not has_method(robot, "set_follower_gripper_angle")
            or not has_method(robot, "step_follower_gripper_towards")
        ):
            raise ValueError(
                f"The robot type '{robot.robot_type}' does not support keyboard gripper control."
            )

        if kinesthetic_gripper_keyboard_enabled:
            non_gripper_motor_names = None
            for arm_name, arm in robot.follower_arms.items():
                if "gripper" not in arm.motor_names:
                    raise ValueError(f"Follower arm '{arm_name}' does not have a 'gripper' motor.")

                arm_non_gripper_motor_names = [
                    motor_name for motor_name in arm.motor_names if motor_name != "gripper"
                ]
                if non_gripper_motor_names is None:
                    non_gripper_motor_names = arm_non_gripper_motor_names
                elif non_gripper_motor_names != arm_non_gripper_motor_names:
                    raise ValueError(
                        "Keyboard gripper control expects all follower arms to share motor names."
                    )

            logging.warning(
                "Kinesthetic recording is enabled. Non-gripper follower torque will be disabled; "
                "support the arm by hand. The follower gripper torque stays enabled."
            )
            robot.set_follower_torque(False, motor_names=non_gripper_motor_names)
            robot.set_follower_torque(True, motor_names="gripper")
            robot.set_follower_gripper_angle(gripper_open_angle)
        else:
            logging.warning(
                "Kinesthetic recording is enabled. Follower torque will be disabled; support the arm by hand."
            )
            robot.set_follower_torque(False)

    listener, events = init_keyboard_listener(
        gripper_open_key=gripper_open_key,
        gripper_close_key=gripper_close_key,
    )
    global_key_listener = None
    file_command_listener = None
    http_listener = None
    if kinesthetic_gripper_keyboard_enabled:
        global_key_listener = GlobalKeyboardPollListener(
            events,
            gripper_open_key=gripper_open_key,
            gripper_close_key=gripper_close_key,
        )
        global_key_listener.start()
        file_command_listener = FileCommandListener(
            gripper_command_file,
            on_gripper_open=lambda: request_gripper_open(events),
            on_gripper_close=lambda: request_gripper_close(events),
        )
        file_command_listener.start()
        if gripper_http_port:
            http_listener = GripperHttpListener(
                port=gripper_http_port,
                gripper_open_key=gripper_open_key,
                gripper_close_key=gripper_close_key,
                on_gripper_open=lambda: request_gripper_open(events),
                on_gripper_close=lambda: request_gripper_close(events),
            )
            http_listener.start()
        listener = KeyboardListenerGroup(
            [listener, global_key_listener, file_command_listener, http_listener]
        )

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", play_sounds)
    warmup_record(
        robot,
        events,
        enable_teleoperation,
        warmup_time_s,
        display_cameras,
        fps,
        record_control_mode=record_control_mode,
        kinesthetic_gripper_keyboard=kinesthetic_gripper_keyboard_enabled,
        gripper_open_angle=gripper_open_angle,
        gripper_closed_angle=gripper_closed_angle,
        gripper_speed=gripper_speed,
        gripper_tap_step=gripper_tap_step,
        gripper_open_key=gripper_open_key,
        gripper_close_key=gripper_close_key,
        gripper_report_interval_s=gripper_report_interval_s,
    )

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    while True:
        if dataset["num_episodes"] >= num_episodes:
            break

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index}", play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=display_cameras,
            policy=policy,
            device=device,
            use_amp=use_amp,
            fps=fps,
            record_control_mode=record_control_mode,
            kinesthetic_gripper_keyboard=kinesthetic_gripper_keyboard_enabled,
            gripper_open_angle=gripper_open_angle,
            gripper_closed_angle=gripper_closed_angle,
            gripper_speed=gripper_speed,
            gripper_tap_step=gripper_tap_step,
            gripper_open_key=gripper_open_key,
            gripper_close_key=gripper_close_key,
            gripper_report_interval_s=gripper_report_interval_s,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (episode_index < num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", play_sounds)
            reset_environment(robot, events, reset_time_s)

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            delete_current_episode(dataset)
            continue

        # Increment by one dataset["current_episode_index"]
        save_current_episode(dataset)

        if events["stop_recording"]:
            break

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    lerobot_dataset = create_lerobot_dataset(dataset, run_compute_stats, push_to_hub, tags, play_sounds)

    log_say("Exiting", play_sounds)
    return lerobot_dataset


@safe_disconnect
def replay(
    robot: Robot, episode: int, fps: int | None = None, root="data", repo_id="lerobot/debug", play_sounds=True
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", play_sounds, blocking=True)
    for idx in range(from_idx, to_idx):
        start_episode_t = time.perf_counter()

        action = items[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    base_parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_calib = subparsers.add_parser("calibrate", parents=[base_parser])
    parser_calib.add_argument(
        "--arms",
        type=str,
        nargs="*",
        help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_teleop.add_argument(
        "--display-cameras",
        type=int,
        default=1,
        help="Display all cameras on screen (set to 1 to display or 0).",
    )

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--record-control-mode",
        type=str,
        choices=["teleoperate", "kinesthetic"],
        default="teleoperate",
        help=(
            "How to collect demonstration actions when no policy is provided. "
            "'teleoperate' uses a leader arm; 'kinesthetic' records hand-guided follower positions."
        ),
    )
    parser_record.add_argument(
        "--kinesthetic-gripper-keyboard",
        type=int,
        default=0,
        help=(
            "Enable keyboard gripper control during kinesthetic recording without a policy. "
            "Set to 1 to keep gripper torque enabled while other follower joints are hand-guided."
        ),
    )
    parser_record.add_argument(
        "--gripper-open-angle",
        dest="gripper_open_angle",
        type=float,
        default=None,
        help=(
            "Follower gripper open target angle used by raw keyboard kinesthetic control. "
            "This follows the reference script angle domain [-90, 90]."
        ),
    )
    parser_record.add_argument(
        "--gripper-open-position",
        dest="gripper_open_position",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility value for older commands. "
            "Values in the old 0..100 or 0..180 position range are mapped to gripper angle [-90, 90]."
        ),
    )
    parser_record.add_argument(
        "--gripper-closed-angle",
        dest="gripper_closed_angle",
        type=float,
        default=None,
        help=(
            "Follower gripper closed target angle used by raw keyboard kinesthetic control. "
            "This follows the reference script angle domain [-90, 90]."
        ),
    )
    parser_record.add_argument(
        "--gripper-closed-position",
        dest="gripper_closed_position",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility value for older commands. "
            "Values in the old 0..100 or 0..180 position range are mapped to gripper angle [-90, 90]."
        ),
    )
    parser_record.add_argument(
        "--gripper-position-range",
        type=float,
        choices=[100, 180],
        default=None,
        help=(
            "Range used to interpret deprecated gripper position arguments. "
            "Use 180 for old 0..180 position values such as 90 as midpoint; "
            "use 100 for old 0..100 percentage-like values."
        ),
    )
    parser_record.add_argument(
        "--gripper-open-key",
        type=str,
        default="o",
        help="Keyboard key held to continuously open the follower gripper.",
    )
    parser_record.add_argument(
        "--gripper-close-key",
        type=str,
        default="c",
        help="Keyboard key held to continuously close the follower gripper.",
    )
    parser_record.add_argument(
        "--gripper-speed",
        type=float,
        default=90,
        help="Follower gripper angle change in degrees per second while an open/close key is held.",
    )
    parser_record.add_argument(
        "--gripper-tap-step",
        type=float,
        default=45,
        help="Follower gripper angle step in degrees for a quick open/close key tap.",
    )
    parser_record.add_argument(
        "--gripper-report-interval-s",
        type=float,
        default=0.25,
        help=(
            "Minimum seconds between GRIPPER_OPEN/CLOSE raw state prints during kinesthetic "
            "keyboard gripper control. Set to 0 to print every commanded step."
        ),
    )
    parser_record.add_argument(
        "--gripper-command-file",
        type=str,
        default=".cache/gripper_command.txt",
        help="Local command file watched during keyboard kinesthetic recording. Write 'open' or 'close' to control the gripper.",
    )
    parser_record.add_argument(
        "--gripper-http-port",
        type=int,
        default=8765,
        help=(
            "Local HTTP gripper control port for kinesthetic recording. "
            "Open http://127.0.0.1:{port}/ and press O/C in the page. Set to 0 to disable."
        ),
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser_record.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "--play-sounds",
        type=int,
        default=1,
        help="Play spoken status prompts during recording. Set to 0 if Windows speech synthesis prints errors.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    if control_mode == "calibrate":
        calibrate(robot, **kwargs)

    elif control_mode == "teleoperate":
        teleoperate(robot, **kwargs)

    elif control_mode == "record":
        record(robot, **kwargs)

    elif control_mode == "replay":
        replay(robot, **kwargs)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
