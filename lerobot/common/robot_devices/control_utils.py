########################################################################################
# Utilities
########################################################################################


import logging
import os
import threading
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch
import tqdm
from termcolor import colored

from lerobot.common.datasets.populate_dataset import add_frame, safe_stop_image_writer
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return False


def has_method(_object: object, method_name: str):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


def _normalized_key_char(key):
    char = getattr(key, "char", None)
    if char is not None:
        return char.lower()

    vk = getattr(key, "vk", None)
    if isinstance(vk, int):
        try:
            return chr(vk).lower()
        except ValueError:
            return None

    return None


def _is_key_char(key, expected_char):
    return _normalized_key_char(key) == expected_char.lower()


def _refresh_gripper_pressed(events):
    events["gripper_open_pressed"] = bool(
        events.get("_gripper_open_listener_pressed", False)
        or events.get("_gripper_open_global_pressed", False)
    )
    events["gripper_close_pressed"] = bool(
        events.get("_gripper_close_listener_pressed", False)
        or events.get("_gripper_close_global_pressed", False)
    )


def _request_gripper_open(events):
    events["gripper_open_requested"] = True
    events["gripper_close_requested"] = False
    events["gripper_requested_direction"] = "open"


def _request_gripper_close(events):
    events["gripper_close_requested"] = True
    events["gripper_open_requested"] = False
    events["gripper_requested_direction"] = "close"


def request_gripper_open(events):
    _request_gripper_open(events)


def request_gripper_close(events):
    _request_gripper_close(events)


def handle_keyboard_press(
    key,
    events,
    keyboard_module,
    gripper_open_key="o",
    gripper_close_key="c",
    on_gripper_open=None,
    on_gripper_close=None,
):
    if key == keyboard_module.Key.right:
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
    elif key == keyboard_module.Key.left:
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif key == keyboard_module.Key.esc:
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True
    elif _is_key_char(key, gripper_open_key):
        if not events.get("gripper_open_pressed", False):
            print(f"Gripper open key '{gripper_open_key}' pressed.")
        events["_gripper_open_listener_pressed"] = True
        _refresh_gripper_pressed(events)
        _request_gripper_open(events)
        if on_gripper_open is not None:
            on_gripper_open()
            events["gripper_open_requested"] = False
            events["gripper_requested_direction"] = None
    elif _is_key_char(key, gripper_close_key):
        if not events.get("gripper_close_pressed", False):
            print(f"Gripper close key '{gripper_close_key}' pressed.")
        events["_gripper_close_listener_pressed"] = True
        _refresh_gripper_pressed(events)
        _request_gripper_close(events)
        if on_gripper_close is not None:
            on_gripper_close()
            events["gripper_close_requested"] = False
            events["gripper_requested_direction"] = None


def handle_keyboard_release(key, events, keyboard_module, gripper_open_key="o", gripper_close_key="c"):
    if _is_key_char(key, gripper_open_key):
        events["_gripper_open_listener_pressed"] = False
        _refresh_gripper_pressed(events)
    elif _is_key_char(key, gripper_close_key):
        events["_gripper_close_listener_pressed"] = False
        _refresh_gripper_pressed(events)


def handle_keyboard_event(key, events, keyboard_module):
    handle_keyboard_press(key, events, keyboard_module)


def handle_polled_keyboard_char(char, events, gripper_open_key="o", gripper_close_key="c"):
    if not char:
        return

    lowered = char.lower()
    if lowered == gripper_open_key.lower():
        _request_gripper_open(events)
    elif lowered == gripper_close_key.lower():
        _request_gripper_close(events)
    elif char == "\x1b":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


def handle_cv2_key_code(key_code, events, gripper_open_key="o", gripper_close_key="c"):
    if key_code in (-1, 255):
        return
    key_code = key_code & 0xFF
    if key_code == 27:
        handle_polled_keyboard_char("\x1b", events, gripper_open_key, gripper_close_key)
        return
    try:
        handle_polled_keyboard_char(chr(key_code), events, gripper_open_key, gripper_close_key)
    except ValueError:
        return


def poll_terminal_keyboard(events, gripper_open_key="o", gripper_close_key="c"):
    if os.name != "nt":
        return

    try:
        import msvcrt
    except ImportError:
        return

    while msvcrt.kbhit():
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            special = msvcrt.getwch()
            if special == "M":
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif special == "K":
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
        else:
            handle_polled_keyboard_char(char, events, gripper_open_key, gripper_close_key)


def _windows_key_state_reader(key: str) -> bool:
    if os.name != "nt" or len(key) != 1:
        return False

    try:
        import ctypes
    except ImportError:
        return False

    return bool(ctypes.windll.user32.GetAsyncKeyState(ord(key.upper())) & 0x8000)


def poll_global_keyboard_state(
    events,
    gripper_open_key="o",
    gripper_close_key="c",
    on_gripper_open=None,
    on_gripper_close=None,
    key_state_reader=None,
):
    if key_state_reader is None:
        key_state_reader = _windows_key_state_reader

    open_down = bool(key_state_reader(gripper_open_key))
    close_down = bool(key_state_reader(gripper_close_key))
    events["_gripper_open_global_pressed"] = open_down
    events["_gripper_close_global_pressed"] = close_down
    _refresh_gripper_pressed(events)

    if open_down and not close_down:
        _request_gripper_open(events)
        if on_gripper_open is not None:
            on_gripper_open()
            events["gripper_open_requested"] = False
            events["gripper_requested_direction"] = None
    elif close_down and not open_down:
        _request_gripper_close(events)
        if on_gripper_close is not None:
            on_gripper_close()
            events["gripper_close_requested"] = False
            events["gripper_requested_direction"] = None


class GlobalKeyboardPollListener:
    def __init__(
        self,
        events,
        gripper_open_key="o",
        gripper_close_key="c",
        on_gripper_open=None,
        on_gripper_close=None,
        poll_interval_s=0.08,
        key_state_reader=None,
    ):
        self.events = events
        self.gripper_open_key = gripper_open_key
        self.gripper_close_key = gripper_close_key
        self.on_gripper_open = on_gripper_open
        self.on_gripper_close = on_gripper_close
        self.poll_interval_s = poll_interval_s
        self.key_state_reader = key_state_reader
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gripper-key-poll", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1)

    def _run(self):
        while not self._stop_event.is_set():
            poll_global_keyboard_state(
                self.events,
                gripper_open_key=self.gripper_open_key,
                gripper_close_key=self.gripper_close_key,
                on_gripper_open=self.on_gripper_open,
                on_gripper_close=self.on_gripper_close,
                key_state_reader=self.key_state_reader,
            )
            self._stop_event.wait(self.poll_interval_s)


def handle_windows_hotkey_id(hotkey_id, on_gripper_open=None, on_gripper_close=None):
    if hotkey_id == 1 and on_gripper_close is not None:
        on_gripper_close()
    elif hotkey_id == 2 and on_gripper_open is not None:
        on_gripper_open()


def handle_gripper_command_text(command_text, on_gripper_open=None, on_gripper_close=None):
    command = command_text.replace("\ufeff", "").strip().lower()
    if command in ("open", "o"):
        if on_gripper_open is not None:
            on_gripper_open()
        return True
    if command in ("close", "c"):
        if on_gripper_close is not None:
            on_gripper_close()
        return True
    return False


class FileCommandListener:
    def __init__(
        self,
        command_file,
        on_gripper_open=None,
        on_gripper_close=None,
        poll_interval_s=0.05,
    ):
        self.command_file = Path(command_file)
        self.on_gripper_open = on_gripper_open
        self.on_gripper_close = on_gripper_close
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gripper-command-file", daemon=True)

    def start(self):
        self.command_file.parent.mkdir(parents=True, exist_ok=True)
        self.command_file.write_text("", encoding="utf-8")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                command = self.command_file.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                command = ""

            if command and handle_gripper_command_text(
                command,
                on_gripper_open=self.on_gripper_open,
                on_gripper_close=self.on_gripper_close,
            ):
                self.command_file.write_text("", encoding="utf-8")

            self._stop_event.wait(self.poll_interval_s)


class GripperHttpListener:
    def __init__(
        self,
        port=8765,
        on_gripper_open=None,
        on_gripper_close=None,
        gripper_open_key="o",
        gripper_close_key="c",
    ):
        self.host = "127.0.0.1"
        self.port = int(port)
        self.on_gripper_open = on_gripper_open
        self.on_gripper_close = on_gripper_close
        self.gripper_open_key = gripper_open_key
        self.gripper_close_key = gripper_close_key
        self.httpd = None
        self._thread = None
        self.url = f"http://{self.host}:{self.port}"

    def start(self):
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                command = parsed.path.strip("/").lower()
                if not command:
                    command = parse_qs(parsed.query).get("command", [""])[0].lower()

                if command in ("open", "o"):
                    if listener.on_gripper_open is not None:
                        listener.on_gripper_open()
                    self._send_text("open\n")
                elif command in ("close", "c"):
                    if listener.on_gripper_close is not None:
                        listener.on_gripper_close()
                    self._send_text("close\n")
                elif parsed.path in ("", "/"):
                    self._send_html(listener._html())
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_text(self, text):
                body = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        last_error = None
        for port in range(self.port, self.port + 20):
            try:
                self.httpd = ThreadingHTTPServer((self.host, port), Handler)
                self.port = self.httpd.server_address[1]
                self.url = f"http://{self.host}:{self.port}"
                break
            except OSError as exc:
                last_error = exc
        if self.httpd is None:
            raise last_error

        self._thread = threading.Thread(target=self.httpd.serve_forever, name="gripper-http", daemon=True)
        self._thread.start()
        print(f"Gripper HTTP control: {self.url}", flush=True)

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1)

    def _html(self):
        open_key = self.gripper_open_key.upper()
        close_key = self.gripper_close_key.upper()
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gripper Control</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; line-height: 1.5; }}
    button {{ font-size: 20px; margin-right: 12px; padding: 10px 18px; }}
    #status {{ margin-top: 16px; font-size: 18px; }}
  </style>
</head>
<body>
  <h1>Gripper Control</h1>
  <button onclick="sendCommand('open')">Open ({open_key})</button>
  <button onclick="sendCommand('close')">Close ({close_key})</button>
  <div id="status">Ready</div>
  <script>
    async function sendCommand(command) {{
      const response = await fetch('/' + command);
      document.getElementById('status').textContent = command + ' ' + response.status;
    }}
    document.addEventListener('keydown', (event) => {{
      const key = event.key.toLowerCase();
      if (key === '{self.gripper_open_key.lower()}') sendCommand('open');
      if (key === '{self.gripper_close_key.lower()}') sendCommand('close');
    }});
  </script>
</body>
</html>
"""


class WindowsHotkeyListener:
    WM_HOTKEY = 0x0312

    def __init__(
        self,
        gripper_open_key="o",
        gripper_close_key="c",
        on_gripper_open=None,
        on_gripper_close=None,
        poll_interval_s=0.02,
    ):
        self.gripper_open_key = gripper_open_key
        self.gripper_close_key = gripper_close_key
        self.on_gripper_open = on_gripper_open
        self.on_gripper_close = on_gripper_close
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gripper-hotkey", daemon=True)

    def start(self):
        if os.name == "nt":
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _run(self):
        try:
            import ctypes
        except ImportError:
            return

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", ctypes.c_void_p),
                ("message", ctypes.c_uint),
                ("wParam", ctypes.c_size_t),
                ("lParam", ctypes.c_size_t),
                ("time", ctypes.c_uint),
                ("pt_x", ctypes.c_long),
                ("pt_y", ctypes.c_long),
            ]

        user32 = ctypes.windll.user32
        close_registered = user32.RegisterHotKey(None, 1, 0, ord(self.gripper_close_key.upper()))
        open_registered = user32.RegisterHotKey(None, 2, 0, ord(self.gripper_open_key.upper()))
        if not close_registered or not open_registered:
            logging.warning(
                "Windows hotkey registration for gripper keys failed. Other keyboard paths remain active."
            )

        msg = MSG()
        try:
            while not self._stop_event.is_set():
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == self.WM_HOTKEY:
                        handle_windows_hotkey_id(
                            int(msg.wParam),
                            on_gripper_open=self.on_gripper_open,
                            on_gripper_close=self.on_gripper_close,
                        )
                self._stop_event.wait(self.poll_interval_s)
        finally:
            if close_registered:
                user32.UnregisterHotKey(None, 1)
            if open_registered:
                user32.UnregisterHotKey(None, 2)


class KeyboardListenerGroup:
    def __init__(self, listeners):
        self.listeners = [listener for listener in listeners if listener is not None]

    def stop(self):
        for listener in self.listeners:
            listener.stop()


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener(
    gripper_open_key="o",
    gripper_close_key="c",
    on_gripper_open=None,
    on_gripper_close=None,
):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["gripper_open_pressed"] = False
    events["gripper_close_pressed"] = False
    events["_gripper_open_listener_pressed"] = False
    events["_gripper_close_listener_pressed"] = False
    events["_gripper_open_global_pressed"] = False
    events["_gripper_close_global_pressed"] = False
    events["gripper_open_requested"] = False
    events["gripper_close_requested"] = False
    events["gripper_requested_direction"] = None

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            handle_keyboard_press(
                key,
                events,
                keyboard,
                gripper_open_key,
                gripper_close_key,
                on_gripper_open=on_gripper_open,
                on_gripper_close=on_gripper_close,
            )
        except Exception as e:
            print(f"Error handling key press: {e}")

    def on_release(key):
        try:
            handle_keyboard_release(key, events, keyboard, gripper_open_key, gripper_close_key)
        except Exception as e:
            print(f"Error handling key release: {e}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    return listener, events


def resolve_gripper_angle_limits(
    gripper_open_angle: float | None = None,
    gripper_closed_angle: float | None = None,
    gripper_open_position: float | None = None,
    gripper_closed_position: float | None = None,
    gripper_position_range: float | None = None,
) -> tuple[float, float]:
    legacy_position_range = _resolve_legacy_gripper_position_range(
        gripper_open_position,
        gripper_closed_position,
        gripper_position_range,
    )
    if gripper_open_angle is None:
        gripper_open_angle = (
            90
            if gripper_open_position is None
            else _legacy_gripper_position_to_angle(gripper_open_position, legacy_position_range)
        )
    if gripper_closed_angle is None:
        gripper_closed_angle = (
            -90
            if gripper_closed_position is None
            else _legacy_gripper_position_to_angle(gripper_closed_position, legacy_position_range)
        )
    return float(gripper_open_angle), float(gripper_closed_angle)


def _resolve_legacy_gripper_position_range(
    gripper_open_position: float | None,
    gripper_closed_position: float | None,
    gripper_position_range: float | None,
) -> float:
    if gripper_position_range is not None:
        gripper_position_range = float(gripper_position_range)
        if gripper_position_range not in (100.0, 180.0):
            raise ValueError("gripper_position_range must be either 100 or 180.")
        return gripper_position_range

    positions = [
        float(position)
        for position in (gripper_open_position, gripper_closed_position)
        if position is not None
    ]
    if any(position > 100.0 for position in positions):
        return 180.0
    return 100.0


def _legacy_gripper_position_to_angle(position: float, position_range: float) -> float:
    """Map old gripper position CLI values onto the raw reference-script angle domain."""
    position = float(position)
    angle = (position / float(position_range)) * 180.0 - 90.0
    return min(max(angle, -90.0), 90.0)


def init_policy(pretrained_policy_name_or_path, policy_overrides):
    """Instantiate the policy and load fps, device and use_amp from config yaml"""
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)
    use_amp = hydra_cfg.use_amp
    policy_fps = hydra_cfg.env.fps

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)
    return policy, policy_fps, device, use_amp


def warmup_record(
    robot,
    events,
    enable_teloperation,
    warmup_time_s,
    display_cameras,
    fps,
    record_control_mode="teleoperate",
    kinesthetic_gripper_keyboard=False,
    gripper_open_angle=None,
    gripper_closed_angle=None,
    gripper_open_position=None,
    gripper_closed_position=None,
    gripper_speed=90,
    gripper_tap_step=45,
    gripper_open_key="o",
    gripper_close_key="c",
    gripper_report_interval_s=None,
):
    gripper_open_angle, gripper_closed_angle = resolve_gripper_angle_limits(
        gripper_open_angle,
        gripper_closed_angle,
        gripper_open_position,
        gripper_closed_position,
    )
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teloperation,
        record_control_mode=record_control_mode,
        kinesthetic_gripper_keyboard=kinesthetic_gripper_keyboard,
        gripper_open_angle=gripper_open_angle,
        gripper_closed_angle=gripper_closed_angle,
        gripper_speed=gripper_speed,
        gripper_tap_step=gripper_tap_step,
        gripper_open_key=gripper_open_key,
        gripper_close_key=gripper_close_key,
        gripper_report_interval_s=gripper_report_interval_s,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    device,
    use_amp,
    fps,
    record_control_mode="teleoperate",
    kinesthetic_gripper_keyboard=False,
    gripper_open_angle=None,
    gripper_closed_angle=None,
    gripper_open_position=None,
    gripper_closed_position=None,
    gripper_speed=90,
    gripper_tap_step=45,
    gripper_open_key="o",
    gripper_close_key="c",
    gripper_report_interval_s=None,
):
    gripper_open_angle, gripper_closed_angle = resolve_gripper_angle_limits(
        gripper_open_angle,
        gripper_closed_angle,
        gripper_open_position,
        gripper_closed_position,
    )
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
        record_control_mode=record_control_mode,
        kinesthetic_gripper_keyboard=kinesthetic_gripper_keyboard,
        gripper_open_angle=gripper_open_angle,
        gripper_closed_angle=gripper_closed_angle,
        gripper_speed=gripper_speed,
        gripper_tap_step=gripper_tap_step,
        gripper_open_key=gripper_open_key,
        gripper_close_key=gripper_close_key,
        gripper_report_interval_s=gripper_report_interval_s,
    )


def _maybe_report_gripper_step(
    robot,
    events,
    direction,
    next_position,
    target_position,
    gripper_report_interval_s,
):
    if not direction or gripper_report_interval_s is None:
        return

    now = time.perf_counter()
    last_report_t = events.get("_last_gripper_report_t")
    if (
        gripper_report_interval_s > 0
        and last_report_t is not None
        and now - last_report_t < gripper_report_interval_s
    ):
        return

    events["_last_gripper_report_t"] = now
    next_position_text = "unknown" if next_position is None else f"{float(next_position):.1f}"
    try:
        from lerobot.common.robot_devices.robots.manipulator import follower_gripper_angle_to_raw_position

        target_raw = follower_gripper_angle_to_raw_position(target_position)
        goal_raw, present_raw = robot.read_follower_gripper_raw_state()
        print(
            f"GRIPPER_{direction.upper()} next_angle={next_position_text} "
            f"target_raw={target_raw} goal_raw={goal_raw} present_raw={present_raw}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"GRIPPER_{direction.upper()} next_angle={next_position_text} "
            f"target={float(target_position):.1f} raw_state_unavailable={exc}",
            flush=True,
        )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    record_control_mode="teleoperate",
    display_cameras=False,
    dataset=None,
    events=None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
    kinesthetic_gripper_keyboard=False,
    gripper_open_angle=None,
    gripper_closed_angle=None,
    gripper_open_position=None,
    gripper_closed_position=None,
    gripper_speed=90,
    gripper_tap_step=45,
    gripper_open_key="o",
    gripper_close_key="c",
    gripper_report_interval_s=None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {
            "exit_early": False,
            "gripper_open_pressed": False,
            "gripper_close_pressed": False,
            "_gripper_open_listener_pressed": False,
            "_gripper_close_listener_pressed": False,
            "_gripper_open_global_pressed": False,
            "_gripper_close_global_pressed": False,
            "gripper_open_requested": False,
            "gripper_close_requested": False,
            "gripper_requested_direction": None,
        }
    else:
        events.setdefault("gripper_open_pressed", False)
        events.setdefault("gripper_close_pressed", False)
        events.setdefault("_gripper_open_listener_pressed", False)
        events.setdefault("_gripper_close_listener_pressed", False)
        events.setdefault("_gripper_open_global_pressed", False)
        events.setdefault("_gripper_close_global_pressed", False)
        events.setdefault("gripper_open_requested", False)
        events.setdefault("gripper_close_requested", False)
        events.setdefault("gripper_requested_direction", None)

    if control_time_s is None:
        control_time_s = float("inf")

    gripper_open_angle, gripper_closed_angle = resolve_gripper_angle_limits(
        gripper_open_angle,
        gripper_closed_angle,
        gripper_open_position,
        gripper_closed_position,
    )

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if record_control_mode not in ["teleoperate", "kinesthetic"]:
        raise ValueError(
            f"record_control_mode must be 'teleoperate' or 'kinesthetic', but got {record_control_mode!r}."
        )

    if dataset is not None and fps is not None and dataset["fps"] != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        poll_terminal_keyboard(events, gripper_open_key, gripper_close_key)

        if teleoperate:
            if record_control_mode == "kinesthetic":
                if kinesthetic_gripper_keyboard:
                    open_pressed = events.get("gripper_open_pressed", False)
                    close_pressed = events.get("gripper_close_pressed", False)
                    open_requested = events.get("gripper_open_requested", False)
                    close_requested = events.get("gripper_close_requested", False)
                    requested_direction = events.get("gripper_requested_direction")
                    target_position = None
                    step_size = None
                    gripper_direction = None

                    if open_pressed != close_pressed:
                        target_position = gripper_open_angle if open_pressed else gripper_closed_angle
                        step_size = abs(gripper_speed) / fps if fps else abs(gripper_speed)
                        gripper_direction = "open" if open_pressed else "close"
                        events["gripper_open_requested"] = False
                        events["gripper_close_requested"] = False
                        events["gripper_requested_direction"] = None
                    elif requested_direction in ("open", "close"):
                        target_position = gripper_open_angle if requested_direction == "open" else gripper_closed_angle
                        step_size = abs(gripper_tap_step)
                        gripper_direction = requested_direction
                        events["gripper_open_requested"] = False
                        events["gripper_close_requested"] = False
                        events["gripper_requested_direction"] = None
                    elif open_requested != close_requested:
                        target_position = gripper_open_angle if open_requested else gripper_closed_angle
                        step_size = abs(gripper_tap_step)
                        gripper_direction = "open" if open_requested else "close"
                        events["gripper_open_requested"] = False
                        events["gripper_close_requested"] = False
                        events["gripper_requested_direction"] = None
                    elif open_requested and close_requested:
                        events["gripper_open_requested"] = False
                        events["gripper_close_requested"] = False
                        events["gripper_requested_direction"] = None

                    if target_position is not None:
                        lower_position = min(gripper_open_angle, gripper_closed_angle)
                        upper_position = max(gripper_open_angle, gripper_closed_angle)
                        next_position = robot.step_follower_gripper_towards(
                            target_position=target_position,
                            step_size=step_size,
                            min_position=lower_position,
                            max_position=upper_position,
                        )
                        _maybe_report_gripper_step(
                            robot,
                            events,
                            gripper_direction,
                            next_position,
                            target_position,
                            gripper_report_interval_s,
                        )
                observation, action = robot.kinesthetic_step(record_data=True)
            else:
                observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            if policy is not None:
                pred_action = predict_action(observation, policy, device, use_amp)
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            add_frame(dataset, observation, action)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            handle_cv2_key_code(cv2.waitKey(1), events, gripper_open_key, gripper_close_key)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    timestamp = 0
    start_vencod_t = time.perf_counter()

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
        while timestamp < reset_time_s:
            time.sleep(1)
            timestamp = time.perf_counter() - start_vencod_t
            pbar.update(1)
            if events["exit_early"]:
                events["exit_early"] = False
                break


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy
    if dataset_name.startswith("eval_") == (policy is None):
        raise ValueError(
            f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
        )
