#!/usr/bin/env python

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Send a gripper command to a running kinesthetic recorder.")
    parser.add_argument("command", choices=["open", "close", "o", "c"])
    parser.add_argument(
        "--command-file",
        default=".cache/gripper_command.txt",
        help="Command file watched by control_robot.py during kinesthetic recording.",
    )
    args = parser.parse_args()

    command = {"o": "open", "c": "close"}.get(args.command, args.command)
    command_file = Path(args.command_file)
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(command, encoding="utf-8")
    print(f"Wrote gripper command: {command}")


if __name__ == "__main__":
    main()
