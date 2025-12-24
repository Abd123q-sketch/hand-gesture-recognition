import argparse
from pathlib import Path
import subprocess
import sys

from hand_gesture_recognition.utils.config import PROJECT_ROOT


def run(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition Entry Point")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train a model")
    tr.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args for training")

    ev = sub.add_parser("evaluate", help="Evaluate a model")
    ev.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args for evaluation")

    rt = sub.add_parser("realtime", help="Run realtime inference")
    rt.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args for realtime")

    args = parser.parse_args()

    if args.command == "train":
        run("hand_gesture_recognition.training.train", args.extra or [])
    elif args.command == "evaluate":
        run("hand_gesture_recognition.training.evaluate", args.extra or [])
    elif args.command == "realtime":
        run("hand_gesture_recognition.inference.realtime_inference", args.extra or [])


if __name__ == "__main__":
    main()
