"""Create a compact frame contact sheet for visual video diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Unable to read frame {index}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--width", type=int, default=360)
    args = parser.parse_args()
    left = cv2.VideoCapture(args.left)
    right = cv2.VideoCapture(args.right)
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Unable to open one of the videos")
    count = min(int(left.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(right.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = ([int(value) for value in args.frames.split(",") if value.strip()]
               if args.frames else np.linspace(0, count - 1, 8, dtype=int).tolist())
    rows = []
    for index in indices:
        panels = []
        for label, capture in (("INPUT", left), ("OUTPUT", right)):
            frame = read_frame(capture, index)
            height = max(1, round(frame.shape[0] * args.width / frame.shape[1]))
            frame = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"{label}  frame {index}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            panels.append(frame)
        rows.append(cv2.hconcat(panels))
    sheet = cv2.vconcat(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Unable to write {output}")
    left.release()
    right.release()
    print(output.resolve())


if __name__ == "__main__":
    main()
