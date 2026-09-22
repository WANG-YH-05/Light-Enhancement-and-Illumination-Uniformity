"""Create a labelled side-by-side comparison MP4 from two videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--left-label", default="Input")
    parser.add_argument("--right-label", default="RRNet output")
    args = parser.parse_args()

    left = cv2.VideoCapture(args.left)
    right = cv2.VideoCapture(args.right)
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Unable to open one of the input videos.")
    fps = left.get(cv2.CAP_PROP_FPS)
    width, height = int(left.get(cv2.CAP_PROP_FRAME_WIDTH)), int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    index = 0
    while True:
        ok_left, left_frame = left.read()
        ok_right, right_frame = right.read()
        if not ok_left or not ok_right:
            break
        if right_frame.shape[:2] != (height, width):
            right_frame = cv2.resize(right_frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(left_frame, args.left_label, (28, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 255, 255), 3)
        cv2.putText(right_frame, args.right_label, (28, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 255, 255), 3)
        writer.write(cv2.hconcat((left_frame, right_frame)))
        index += 1
    left.release()
    right.release()
    writer.release()
    print(f"Wrote {index} frames to {output}")


if __name__ == "__main__":
    main()
