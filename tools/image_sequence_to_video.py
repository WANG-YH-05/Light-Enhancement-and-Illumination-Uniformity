"""Encode an ordered PNG/JPEG frame directory as an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    frames = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Unable to read {frames[0]}")
    height, width = first.shape[:2]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {output}")

    try:
        for path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Unable to read {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved {output} ({len(frames)} frames at {args.fps:.3f} FPS)")


if __name__ == "__main__":
    main()
