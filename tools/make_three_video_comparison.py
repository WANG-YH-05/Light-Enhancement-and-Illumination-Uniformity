"""Combine three synchronized videos into a labelled horizontal comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def labelled(frame, label: str, width: int, height: int):
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (width, 54), (0, 0, 0), -1)
    cv2.putText(frame, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--middle", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--left-label", default="Input")
    parser.add_argument("--middle-label", default="Model A")
    parser.add_argument("--right-label", default="Model B")
    parser.add_argument("--panel-width", type=int, default=360)
    args = parser.parse_args()

    captures = [cv2.VideoCapture(path) for path in
                (args.left, args.middle, args.right)]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("Unable to open one or more input videos")

    fps = min(capture.get(cv2.CAP_PROP_FPS) for capture in captures)
    source_width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = args.panel_width
    panel_height = max(1, round(source_height * panel_width / source_width))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (panel_width * 3, panel_height),
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError(f"Unable to open output writer: {output_path}")

    labels = (args.left_label, args.middle_label, args.right_label)
    frame_count = 0
    while True:
        frames = [capture.read() for capture in captures]
        if not all(ok for ok, _ in frames):
            break
        panels = [
            labelled(frame, label, panel_width, panel_height)
            for (_, frame), label in zip(frames, labels)
        ]
        writer.write(cv2.hconcat(panels))
        frame_count += 1

    for capture in captures:
        capture.release()
    writer.release()
    print(f"Wrote {frame_count} frames to {output_path.resolve()}")


if __name__ == "__main__":
    main()
