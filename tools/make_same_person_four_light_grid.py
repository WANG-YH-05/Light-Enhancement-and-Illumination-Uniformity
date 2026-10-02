"""Compose four lighting inputs and their fixed-reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 14, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)),
         max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, text: str, xy: tuple[int, int], scale: float,
          color: tuple[int, int, int] = (245, 245, 245), thickness: int = 2) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=4, required=True)
    parser.add_argument("--outputs", nargs=4, required=True)
    parser.add_argument("--labels", nargs=4, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    captures = [cv2.VideoCapture(path) for path in [*args.inputs, *args.outputs]]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError("Unable to open one or more videos")
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures)
    fps = min(cap.get(cv2.CAP_PROP_FPS) for cap in captures)
    reference = cv2.imread(args.reference, cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError("Unable to open reference image")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {destination}")

    reference_thumb = fit_image(reference, 170, 170)
    try:
        for _ in range(frame_count):
            frames: list[np.ndarray] = []
            for capture in captures:
                ok, frame = capture.read()
                if not ok:
                    frames = []
                    break
                frames.append(frame)
            if not frames:
                break

            canvas = np.full((1080, 1920, 3), 12, dtype=np.uint8)
            label(canvas, "SAME PERSON + FOUR INPUT LIGHTS + ONE FIXED REFERENCE",
                  (38, 55), 1.0)
            label(canvas, "All outputs should converge toward the same target lighting",
                  (40, 96), 0.65, (190, 205, 220), 1)
            canvas[18:188, 1715:1885] = reference_thumb
            label(canvas, "FIXED REF", (1735, 207), 0.54, (120, 235, 180), 1)

            for index in range(4):
                grid_row, grid_column = divmod(index, 2)
                x0 = grid_column * 960
                y0 = 215 + grid_row * 430
                label(canvas, args.labels[index].replace("_", " ").upper(),
                      (x0 + 28, y0 + 34), 0.65, (245, 215, 80), 2)
                label(canvas, "INPUT", (x0 + 185, y0 + 70),
                      0.55, (90, 210, 255), 1)
                label(canvas, "OUTPUT", (x0 + 615, y0 + 70),
                      0.55, (110, 235, 170), 1)
                input_panel = fit_image(frames[index], 340, 340)
                output_panel = fit_image(frames[index + 4], 340, 340)
                canvas[y0 + 78:y0 + 418, x0 + 75:x0 + 415] = input_panel
                canvas[y0 + 78:y0 + 418, x0 + 505:x0 + 845] = output_panel
                if grid_column == 0:
                    cv2.line(canvas, (955, y0), (955, y0 + 420),
                             (55, 55, 55), 1, cv2.LINE_AA)
                if grid_row == 0:
                    cv2.line(canvas, (x0 + 15, y0 + 425), (x0 + 945, y0 + 425),
                             (55, 55, 55), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()
        for capture in captures:
            capture.release()

    print(f"Saved {destination} ({frame_count} frames at {fps:.3f} FPS)")


if __name__ == "__main__":
    main()
