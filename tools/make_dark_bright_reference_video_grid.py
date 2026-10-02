"""Compose three source videos and their dark/bright-reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 15, dtype=np.uint8)
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


def put_label(image: np.ndarray, text: str, origin: tuple[int, int],
              scale: float = 0.75, color: tuple[int, int, int] = (245, 245, 245),
              thickness: int = 2) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--dark-outputs", nargs=3, required=True)
    parser.add_argument("--bright-outputs", nargs=3, required=True)
    parser.add_argument("--labels", nargs=3, required=True)
    parser.add_argument("--dark-reference", required=True)
    parser.add_argument("--bright-reference", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [*args.inputs, *args.dark_outputs, *args.bright_outputs]
    captures = [cv2.VideoCapture(path) for path in paths]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError("Unable to open one or more input videos")

    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    fps = min(value for value in fps_values if value > 0)
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures)

    dark_reference = cv2.imread(args.dark_reference, cv2.IMREAD_COLOR)
    bright_reference = cv2.imread(args.bright_reference, cv2.IMREAD_COLOR)
    if dark_reference is None or bright_reference is None:
        raise RuntimeError("Unable to open a reference image")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {output}")

    header_height = 210
    row_height = 270
    column_width = 640
    image_width = 444
    image_height = 250
    dark_thumb = fit_image(dark_reference, 170, 170)
    bright_thumb = fit_image(bright_reference, 170, 170)

    try:
        for _ in range(frame_count):
            frames = []
            for cap in captures:
                ok, frame = cap.read()
                if not ok:
                    frames = []
                    break
                frames.append(frame)
            if not frames:
                break

            canvas = np.full((1080, 1920, 3), 12, dtype=np.uint8)
            put_label(canvas, "ONE REFERENCE PERSON, TWO LIGHTING CONDITIONS",
                      (42, 62), 1.15, (255, 255, 255), 2)
            put_label(canvas, "Same reference identity and frame; only lighting changes",
                      (44, 104), 0.68, (190, 205, 220), 1)

            canvas[20:190, 1420:1590] = dark_thumb
            canvas[20:190, 1695:1865] = bright_thumb
            put_label(canvas, "DARK REF", (1431, 205), 0.55, (130, 205, 255), 1)
            put_label(canvas, "BRIGHT REF", (1695, 205), 0.55, (120, 235, 180), 1)

            headers = ["ORIGINAL INPUT", "OUTPUT: DARK REF", "OUTPUT: BRIGHT REF"]
            for column, header in enumerate(headers):
                put_label(canvas, header, (column * column_width + 42, 245),
                          0.66, (245, 215, 80), 2)

            for row in range(3):
                y0 = header_height + row * row_height + 45
                row_frames = [frames[row], frames[3 + row], frames[6 + row]]
                for column, frame in enumerate(row_frames):
                    panel = fit_image(frame, image_width, image_height)
                    x0 = column * column_width + 155
                    canvas[y0:y0 + image_height, x0:x0 + image_width] = panel
                put_label(canvas, args.labels[row], (20, y0 + 34),
                          0.58, (255, 255, 255), 1)
                if row < 2:
                    line_y = y0 + image_height + 4
                    cv2.line(canvas, (20, line_y), (1900, line_y),
                             (55, 55, 55), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()
        for cap in captures:
            cap.release()

    print(f"Saved {output} ({frame_count} frames at {fps:.3f} FPS)")


if __name__ == "__main__":
    main()
