"""Blend an enhanced video into only the tracked face region.

This is a lightweight proof-of-concept tool.  It uses OpenCV's frontal-face
detector, interpolates missed detections, temporally smooths the face box, and
builds a feathered elliptical mask.  The original background is copied exactly
outside that mask.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original input video")
    parser.add_argument("--enhanced", required=True, help="RRNet enhanced video")
    parser.add_argument("--output", required=True, help="Face-only relighted video")
    parser.add_argument("--compare-output", help="Optional input/full/face-only comparison video")
    parser.add_argument("--detect-width", type=int, default=360)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--smooth-radius", type=int, default=5)
    parser.add_argument("--feather", type=float, default=10.0)
    parser.add_argument("--expand-x", type=float, default=1.02)
    parser.add_argument("--expand-y", type=float, default=1.18)
    return parser.parse_args()


def video_info(path: str) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return width, height, fps, frames


def detect_boxes(path: str, detect_width: int,
                 detect_every: int) -> tuple[np.ndarray, float]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Unable to load face detector: {cascade_path}")
    capture = cv2.VideoCapture(path)
    boxes: list[list[float]] = []
    detection_seconds = 0.0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % detect_every != 0:
            boxes.append([np.nan] * 4)
            frame_index += 1
            continue
        height, width = frame.shape[:2]
        scale = min(1.0, detect_width / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60))
        detection_seconds += time.perf_counter() - started
        if len(faces):
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            boxes.append([x / scale, y / scale, w / scale, h / scale])
        else:
            boxes.append([np.nan] * 4)
        frame_index += 1
    capture.release()
    result = np.asarray(boxes, dtype=np.float32)
    if result.size == 0 or np.isnan(result[:, 0]).all():
        raise RuntimeError("No face was detected in the input video.")
    return result, detection_seconds


def interpolate_and_smooth(boxes: np.ndarray, radius: int) -> np.ndarray:
    indices = np.arange(len(boxes), dtype=np.float32)
    result = boxes.copy()
    for column in range(4):
        valid = ~np.isnan(result[:, column])
        result[:, column] = np.interp(indices, indices[valid], result[valid, column])
    if radius > 0:
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        sigma = max(radius / 2.0, 1.0)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        for column in range(4):
            padded = np.pad(result[:, column], (radius, radius), mode="edge")
            result[:, column] = np.convolve(padded, kernel, mode="valid")
    return result


def face_mask_roi(shape: tuple[int, int], box: np.ndarray, expand_x: float,
                  expand_y: float, feather: float) -> tuple[int, int, int, int, np.ndarray]:
    height, width = shape
    x, y, w, h = box
    center = (int(round(x + 0.5 * w)), int(round(y + 0.52 * h)))
    axes = (max(1, int(round(0.5 * w * expand_x))),
            max(1, int(round(0.5 * h * expand_y))))
    padding = int(round(3.0 * feather))
    x0 = max(0, center[0] - axes[0] - padding)
    y0 = max(0, center[1] - axes[1] - padding)
    x1 = min(width, center[0] + axes[0] + padding + 1)
    y1 = min(height, center[1] + axes[1] + padding + 1)
    local_center = (center[0] - x0, center[1] - y0)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    cv2.ellipse(mask, local_center, axes, 0.0, 0.0, 360.0, 1.0, thickness=-1,
                lineType=cv2.LINE_AA)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    return x0, y0, x1, y1, np.clip(mask, 0.0, 1.0)[..., None]


def label_tile(frame: np.ndarray, label: str, width: int = 360) -> np.ndarray:
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    tile = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (width, 36), (0, 0, 0), thickness=-1)
    cv2.putText(tile, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    args = parse_args()
    input_info = video_info(args.input)
    enhanced_info = video_info(args.enhanced)
    if input_info[:2] != enhanced_info[:2]:
        raise ValueError(f"Resolution mismatch: input={input_info[:2]}, enhanced={enhanced_info[:2]}")
    width, height, fps, _ = input_info
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.compare_output:
        Path(args.compare_output).parent.mkdir(parents=True, exist_ok=True)

    if args.detect_every < 1:
        raise ValueError("--detect-every must be at least 1.")
    raw_boxes, detection_seconds = detect_boxes(args.input, args.detect_width,
                                                 args.detect_every)
    detected = int(np.isfinite(raw_boxes[:, 0]).sum())
    boxes = interpolate_and_smooth(raw_boxes, args.smooth_radius)

    source = cv2.VideoCapture(args.input)
    enhanced = cv2.VideoCapture(args.enhanced)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    compare_writer = None
    if args.compare_output:
        tile_width = 360
        tile_height = int(round(height * tile_width / width))
        compare_writer = cv2.VideoWriter(args.compare_output,
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (tile_width * 3, tile_height))
    if not writer.isOpened() or (compare_writer is not None and not compare_writer.isOpened()):
        raise RuntimeError("Unable to open an output video writer.")

    blend_seconds = 0.0
    frame_index = 0
    while frame_index < len(boxes):
        ok_input, input_frame = source.read()
        ok_enhanced, enhanced_frame = enhanced.read()
        if not ok_input or not ok_enhanced:
            break
        started = time.perf_counter()
        x0, y0, x1, y1, mask = face_mask_roi(
            (height, width), boxes[frame_index], args.expand_x,
            args.expand_y, args.feather)
        blended = input_frame.copy()
        input_roi = input_frame[y0:y1, x0:x1].astype(np.float32)
        enhanced_roi = enhanced_frame[y0:y1, x0:x1].astype(np.float32)
        blended_roi = enhanced_roi * mask + input_roi * (1.0 - mask)
        blended[y0:y1, x0:x1] = np.uint8(np.clip(blended_roi + 0.5, 0, 255))
        blend_seconds += time.perf_counter() - started
        writer.write(blended)
        if compare_writer is not None:
            compare = np.hstack((label_tile(input_frame, "INPUT"),
                                 label_tile(enhanced_frame, "FULL RRNET"),
                                 label_tile(blended, "FACE ONLY")))
            compare_writer.write(compare)
        frame_index += 1

    source.release()
    enhanced.release()
    writer.release()
    if compare_writer is not None:
        compare_writer.release()
    detection_ms = 1000.0 * detection_seconds / max(frame_index, 1)
    blend_ms = 1000.0 * blend_seconds / max(frame_index, 1)
    print(f"Processed frames: {frame_index}")
    print(f"Face detections: {detected}/{len(raw_boxes)}")
    print(f"Pure detection time: {detection_ms:.3f} ms/frame")
    print(f"Pure mask+blend time: {blend_ms:.3f} ms/frame")
    print(f"Output: {Path(args.output).resolve()}")
    if args.compare_output:
        print(f"Comparison: {Path(args.compare_output).resolve()}")


if __name__ == "__main__":
    main()
