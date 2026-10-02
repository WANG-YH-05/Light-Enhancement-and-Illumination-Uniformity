"""Person-only global-luminance baseline for reference-light evaluation.

This deliberately contains no learned model, depth, virtual lights, or local
gain map. It simply matches the input face-region mean luminance to a selected
reference face and composites the uniformly scaled person back on the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.person_mask import MediaPipePersonMasker, composite_person
from rrnet.reference_mask import face_attention_from_person_mask


def read_reference(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read reference image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def luminance(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2])


def masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    weights = np.clip(mask[..., 0], 0.0, 1.0)
    return float((value * weights).sum() / max(float(weights.sum()), 1.0e-6))


def make_masker(model_path: str, beta: float) -> MediaPipePersonMasker:
    return MediaPipePersonMasker(
        model_path,
        threshold_low=0.30,
        threshold_high=0.70,
        close_radius=2,
        dilate_radius=0,
        feather=1.5,
        temporal_beta=beta,
        guided_radius=5,
        guided_epsilon=1.0e-3,
        edge_power=1.5,
        flow_width=256,
        boundary_fade=8.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--person-model",
                        default="third_party/mediapipe/models/selfie_segmenter.tflite")
    parser.add_argument("--gain-min", type=float, default=0.40)
    parser.add_argument("--gain-max", type=float, default=5.00)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--progress-every", type=int, default=30)
    args = parser.parse_args()
    if not 0.0 < args.gain_min <= 1.0 <= args.gain_max:
        raise ValueError("Expected gain_min in (0,1] and gain_max >= 1")

    reference_rgb = read_reference(args.reference)
    reference_masker = make_masker(args.person_model, beta=0.0)
    reference_person = reference_masker.segment(reference_rgb, 0)
    reference_masker.close()
    reference_face = face_attention_from_person_mask(reference_person)
    reference_luma = masked_mean(luminance(reference_rgb), reference_face)

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output writer: {output}")

    masker = make_masker(args.person_model, beta=args.beta)
    frame_index = 0
    gain_ema: float | None = None
    print(
        f"Global-luminance baseline: {width}x{height}, frames={total_frames}, "
        f"reference_face_Y={reference_luma:.3f}", flush=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round(1000.0 * frame_index / max(fps, 1.0e-6)))
            person_mask = masker.segment(source, timestamp_ms)
            face_mask = face_attention_from_person_mask(person_mask)
            source_luma = masked_mean(luminance(source), face_mask)
            instant_gain = float(np.clip(
                reference_luma / max(source_luma, 1.0e-4),
                args.gain_min, args.gain_max))
            gain_ema = (instant_gain if gain_ema is None
                        else args.beta * gain_ema + (1.0 - args.beta) * instant_gain)
            enhanced = np.clip(
                source.astype(np.float32) * gain_ema, 0.0, 255.0).astype(np.uint8)
            result = composite_person(source, enhanced, person_mask)
            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            frame_index += 1
            if args.progress_every and (
                    frame_index % args.progress_every == 0
                    or frame_index == total_frames):
                print(f"Processed {frame_index}/{total_frames}", flush=True)
    finally:
        capture.release()
        writer.release()
        masker.close()
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
