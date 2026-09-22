"""Measure camera-frame-to-output latency for reference-relative RRNet.

Video decoding/encoding, disk I/O, metrics, logging and one-time reference
encoding are deliberately outside the timed region.  The timed region mirrors
the required per-frame algorithm in infer_reference_relative_video.py:

* BGR -> RGB conversion and CPU -> CUDA transfer
* MediaPipe person segmentation and all mask refinement/temporal processing
* face-attention mask construction and transfer to CUDA
* scheduled lighting/depth inference and illumination rendering
* per-frame transfer gain and RGB relighting
* CUDA -> CPU transfer and person/background compositing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_ROOT = Path(__file__).resolve().parents[2]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from infer_reference_relative_video import make_masker, read_reference, to_tensor
from rrnet.config import load_config, model_kwargs
from rrnet.person_mask import CudaPersonMasker, composite_person, composite_person_tensor
from rrnet.reference_mask import (
    face_attention_from_person_mask,
    face_attention_from_person_mask_tensor,
)
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.temporal import LightingEMA


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "fps_from_mean": 1000.0 / mean,
    }


def load_frames(path: str, count: int, width: int, height: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(np.ascontiguousarray(frame))
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"Needed {count} frames but loaded {len(frames)}.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--person-model", default="third_party/mediapipe/models/selfie_segmenter.tflite")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--light-every", type=int, default=10)
    parser.add_argument("--depth-every", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mask-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--mask-every", type=int, default=3)
    parser.add_argument("--mask-work-width", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile-components", action="store_true",
        help="Synchronize between stages and report a diagnostic latency breakdown.",
    )
    parser.add_argument(
        "--record-components", action="store_true",
        help="Record natural-pipeline stage timings without inserting extra synchronization.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FP16 benchmark.")
    if args.measured_frames < 30:
        raise ValueError("Use at least 30 measured frames for a full L10/D3 cycle.")

    config = load_config(args.config)
    if config.get("task", "").lower() != "reference_relative":
        raise ValueError("The config must use task: reference_relative.")
    device = torch.device("cuda")
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    # Reference work is a one-time meeting setup cost and is intentionally not timed.
    mask_args = argparse.Namespace(
        person_model=args.person_model,
        mask_threshold_low=0.30,
        mask_threshold_high=0.70,
        mask_close_radius=2,
        mask_dilate_radius=0,
        mask_feather=1.5,
        mask_beta=0.80,
        mask_guided_radius=5,
        mask_guided_epsilon=1.0e-3,
        mask_edge_power=1.5,
        mask_flow_width=256,
        mask_boundary_fade=8.0,
    )
    reference_rgb = read_reference(args.reference, 0)
    reference_masker = make_masker(mask_args)
    reference_person_mask = reference_masker.segment(reference_rgb, 0)
    reference_masker.close()
    reference_face_mask = face_attention_from_person_mask(reference_person_mask)
    reference_tensor = to_tensor(reference_rgb, device)
    reference_mask_tensor = torch.from_numpy(reference_face_mask).permute(
        2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        reference_theta = model.encode_reference(
            reference_tensor, reference_mask_tensor)["theta"].detach()
    torch.cuda.synchronize()

    # Decode and resolution conversion are setup-only. The timer begins with a
    # camera-style BGR uint8 frame already resident in CPU memory.
    frame_pool = load_frames(args.input, 30, args.width, args.height)
    if args.mask_backend == "gpu":
        masker = CudaPersonMasker(
            args.person_model,
            device=device,
            mask_every=args.mask_every,
            work_width=args.mask_work_width,
            threshold_low=mask_args.mask_threshold_low,
            threshold_high=mask_args.mask_threshold_high,
            close_radius=mask_args.mask_close_radius,
            dilate_radius=mask_args.mask_dilate_radius,
            feather=mask_args.mask_feather,
            temporal_beta=mask_args.mask_beta,
            guided_radius=mask_args.mask_guided_radius,
            guided_epsilon=mask_args.mask_guided_epsilon,
            edge_power=mask_args.mask_edge_power,
            boundary_fade=mask_args.mask_boundary_fade,
        )
    else:
        masker = make_masker(mask_args)
    theta_ema = LightingEMA(float(config.get("video", {}).get("theta_beta", 0.95)))
    source_theta = depth = source_light = reference_light = None
    records: list[tuple[str, float]] = []
    component_records: dict[str, list[float]] = {
        "bgr_to_rgb": [],
        "input_to_gpu": [],
        "person_mask": [],
        "face_mask_and_to_gpu": [],
        "rrnet": [],
        "output_to_cpu": [],
        "person_composite": [],
    }
    total_frames = args.warmup_frames + args.measured_frames

    with torch.inference_mode():
        for frame_index in range(total_frames):
            frame = frame_pool[frame_index % len(frame_pool)]
            update_light = source_theta is None or frame_index % args.light_every == 0
            update_depth = depth is None or frame_index % args.depth_every == 0

            torch.cuda.synchronize()
            start = time.perf_counter()

            rgb_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stage_rgb = time.perf_counter()
            image = to_tensor(rgb_u8, device)
            if args.profile_components:
                torch.cuda.synchronize()
            stage_input = time.perf_counter()
            timestamp_ms = int(round(1000.0 * frame_index / args.fps))
            person_mask = None
            person_mask_tensor = None
            if args.mask_backend == "gpu":
                person_mask_tensor = masker.segment(
                    rgb_u8, timestamp_ms, frame_index, image)
            else:
                person_mask = masker.segment(rgb_u8, timestamp_ms)
            stage_person_mask = time.perf_counter()
            source_mask_tensor = None
            if update_light:
                if person_mask_tensor is not None:
                    source_mask_tensor = face_attention_from_person_mask_tensor(
                        person_mask_tensor, work_width=args.mask_work_width)
                else:
                    source_face_mask = face_attention_from_person_mask(person_mask)
                    source_mask_tensor = torch.from_numpy(source_face_mask).permute(
                        2, 0, 1).unsqueeze(0).to(device)
            if args.profile_components:
                torch.cuda.synchronize()
            stage_face_mask = time.perf_counter()

            with torch.autocast("cuda", dtype=torch.float16):
                if update_light:
                    prediction = model.estimate_light(image, source_mask_tensor)
                    source_theta = theta_ema.update(prediction["theta"])
                if update_depth:
                    depth = model.depth(image).detach()
                if source_light is None or reference_light is None or update_light or update_depth:
                    source_light, _ = model.illumination_from_theta(depth, source_theta)
                    reference_light, _ = model.illumination_from_theta(depth, reference_theta)
                gain = model.compute_transfer_gain(image, source_light, reference_light).detach()
                result = (image * gain).clamp(0.0, 1.0)
                if person_mask_tensor is not None:
                    result = composite_person_tensor(image, result, person_mask_tensor)
            if args.profile_components:
                torch.cuda.synchronize()
            stage_rrnet = time.perf_counter()

            output = (
                (result[0].clamp(0.0, 1.0) * 255.0 + 0.5)
                .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            )
            stage_output = time.perf_counter()
            if person_mask is not None:
                composite_person(rgb_u8, output, person_mask)
            torch.cuda.synchronize()
            stage_composite = time.perf_counter()

            elapsed_ms = (stage_composite - start) * 1000.0
            if frame_index >= args.warmup_frames:
                if update_light and update_depth:
                    kind = "light_and_depth_update"
                elif update_light:
                    kind = "light_update_only"
                elif update_depth:
                    kind = "depth_update_only"
                else:
                    kind = "cached"
                records.append((kind, elapsed_ms))
                if args.profile_components or args.record_components:
                    component_records["bgr_to_rgb"].append((stage_rgb - start) * 1000.0)
                    component_records["input_to_gpu"].append((stage_input - stage_rgb) * 1000.0)
                    component_records["person_mask"].append(
                        (stage_person_mask - stage_input) * 1000.0)
                    component_records["face_mask_and_to_gpu"].append(
                        (stage_face_mask - stage_person_mask) * 1000.0)
                    component_records["rrnet"].append(
                        (stage_rrnet - stage_face_mask) * 1000.0)
                    component_records["output_to_cpu"].append(
                        (stage_output - stage_rrnet) * 1000.0)
                    component_records["person_composite"].append(
                        (stage_composite - stage_output) * 1000.0)

    masker.close()
    groups: dict[str, list[float]] = {}
    for kind, elapsed_ms in records:
        groups.setdefault(kind, []).append(elapsed_ms)
    results = {
        "overall_l10_d3": summarize([value for _, value in records]),
        **{kind: summarize(values) for kind, values in groups.items()},
    }
    component_results = (
        {name: summarize(values) for name, values in component_records.items()}
        if (args.profile_components or args.record_components) else {}
    )
    payload = {
        "scope": {
            "included": [
                "BGR-to-RGB", "scheduled MediaPipe person inference and mask processing",
                "CPU-to-GPU transfers", "RRNet FP16 L10/D3 inference",
                "GPU-to-CPU transfer", "person/background compositing",
            ],
            "excluded": [
                "video decode", "video encode", "disk I/O", "metrics", "logging",
                "model initialization", "one-time reference encoding",
            ],
        },
        "system": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "resolution": [args.width, args.height],
            "precision": "fp16",
            "light_every": args.light_every,
            "depth_every": args.depth_every,
            "mask_backend": args.mask_backend,
            "mask_every": args.mask_every,
            "mask_work_width": args.mask_work_width,
            "warmup_frames": args.warmup_frames,
            "measured_frames": args.measured_frames,
        },
        "results": results,
        "component_profile": component_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "realtime_latency_results.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, stats in results.items():
        print(
            f"{name}: n={stats['count']} mean={stats['mean_ms']:.3f} ms "
            f"median={stats['median_ms']:.3f} ms p95={stats['p95_ms']:.3f} ms "
            f"fps={stats['fps_from_mean']:.2f}")
    if component_results:
        mode = ("diagnostic synchronization enabled" if args.profile_components
                else "natural pipeline; asynchronous GPU work may finish in output_to_cpu")
        print(f"Component profile ({mode}):")
        for name, stats in component_results.items():
            print(
                f"  {name}: mean={stats['mean_ms']:.3f} ms "
                f"median={stats['median_ms']:.3f} ms p95={stats['p95_ms']:.3f} ms")
    print(f"Saved: {output_file.resolve()}")


if __name__ == "__main__":
    main()
