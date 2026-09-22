"""Video inference for RRNet relative illumination transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rrnet.config import load_config, model_kwargs
from rrnet.person_mask import (
    CudaPersonMasker,
    MediaPipePersonMasker,
    composite_person,
    composite_person_tensor,
)
from rrnet.reference_mask import (
    canonical_face_light_roi,
    face_attention_from_person_mask,
    face_attention_from_person_mask_tensor,
)
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.temporal import LightingEMA


def read_reference(path: str, frame_index: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open reference image/video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read reference frame {frame_index}: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_masker(args: argparse.Namespace) -> MediaPipePersonMasker:
    return MediaPipePersonMasker(
        args.person_model,
        threshold_low=args.mask_threshold_low,
        threshold_high=args.mask_threshold_high,
        close_radius=args.mask_close_radius,
        dilate_radius=args.mask_dilate_radius,
        feather=args.mask_feather,
        temporal_beta=args.mask_beta,
        guided_radius=args.mask_guided_radius,
        guided_epsilon=args.mask_guided_epsilon,
        edge_power=args.mask_edge_power,
        flow_width=args.mask_flow_width,
        boundary_fade=args.mask_boundary_fade,
    )


def to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(
        2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead_reference_relative.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--light-every", type=int, choices=(1, 3, 10), default=3)
    parser.add_argument("--depth-every", type=int, choices=(1, 3, 10), default=3)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--theta-beta", type=float)
    parser.add_argument(
        "--max-transfer-gain", type=float,
        help="Inference-only override for the configured maximum relative-light gain.",
    )
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--mask-mode", choices=("person", "none"), default="person")
    parser.add_argument(
        "--mask-backend", choices=("gpu", "cpu"), default="gpu",
        help="gpu moves mask refinement, temporal smoothing and compositing to CUDA.",
    )
    parser.add_argument(
        "--mask-every", type=int, default=3,
        help="Run MediaPipe every N frames; cached CUDA masks are reused between updates.",
    )
    parser.add_argument(
        "--mask-work-width", type=int, default=512,
        help="Working width for CUDA mask refinement before full-resolution upsampling.",
    )
    parser.add_argument(
        "--light-input", choices=("full", "face_roi"), default="full",
        help=("Input geometry used only by LPRM. face_roi canonicalizes the "
              "face scale for source/reference light estimation."),
    )
    parser.add_argument(
        "--light-roi-context", type=float, default=0.30,
        help="Extra square context around the heuristic face ROI in face_roi mode.",
    )
    parser.add_argument("--person-model",
                        default="third_party/mediapipe/models/selfie_segmenter.tflite")
    parser.add_argument("--mask-threshold-low", type=float, default=0.30)
    parser.add_argument("--mask-threshold-high", type=float, default=0.70)
    parser.add_argument("--mask-close-radius", type=int, default=2)
    parser.add_argument("--mask-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-feather", type=float, default=1.5)
    parser.add_argument("--mask-beta", type=float, default=0.80)
    parser.add_argument("--mask-flow-width", type=int, default=256)
    parser.add_argument("--mask-boundary-fade", type=float, default=8.0)
    parser.add_argument("--mask-guided-radius", type=int, default=5)
    parser.add_argument("--mask-guided-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--mask-edge-power", type=float, default=1.5)
    args = parser.parse_args()

    if args.light_input == "face_roi" and args.mask_mode != "person":
        raise ValueError("--light-input face_roi requires --mask-mode person")
    if args.mask_every < 1:
        raise ValueError("--mask-every must be at least 1")
    if args.mask_backend == "gpu" and args.light_input == "face_roi":
        raise ValueError(
            "CUDA masking currently supports --light-input full; use full or "
            "select --mask-backend cpu for face_roi.")

    config = load_config(args.config)
    if config.get("task", "").lower() != "reference_relative":
        raise ValueError("This script requires task: reference_relative")
    theta_beta = (float(args.theta_beta) if args.theta_beta is not None
                  else float(config.get("video", {}).get("theta_beta", 0.95)))
    kwargs = model_kwargs(config, args.config)
    if args.max_transfer_gain is not None:
        if args.max_transfer_gain < 1.0:
            raise ValueError("--max-transfer-gain must be at least 1.0")
        kwargs["max_transfer_gain"] = float(args.max_transfer_gain)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**kwargs).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    reference_rgb = read_reference(args.reference, args.reference_frame)
    reference_mask_tensor = None
    if args.mask_mode == "person":
        reference_masker = make_masker(args)
        reference_person_mask = reference_masker.segment(reference_rgb, 0)
        reference_masker.close()
        reference_face_mask = face_attention_from_person_mask(reference_person_mask)
        reference_light_rgb = reference_rgb
        if args.light_input == "face_roi":
            reference_light_rgb, reference_face_mask = canonical_face_light_roi(
                reference_rgb, reference_person_mask, context=args.light_roi_context)
        reference_tensor = to_tensor(reference_light_rgb, device)
        reference_mask_tensor = torch.from_numpy(reference_face_mask).permute(
            2, 0, 1).unsqueeze(0).to(device)
    else:
        reference_tensor = to_tensor(reference_rgb, device)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        reference_theta = model.encode_reference(
            reference_tensor, reference_mask_tensor)["theta"].detach()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open output writer: {output_path}")

    masker = None
    if args.mask_mode == "person":
        if args.mask_backend == "gpu":
            if device.type != "cuda":
                raise RuntimeError("--mask-backend gpu requires CUDA.")
            masker = CudaPersonMasker(
                args.person_model,
                device=device,
                mask_every=args.mask_every,
                work_width=args.mask_work_width,
                threshold_low=args.mask_threshold_low,
                threshold_high=args.mask_threshold_high,
                close_radius=args.mask_close_radius,
                dilate_radius=args.mask_dilate_radius,
                feather=args.mask_feather,
                temporal_beta=args.mask_beta,
                guided_radius=args.mask_guided_radius,
                guided_epsilon=args.mask_guided_epsilon,
                edge_power=args.mask_edge_power,
                boundary_fade=args.mask_boundary_fade,
            )
        else:
            masker = make_masker(args)
    source_theta_ema = LightingEMA(theta_beta)
    source_theta = depth = source_light = reference_light = None
    frame_index = 0
    print(
        f"Relative-light inference: {width}x{height}, frames={total_frames}, "
        f"L{args.light_every}/D{args.depth_every}, theta_beta={theta_beta:.2f}, "
        f"max_gain={model.max_transfer_gain:.2f}, precision={args.precision}, "
        f"light_input={args.light_input}, mask_backend={args.mask_backend}, "
        f"mask_every={args.mask_every}",
        flush=True)

    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = to_tensor(rgb_u8, device)
            person_mask = None
            person_mask_tensor = None
            source_light_mask_tensor = None
            source_light_image = image
            update_light = source_theta is None or frame_index % args.light_every == 0
            update_depth = depth is None or frame_index % args.depth_every == 0
            if masker is not None:
                timestamp_ms = int(round(
                    1000.0 * frame_index / max(fps, 1.0e-6)))
                if args.mask_backend == "gpu":
                    person_mask_tensor = masker.segment(
                        rgb_u8, timestamp_ms, frame_index, image)
                    if update_light:
                        source_light_mask_tensor = face_attention_from_person_mask_tensor(
                            person_mask_tensor, work_width=args.mask_work_width)
                else:
                    person_mask = masker.segment(rgb_u8, timestamp_ms)
                    # This attention mask is consumed only when LPRM runs.
                    if update_light:
                        source_face_mask = face_attention_from_person_mask(person_mask)
                        if args.light_input == "face_roi":
                            source_light_rgb, source_face_mask = canonical_face_light_roi(
                                rgb_u8, person_mask, context=args.light_roi_context)
                            source_light_image = to_tensor(source_light_rgb, device)
                        source_light_mask_tensor = torch.from_numpy(
                            source_face_mask).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                if update_light:
                    source_prediction = model.estimate_light(
                        source_light_image, source_light_mask_tensor)
                    source_theta = source_theta_ema.update(
                        source_prediction["theta"])
                if update_depth:
                    depth = model.depth(image).detach()
                if source_light is None or reference_light is None or update_light or update_depth:
                    source_light, _ = model.illumination_from_theta(
                        depth, source_theta)
                    reference_light, _ = model.illumination_from_theta(
                        depth, reference_theta)
                # The expensive illumination maps follow the configured L/D
                # cadence, but the clipping-safe cap depends on the current
                # RGB frame and must be refreshed every frame.  Otherwise a
                # brighter in-between frame can clip one channel and shift hue.
                transfer_gain = model.compute_transfer_gain(
                    image, source_light, reference_light).detach()
                result = (image * transfer_gain).clamp(0.0, 1.0)
                if person_mask_tensor is not None:
                    result = composite_person_tensor(image, result, person_mask_tensor)

            output = (
                (result[0].clamp(0.0, 1.0) * 255.0 + 0.5)
                .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            )
            if person_mask is not None:
                output = composite_person(rgb_u8, output, person_mask)
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            frame_index += 1
            if args.progress_every and (
                    frame_index % args.progress_every == 0
                    or frame_index == total_frames):
                print(f"Processed {frame_index}/{total_frames} frames", flush=True)

    capture.release()
    writer.release()
    if masker is not None:
        masker.close()
    print(f"Saved: {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
