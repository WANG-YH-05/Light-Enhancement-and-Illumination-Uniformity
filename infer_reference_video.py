"""Reference-conditioned FP16 video inference with theta and delta-Y smoothing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.person_mask import MediaPipePersonMasker, composite_person
from rrnet.reference_model import ReferenceRRNet
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.temporal import LightingEMA, ResidualEMA


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead_reference.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True,
                        help="Reference participant image, or a video from which one frame is selected.")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--light-every", type=int, choices=(1, 3, 10), default=3)
    parser.add_argument("--depth-every", type=int, choices=(1, 3, 10), default=3)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--disable-luma-residual", action="store_true",
        help=("Bypass the low-resolution Delta-Y decoder. This preserves the "
              "renderer output and is useful when inspecting hair/detail fidelity."),
    )
    parser.add_argument("--theta-beta", type=float)
    parser.add_argument("--delta-beta", type=float)
    parser.add_argument("--delta-motion-threshold", type=float)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--mask-mode", choices=("person", "none"), default="person")
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

    config = load_config(args.config)
    video_config = config.get("video", {})
    theta_beta = (float(args.theta_beta) if args.theta_beta is not None
                  else float(video_config.get("theta_beta", 0.95)))
    delta_beta = (float(args.delta_beta) if args.delta_beta is not None
                  else float(video_config.get("delta_beta", 0.85)))
    motion_threshold = (
        float(args.delta_motion_threshold) if args.delta_motion_threshold is not None
        else float(video_config.get("delta_motion_threshold", 0.05))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    if model.agm is not None:
        raise ValueError("Cached reference video inference requires use_agm: false.")

    reference_rgb = read_reference(args.reference, args.reference_frame)
    reference_tensor = torch.from_numpy(reference_rgb.astype(np.float32) / 255.0).permute(
        2, 0, 1).unsqueeze(0).to(device)
    reference_mask_tensor = None
    if args.mask_mode == "person":
        reference_masker = make_masker(args)
        reference_mask = reference_masker.segment(reference_rgb, 0)
        reference_masker.close()
        reference_face_mask = face_attention_from_person_mask(reference_mask)
        reference_mask_tensor = torch.from_numpy(reference_face_mask).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        reference_code = model.encode_reference(reference_tensor, reference_mask_tensor).detach()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output writer: {args.output}")

    masker = make_masker(args) if args.mask_mode == "person" else None
    theta_ema = LightingEMA(theta_beta)
    delta_ema = ResidualEMA(delta_beta, motion_threshold)
    theta = depth = illumination = None
    frame_index = 0
    print(
        f"Reference video inference: {width}x{height}, frames={total_frames}, "
        f"L{args.light_every}/D{args.depth_every}, theta_beta={theta_beta:.2f}, "
        f"delta_beta={delta_beta:.2f}, precision={args.precision}", flush=True)
    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(
                2, 0, 1).unsqueeze(0).to(device)
            update_light = theta is None or frame_index % args.light_every == 0
            update_depth = depth is None or frame_index % args.depth_every == 0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                if update_light:
                    theta = theta_ema.update(model.base.estimate_lighting(image)["theta"])
                if update_depth:
                    depth = model.depth(image).detach()
                if illumination is None or update_light or update_depth:
                    illumination, _ = model.renderer.illumination(depth, theta)
                    illumination = model.sanitize_illumination(illumination).detach()
                canonical = model.renderer.apply_illumination(image, illumination)
                base_prediction = {"output": canonical, "depth": depth, "theta": theta}
                if args.disable_luma_residual:
                    # A dense, quarter-resolution additive lift can erase texture in
                    # near-black hair.  Keep this ablation at the renderer output so
                    # it is clear whether the artifact is caused by Delta-Y rather
                    # than by masking, depth, or the reference image.
                    result = canonical
                else:
                    current_delta = model.decoder(image, canonical, depth, reference_code)
                    smooth_delta = delta_ema.update(current_delta, luminance(image))
                    result = model.apply_luma_residual(
                        image, base_prediction, smooth_delta, reference_code)["output"]
            output = np.uint8(
                result[0].permute(1, 2, 0).float().cpu().numpy().clip(0, 1) * 255.0 + 0.5)
            if masker is not None:
                timestamp_ms = int(round(1000.0 * frame_index / max(fps, 1.0e-6)))
                person_mask = masker.segment(rgb_u8, timestamp_ms)
                output = composite_person(rgb_u8, output, person_mask)
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            frame_index += 1
            if args.progress_every and (frame_index % args.progress_every == 0
                                        or frame_index == total_frames):
                print(f"Processed {frame_index}/{total_frames} frames", flush=True)
    capture.release()
    writer.release()
    if masker is not None:
        masker.close()
    print(f"Saved: {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
