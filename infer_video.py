"""FP16 video inference with independent lighting/depth/illumination caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rrnet.config import load_config, model_kwargs
from rrnet.model import RRNet
from rrnet.person_mask import MediaPipePersonMasker, composite_person
from rrnet.temporal import LightingEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_video.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--light-every", type=int, choices=(1, 3, 10))
    parser.add_argument("--depth-every", type=int, choices=(1, 3, 10), default=1)
    parser.add_argument("--estimate-every", type=int, choices=(1, 3, 10),
                        help="Deprecated alias for --light-every")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--mask-mode", choices=("person", "none"), default="person",
                        help="Keep the original background with a full-person mask (default).")
    parser.add_argument("--person-model",
                        default="third_party/mediapipe/models/selfie_segmenter.tflite")
    parser.add_argument("--mask-threshold-low", type=float, default=0.30)
    parser.add_argument("--mask-threshold-high", type=float, default=0.70)
    parser.add_argument("--mask-close-radius", type=int, default=2)
    parser.add_argument("--mask-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-feather", type=float, default=1.5)
    parser.add_argument("--mask-beta", type=float, default=0.80,
                        help="EMA weight after optical-flow mask alignment.")
    parser.add_argument("--mask-flow-width", type=int, default=256)
    parser.add_argument("--mask-boundary-fade", type=float, default=8.0,
                        help="Pixels over which relighting fades to identity at the silhouette.")
    parser.add_argument("--mask-guided-radius", type=int, default=5)
    parser.add_argument("--mask-guided-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--mask-edge-power", type=float, default=1.5)
    parser.add_argument("--mask-output", help="Optional debug video showing the person mask.")
    args = parser.parse_args()
    if args.light_every is not None and args.estimate_every is not None:
        parser.error("Use --light-every or --estimate-every, not both.")
    light_every = args.light_every or args.estimate_every or 3
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mask_output:
        Path(args.mask_output).parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRNet(**model_kwargs(config, args.config)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()
    if model.agm is not None:
        raise ValueError("Cached video inference requires a configuration with use_agm: false.")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    ema = LightingEMA(args.beta)
    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    mask_writer = None
    if args.mask_output:
        mask_writer = cv2.VideoWriter(args.mask_output, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (width, height))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {args.input}")
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video writer: {args.output}")
    if mask_writer is not None and not mask_writer.isOpened():
        raise RuntimeError(f"Unable to open mask video writer: {args.mask_output}")
    masker = None
    if args.mask_mode == "person":
        masker = MediaPipePersonMasker(
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
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index, theta, depth, normals, illumination = 0, None, None, None, None
    print(
        f"Video inference: {width}x{height}, frames={total_frames}, precision={args.precision}, "
        f"light_every={light_every}, depth_every={args.depth_every}, mask_mode={args.mask_mode}",
        flush=True,
    )
    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = rgb_u8.astype(np.float32) / 255.0
            image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            update_light = theta is None or frame_index % light_every == 0
            update_depth = depth is None or frame_index % args.depth_every == 0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                if update_light:
                    theta = ema.update(model.estimate_lighting(image)["theta"])
                if update_depth:
                    depth = model.depth(image)
                if illumination is None or update_light or update_depth:
                    illumination, normals = model.renderer.illumination(depth, theta)
                    illumination = illumination.detach()
                    normals = normals.detach()
                result = model.renderer.apply_illumination(image, illumination)
            output = np.uint8(
                result[0].permute(1, 2, 0).float().cpu().numpy().clip(0, 1) * 255.0 + 0.5
            )
            if masker is not None:
                timestamp_ms = int(round(1000.0 * frame_index / max(fps, 1.0e-6)))
                person_mask = masker.segment(rgb_u8, timestamp_ms)
                output = composite_person(rgb_u8, output, person_mask)
                if mask_writer is not None:
                    mask_frame = np.uint8(np.clip(person_mask * 255.0 + 0.5, 0, 255))
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                    mask_writer.write(mask_frame)
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            frame_index += 1
            if args.progress_every and (frame_index % args.progress_every == 0 or frame_index == total_frames):
                print(f"Processed {frame_index}/{total_frames} frames", flush=True)
    capture.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    if masker is not None:
        masker.close()


if __name__ == "__main__":
    main()
