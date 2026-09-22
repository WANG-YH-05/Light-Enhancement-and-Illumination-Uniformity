"""Render one cross-person MEAD triplet with known virtual-light parameters.

This is a visual gate for the proposed physics-consistent data stream.  It is
not a training script and does not load a trained RRNet checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.depth import DepthAnythingV2Small
from rrnet.lighting import default_parameter_statistics
from rrnet.renderer import RenderingModule


def read_rgb(path: str) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return rgb, tensor.unsqueeze(0)


def corresponding_mask(clean_path: str) -> np.ndarray:
    path = Path(clean_path)
    mask_path = path.parent.parent / "relight_mask" / path.name
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Relight mask not found: {mask_path}")
    return (mask.astype(np.float32) / 255.0)[..., None]


def make_theta(num_lights: int, kind: str, device: torch.device) -> torch.Tensor:
    """Two plausible achromatic lighting configurations for a preview only."""
    mean, _ = default_parameter_statistics(num_lights)
    theta = mean.clone().reshape(1, -1)
    lights = theta[:, :num_lights * 10].reshape(1, num_lights, 10)
    # Start with a low, spatially neutral ambient component.  A few point
    # lights are then placed near the image plane to create soft directional
    # variation through the existing RRNet renderer.
    lights[..., 0:3] = 0.0
    lights[..., 3:6] = torch.tensor([0.0, 0.0, 1.0])
    lights[..., 6:9] = torch.tensor([0.5, 0.5, 0.65])
    lights[..., 9] = 1.25
    if kind == "input":
        theta[:, -3:] = 0.26
        lights[:, 0, 0:3] = 0.045
        lights[:, 0, 6:9] = torch.tensor([0.30, 0.40, 0.45])
        lights[:, 1, 0:3] = 0.025
        lights[:, 1, 6:9] = torch.tensor([0.75, 0.30, 0.70])
    elif kind == "reference":
        theta[:, -3:] = 0.43
        lights[:, 0, 0:3] = 0.095
        lights[:, 0, 6:9] = torch.tensor([0.28, 0.34, 0.42])
        lights[:, 1, 0:3] = 0.050
        lights[:, 1, 6:9] = torch.tensor([0.72, 0.22, 0.72])
        lights[:, 2, 0:3] = 0.025
        lights[:, 2, 6:9] = torch.tensor([0.50, 0.78, 0.55])
    else:
        raise ValueError(kind)
    return theta.to(device)


def apply(clean: torch.Tensor, illumination: torch.Tensor,
          mask: np.ndarray) -> np.ndarray:
    rendered = (clean * illumination).clamp(0.0, 1.0)
    original = clean[0].permute(1, 2, 0).float().cpu().numpy()
    result = rendered[0].permute(1, 2, 0).float().cpu().numpy()
    return original + mask * (result - original)


def labelled(rgb: np.ndarray, label: str, width: int = 300) -> np.ndarray:
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height))
    cv2.rectangle(bgr, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.putText(bgr, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead_reference_relative_gain.yaml")
    parser.add_argument("--source", required=True, help="Clean MEAD frame of person A")
    parser.add_argument("--reference", required=True, help="Clean MEAD frame of person B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    kwargs = model_kwargs(config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_rgb, source = read_rgb(args.source)
    reference_rgb, reference = read_rgb(args.reference)
    source, reference = source.to(device), reference.to(device)
    source_mask = corresponding_mask(args.source)
    reference_mask = corresponding_mask(args.reference)
    depth = DepthAnythingV2Small(
        kwargs["depth_vendor_root"], kwargs["depth_checkpoint"],
        kwargs["depth_input_size"], kwargs["depth_invert"],
    ).to(device).eval()
    renderer = RenderingModule(
        kwargs["num_lights"], kwargs["sigma1"], kwargs["sigma2"],
        clamp_output=True,
    ).to(device).eval()
    theta_in = make_theta(kwargs["num_lights"], "input", device)
    theta_ref = make_theta(kwargs["num_lights"], "reference", device)
    with torch.inference_mode():
        source_depth = depth(source)
        reference_depth = depth(reference)
        lin, _ = renderer.illumination(source_depth, theta_in)
        lref_on_source, _ = renderer.illumination(source_depth, theta_ref)
        lref_on_reference, _ = renderer.illumination(reference_depth, theta_ref)
        source_input = apply(source, lin, source_mask)
        reference_lit = apply(reference, lref_on_reference, reference_mask)
        target = apply(source, lref_on_source, source_mask)

    gain = (lref_on_source / lin.clamp_min(1.0e-4)).mean(1)[0].float().cpu().numpy()
    gain = cv2.normalize(gain, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gain_rgb = cv2.cvtColor(gain, cv2.COLOR_GRAY2RGB)
    board = cv2.hconcat((
        labelled(source_rgb, "A clean"),
        labelled(np.uint8(np.clip(source_input * 255.0 + 0.5, 0, 255)), "A input: Lin"),
        labelled(np.uint8(np.clip(reference_lit * 255.0 + 0.5, 0, 255)), "B reference: Lref"),
        labelled(np.uint8(np.clip(target * 255.0 + 0.5, 0, 255)), "A target: Lref"),
        labelled(gain_rgb, "Lref / Lin"),
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), board):
        raise RuntimeError(f"Unable to write {output}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
