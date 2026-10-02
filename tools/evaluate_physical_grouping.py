"""Visual gate for physical grouped reference-light training.

Creates two read-only test-split boards:
1) four different Lin values with one fixed Lref/Lout (uniformity), and
2) one fixed Lin with dark/normal references (reference sensitivity).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def panel(image: np.ndarray, title: str, subtitle: str, width: int = 300) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    bgr = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (width, height))
    cv2.rectangle(bgr, (0, 0), (width, 56), (6, 8, 12), -1)
    cv2.putText(bgr, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, subtitle, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (210, 220, 230), 1, cv2.LINE_AA)
    return bgr


def rgb(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def masked_luma(image: torch.Tensor, mask: torch.Tensor) -> float:
    value = luminance(image.unsqueeze(0))[0, 0]
    weight = mask[0].float()
    return float((value * weight).sum() / weight.sum().clamp_min(1.0))


def flatten(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    return value.flatten(0, 1).to(device, non_blocking=True)


@torch.inference_mode()
def infer_group(model: ReferenceRelativeRRNet, item: dict, device: torch.device,
                fp16: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
             for key, value in item.items()}
    source_clean = flatten(batch, "source_clean", device)
    reference_clean = flatten(batch, "reference_clean", device)
    relight_mask = flatten(batch, "relight_mask", device)
    reference_relight_mask = flatten(batch, "reference_relight_mask", device)
    source_face = flatten(batch, "source_light_mask", device)
    reference_face = flatten(batch, "reference_mask", device)
    theta_in = flatten(batch, "source_theta_target", device)
    theta_ref = flatten(batch, "reference_theta_target", device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        source_depth = model.depth(source_clean)
        reference_depth = model.depth(reference_clean)
        source_light, _ = model.illumination_from_theta(source_depth, theta_in)
        target_light, _ = model.illumination_from_theta(source_depth, theta_ref)
        reference_light, _ = model.illumination_from_theta(reference_depth, theta_ref)
        source = model.renderer.blend_relight(
            source_clean * source_light, source_clean, relight_mask).clamp(0, 0.999)
        target = model.renderer.blend_relight(
            source_clean * target_light, source_clean, relight_mask).clamp(0, 0.999)
        reference = model.renderer.blend_relight(
            reference_clean * reference_light, reference_clean,
            reference_relight_mask).clamp(0, 0.999)
        prediction = model(source, reference, relight_mask, reference_face, source_face,
                           source_depth_override=source_depth,
                           reference_depth_override=reference_depth)
    values = {
        "source": source, "target": target, "reference": reference,
        "output": prediction["loss_output"], "mask": relight_mask,
    }
    return values, {"theta_in": theta_in, "theta_ref": theta_ref}


def save_uniformity(output_dir: Path, values: dict[str, torch.Tensor], item: dict) -> dict:
    rows = []
    output_lumas = []
    output_target_maes = []
    for index in range(values["source"].shape[0]):
        mask = values["mask"][index]
        source_y = masked_luma(values["source"][index], mask)
        target_y = masked_luma(values["target"][index], mask)
        output_y = masked_luma(values["output"][index], mask)
        mae = float(((values["output"][index] - values["target"][index]).abs() * mask).sum()
                    / (mask.sum().clamp_min(1.0) * 3.0))
        output_lumas.append(output_y)
        output_target_maes.append(mae)
        rows.append(np.hstack((
            panel(rgb(values["source"][index]), f"INPUT {index + 1}", f"face Y={source_y:.3f}"),
            panel(rgb(values["reference"][index]), "FIXED REFERENCE",
                  f"{item['reference_lighting_mode'][index]} | face target"),
            panel(rgb(values["target"][index]), "EXACT TARGET", f"face Y={target_y:.3f}"),
            panel(rgb(values["output"][index]), "MODEL OUTPUT",
                  f"Y={output_y:.3f} | MAE={mae:.3f}"),
        )))
    board = np.vstack(rows)
    path = output_dir / "uniformity_four_inputs.jpg"
    cv2.imwrite(str(path), board, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return {
        "board": str(path),
        "input_face_luma": [round(masked_luma(values["source"][i], values["mask"][i]), 5)
                              for i in range(4)],
        "target_face_luma": [round(masked_luma(values["target"][i], values["mask"][i]), 5)
                               for i in range(4)],
        "output_face_luma": [round(value, 5) for value in output_lumas],
        "output_target_mae": [round(value, 5) for value in output_target_maes],
        "output_luma_span": round(max(output_lumas) - min(output_lumas), 5),
    }


def save_sensitivity(output_dir: Path, values: dict[str, torch.Tensor], item: dict) -> dict:
    rows = []
    output_lumas = []
    target_lumas = []
    for index in range(values["source"].shape[0]):
        mask = values["mask"][index]
        source_y = masked_luma(values["source"][index], mask)
        target_y = masked_luma(values["target"][index], mask)
        output_y = masked_luma(values["output"][index], mask)
        output_lumas.append(output_y)
        target_lumas.append(target_y)
        rows.append(np.hstack((
            panel(rgb(values["source"][index]), "FIXED INPUT", f"face Y={source_y:.3f}"),
            panel(rgb(values["reference"][index]), "REFERENCE",
                  f"{item['reference_lighting_mode'][index]}"),
            panel(rgb(values["target"][index]), "EXACT TARGET", f"face Y={target_y:.3f}"),
            panel(rgb(values["output"][index]), "MODEL OUTPUT", f"face Y={output_y:.3f}"),
        )))
    path = output_dir / "sensitivity_fixed_input.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 96])
    return {
        "board": str(path),
        "target_face_luma": [round(value, 5) for value in target_lumas],
        "target_luma_span": round(max(target_lumas) - min(target_lumas), 5),
        "output_face_luma": [round(value, 5) for value in output_lumas],
        "output_luma_span": round(max(output_lumas) - min(output_lumas), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index", type=int, default=11)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    data = config["data"]
    common = dict(root=data["root"], split="test", metadata_file=data.get("metadata_file", "metadata.csv"),
                  seed=int(data.get("sampling_seed", 20260924)) + 77, variants_per_source=4,
                  num_lights=int(config["model"].get("num_lights", 9)), dynamic_epoch=False,
                  reference_lighting_weights=data["reference_lighting_weights"], group_size=4)
    uniform = MEADPhysicalReferencePairs(**common, reference_sensitivity_fraction=0.0)
    sensitive = MEADPhysicalReferencePairs(**common, reference_sensitivity_fraction=1.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = args.index % min(len(uniform), len(sensitive))
    uniform_item = uniform[sample_index]
    sensitive_item = sensitive[sample_index]
    uniform_values, _ = infer_group(model, uniform_item, device, fp16)
    sensitive_values, _ = infer_group(model, sensitive_item, device, fp16)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_index": sample_index,
        "source_person": uniform_item["source_person_id"],
        "reference_person": uniform_item["reference_person_id"],
        "uniformity": save_uniformity(output_dir, uniform_values, uniform_item),
        "reference_sensitivity": save_sensitivity(output_dir, sensitive_values, sensitive_item),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
