"""Create still-image comparison boards for reference-relative RRNet.

Each board contains one fixed reference image and four deliberately different
degraded inputs with their corresponding model outputs. Samples come from the
held-out MEAD validation/test identities and never use a reference identity as
an input identity in the same board.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.person_mask import suppress_boundary_gain
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet


@dataclass(frozen=True)
class Request:
    person: str
    clip_number: int
    frame: int
    variant: str


GROUPS = [
    {
        "name": "Group 1 - Reference: participant 37",
        "reference": Request("MEAD_front_37", 1, 0, "identity"),
        "inputs": [
            Request("MEAD_front_31", 2, 10, "underexposed_cool"),
            Request("MEAD_front_8", 3, 20, "warm_overexposure"),
            Request("MEAD_front_11", 4, 30, "window_backlight"),
            Request("MEAD_front_14", 5, 40, "warm_side_light"),
        ],
    },
    {
        "name": "Group 2 - Reference: participant 11",
        "reference": Request("MEAD_front_11", 2, 15, "identity"),
        "inputs": [
            Request("MEAD_front_8", 1, 0, "top_light_shadow"),
            Request("MEAD_front_31", 3, 20, "mixed_office"),
            Request("MEAD_front_14", 4, 30, "underexposed_cool"),
            Request("MEAD_front_37", 5, 40, "warm_overexposure"),
        ],
    },
    {
        "name": "Group 3 - Reference: participant 8",
        "reference": Request("MEAD_front_8", 3, 20, "identity"),
        "inputs": [
            Request("MEAD_front_14", 1, 0, "window_backlight"),
            Request("MEAD_front_37", 2, 15, "warm_side_light"),
            Request("MEAD_front_31", 4, 30, "top_light_shadow"),
            Request("MEAD_front_11", 5, 40, "mixed_office"),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def clip_number(clip_id: str) -> int:
    return int(clip_id.split("_", 2)[1])


def load_rows(root: Path) -> list[dict[str, str]]:
    wanted_people = {request.person for group in GROUPS
                     for request in [group["reference"], *group["inputs"]]}
    rows = []
    with (root / "metadata.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in {"test", "val"} and row["person_id"] in wanted_people:
                rows.append(row)
    return rows


def find_row(rows: list[dict[str, str]], request: Request) -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["person_id"] == request.person
        and clip_number(row["clip_id"]) == request.clip_number
        and row["variant"] == request.variant
    ]
    if not candidates:
        raise LookupError(f"No dataset row for {request}")
    return min(candidates, key=lambda row: abs(int(row["frame_id"]) - request.frame))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)[..., None] / 255.0


def tensor_rgb(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_mask(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def infer(model: ReferenceRelativeRRNet, source: np.ndarray, reference_theta: torch.Tensor,
          person_mask: np.ndarray, device: torch.device, use_fp16: bool) -> np.ndarray:
    source_tensor = tensor_rgb(source, device)
    light_mask = tensor_mask(face_attention_from_person_mask(person_mask), device)
    # Match video inference: preserve the background and fade the gain at the
    # silhouette so the comparison does not overstate boundary artifacts.
    output_mask = suppress_boundary_gain(person_mask, fade_pixels=8.0)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        source_theta = model.estimate_light(source_tensor, light_mask)["theta"]
        depth = model.depth(source_tensor)
        prediction = model.transfer(
            source_tensor, depth, source_theta, reference_theta,
            tensor_mask(output_mask, device),
        )["output"]
    return np.uint8(np.clip(
        prediction[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    ) * 255.0 + 0.5)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    paths = [Path("C:/Windows/Fonts") / name, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def chinese_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return font(size, bold)


def fit_square(image: np.ndarray, size: int) -> Image.Image:
    pil = Image.fromarray(image)
    pil.thumbnail((size, size), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (size, size), "#111318")
    result.paste(pil, ((size - pil.width) // 2, (size - pil.height) // 2))
    return result


def draw_tile(canvas: Image.Image, image: np.ndarray, xy: tuple[int, int], size: int,
              heading: str, detail: str) -> None:
    x, y = xy
    canvas.paste(fit_square(image, size), (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + size, y + 40), fill=(0, 0, 0, 190))
    heading_font = chinese_font(22, True) if any(ord(char) > 127 for char in heading) else font(22, True)
    detail_font = chinese_font(17) if any(ord(char) > 127 for char in detail) else font(17)
    draw.text((x + 12, y + 8), heading, font=heading_font, fill="white")
    draw.text((x + 8, y + size + 7), detail, font=detail_font, fill="#D7DCE5")


def make_board(group_index: int, reference_person: str, reference: np.ndarray,
               pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]], output: Path) -> None:
    width, height = 1510, 1335
    canvas = Image.new("RGB", (width, height), "#0B0D12")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 18), f"第 {group_index} 组：固定参考人物 {reference_person}",
              font=chinese_font(32, True), fill="#FFFFFF")
    draw.text((35, 64), "右侧每一行：不同光照输入 → 正确目标 → 模型输出",
              font=chinese_font(20), fill="#F6D365")
    draw_tile(canvas, reference, (35, 115), 420, "固定参考图", "本组四个输入都使用这一参考光照")
    draw.text((35, 575), "如何查看：", font=chinese_font(21, True), fill="#FFFFFF")
    draw.text((35, 614), "1. 输入：人为制造的不同光照", font=chinese_font(18), fill="#D7DCE5")
    draw.text((35, 649), "2. 正确目标：该输入人物处于参考光照下", font=chinese_font(18), fill="#D7DCE5")
    draw.text((35, 684), "3. 模型输出应尽量接近正确目标", font=chinese_font(18), fill="#D7DCE5")
    draw.text((35, 736), "注意：比较“正确目标”和“模型输出”，",
              font=chinese_font(18, True), fill="#F6D365")
    draw.text((35, 770), "不是要求不同肤色的人像素亮度完全一样。",
              font=chinese_font(18, True), fill="#F6D365")

    tile = 270
    start_x = 530
    for index, (source, target, result, variant, _) in enumerate(pairs):
        row = index
        y = 115 + row * 300
        draw_tile(canvas, source, (start_x, y), tile, f"输入 {index + 1}", variant)
        draw_tile(canvas, target, (start_x + tile + 25, y), tile,
                  "正确目标", "同一个人 + 参考光照")
        draw_tile(canvas, result, (start_x + 2 * (tile + 25), y), tile,
                  "模型输出", "应与左侧正确目标接近")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=96)


def make_unification_board(group_index: int, reference_person: str,
                           reference: np.ndarray,
                           pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]],
                           output: Path) -> None:
    """Simple user-facing board: one light reference and four distinct people."""
    width, height = 1260, 1390
    canvas = Image.new("RGB", (width, height), "#0B0D12")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 18), f"第 {group_index} 组：所有人物统一到同一参考光照",
              font=chinese_font(31, True), fill="white")
    draw.text((35, 62), "参考图只指定光照；输出保留每个输入人物的身份",
              font=chinese_font(19), fill="#F6D365")
    draw_tile(canvas, reference, (35, 115), 400, "目标光照参考",
              f"参考人物 {reference_person}（只取光照）")
    draw.text((35, 555), "右侧四行是四个不同人物：",
              font=chinese_font(20, True), fill="white")
    draw.text((35, 595), "不同人物、不同输入光照",
              font=chinese_font(18), fill="#D7DCE5")
    draw.text((35, 630), "↓",
              font=chinese_font(26, True), fill="#F6D365")
    draw.text((35, 675), "各自身份保持不变，",
              font=chinese_font(18), fill="#D7DCE5")
    draw.text((35, 710), "光照向左侧参考统一",
              font=chinese_font(18), fill="#D7DCE5")

    tile, start_x = 320, 500
    for index, (source, _, result, variant, person_id) in enumerate(pairs):
        y = 115 + index * 315
        short_id = person_id.removeprefix("MEAD_front_")
        draw_tile(canvas, source, (start_x, y), tile,
                  f"人物 {short_id}｜输入", variant)
        draw_tile(canvas, result, (start_x + tile + 35, y), tile,
                  f"人物 {short_id}｜输出", "目标：左侧参考光照")
        draw.line((start_x + tile + 8, y + tile // 2,
                   start_x + tile + 27, y + tile // 2),
                  fill="#F6D365", width=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=96)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    rows = load_rows(root)

    for group_index, group in enumerate(GROUPS, start=1):
        reference_row = find_row(rows, group["reference"])
        reference = load_rgb(root / reference_row["clean_frame"])
        reference_person_mask = load_mask(root / reference_row["relight_mask"])
        reference_mask = tensor_mask(
            face_attention_from_person_mask(reference_person_mask), device)
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            reference_theta = model.encode_reference(
                tensor_rgb(reference, device), reference_mask)["theta"].detach()

        pairs = []
        for request in group["inputs"]:
            row = find_row(rows, request)
            source = load_rgb(root / row["bad_light_frame"])
            # All references in these boards use the identity/normal lighting
            # category, therefore the exact target is this source person's
            # clean frame rather than the reference person's RGB appearance.
            target = load_rgb(root / row["clean_frame"])
            person_mask = load_mask(root / row["relight_mask"])
            result = infer(model, source, reference_theta, person_mask, device, use_fp16)
            pairs.append((source, target, result, row["variant"], row["person_id"]))
        output = output_dir / f"reference_comparison_group_{group_index}.jpg"
        make_board(group_index, group["reference"].person.removeprefix("MEAD_front_"),
                   reference, pairs, output)
        simple_output = output_dir / f"unification_group_{group_index}.jpg"
        make_unification_board(
            group_index, group["reference"].person.removeprefix("MEAD_front_"),
            reference, pairs, simple_output)
        print(f"Saved: {output}", flush=True)
        print(f"Saved: {simple_output}", flush=True)


if __name__ == "__main__":
    main()
