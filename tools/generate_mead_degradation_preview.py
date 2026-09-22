"""Generate deterministic, mask-aware MEAD lighting degradation previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--relight-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--severity", choices=("standard", "extreme"),
                        default="standard")
    return parser.parse_args()


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92,
                    ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(image <= 0.0031308, 12.92 * image,
                    1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def gamma_srgb(image: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(image, 0.0, 1.0), gamma)


def expose(image: np.ndarray, ev: float) -> np.ndarray:
    return linear_to_srgb(srgb_to_linear(image) * (2.0 ** ev))


def white_balance(image: np.ndarray, rgb_gain: tuple[float, float, float]) -> np.ndarray:
    linear = srgb_to_linear(image)
    gain = np.asarray(rgb_gain, dtype=np.float32)[None, None, :]
    return linear_to_srgb(linear * gain)


def composite(original: np.ndarray, changed: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return changed * alpha + original * (1.0 - alpha)


def vignette(height: int, width: int, edge: float = 0.58) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(((xx - 0.5 * width) / (0.72 * width)) ** 2
                     + ((yy - 0.48 * height) / (0.72 * height)) ** 2)
    return np.clip(1.0 - (1.0 - edge) * radius ** 1.7, edge, 1.0)[..., None]


def directional_gradient(height: int, width: int, start: float,
                         end: float, angle_degrees: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1) - 0.5
    y = yy / max(height - 1, 1) - 0.5
    angle = np.deg2rad(angle_degrees)
    coordinate = np.clip(x * np.cos(angle) + y * np.sin(angle) + 0.5, 0.0, 1.0)
    return (start + (end - start) * coordinate)[..., None]


def soft_shadow(height: int, width: int, center_x: float, center_y: float,
                sigma_x: float, sigma_y: float, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gaussian = np.exp(-0.5 * (((xx / width - center_x) / sigma_x) ** 2
                              + ((yy / height - center_y) / sigma_y) ** 2))
    return (1.0 - strength * gaussian)[..., None]


def make_candidates(clean: np.ndarray, person: np.ndarray) -> dict[str, np.ndarray]:
    height, width = clean.shape[:2]
    candidates: dict[str, np.ndarray] = {}

    # Low-exposure webcam: exposure, dark gamma, cool WB and lens falloff.
    changed = expose(clean, -1.15)
    changed = gamma_srgb(changed, 1.18)
    changed = white_balance(changed, (0.86, 0.96, 1.10))
    changed = changed * vignette(height, width, 0.62)
    candidates["01_underexposed_cool"] = np.clip(changed, 0.0, 1.0)

    # Window backlight: the room is bright while the foreground person is dark.
    bright_background = white_balance(expose(clean, 0.55), (0.94, 1.00, 1.08))
    dark_person = gamma_srgb(expose(clean, -1.05), 1.12)
    candidates["02_window_backlight"] = composite(bright_background, dark_person, person)

    # Warm desk lamp from one side, restricted to the foreground person.
    side = directional_gradient(height, width, 1.28, 0.42, 0.0)
    changed = linear_to_srgb(srgb_to_linear(clean) * side)
    changed = white_balance(changed, (1.16, 1.02, 0.75))
    candidates["03_warm_side_light"] = composite(clean, changed, person)

    # Ceiling light plus a broad soft shadow around the eye/upper-face region.
    top = directional_gradient(height, width, 1.18, 0.55, 90.0)
    shadow = soft_shadow(height, width, 0.50, 0.39, 0.34, 0.075, 0.46)
    changed = linear_to_srgb(srgb_to_linear(clean) * top * shadow)
    candidates["04_top_light_soft_shadow"] = composite(clean, changed, person)

    # Moderately clipped warm foreground, representing auto-exposure failure.
    changed = white_balance(expose(clean, 0.90), (1.12, 1.01, 0.80))
    changed = gamma_srgb(changed, 0.90)
    candidates["05_warm_overexposure"] = composite(clean, changed, person)

    # Common mixed office lighting: mild underexposure, vignette, side gradient,
    # slight green cast and a soft local shadow.
    changed = expose(clean, -0.45)
    changed = white_balance(changed, (0.92, 1.08, 0.96))
    gradient = directional_gradient(height, width, 0.70, 1.08, -18.0)
    shadow = soft_shadow(height, width, 0.66, 0.48, 0.18, 0.28, 0.34)
    changed = linear_to_srgb(srgb_to_linear(changed) * gradient * shadow)
    changed = changed * vignette(height, width, 0.78)
    candidates["06_mixed_office"] = np.clip(changed, 0.0, 1.0)
    return candidates


def make_extreme_candidates(clean: np.ndarray,
                            person: np.ndarray) -> dict[str, np.ndarray]:
    height, width = clean.shape[:2]
    candidates: dict[str, np.ndarray] = {}

    changed = gamma_srgb(expose(clean, -2.25), 1.32)
    changed = white_balance(changed, (0.76, 0.90, 1.18))
    changed = changed * vignette(height, width, 0.38)
    candidates["01_extreme_near_dark"] = np.clip(changed, 0.0, 1.0)

    bright_background = white_balance(expose(clean, 1.25), (0.88, 0.98, 1.18))
    dark_person = gamma_srgb(expose(clean, -1.75), 1.25)
    candidates["02_extreme_backlight"] = composite(bright_background, dark_person, person)

    split = directional_gradient(height, width, 1.75, 0.18, 0.0)
    changed = linear_to_srgb(srgb_to_linear(clean) * split)
    changed = white_balance(changed, (1.12, 1.00, 0.82))
    candidates["03_extreme_split_light"] = composite(clean, changed, person)

    top = directional_gradient(height, width, 1.65, 0.25, 90.0)
    eyes = soft_shadow(height, width, 0.50, 0.40, 0.36, 0.065, 0.78)
    changed = linear_to_srgb(srgb_to_linear(clean) * top * eyes)
    candidates["04_extreme_top_shadow"] = composite(clean, changed, person)

    changed = white_balance(expose(clean, 1.60), (1.20, 1.00, 0.67))
    changed = gamma_srgb(changed, 0.80)
    candidates["05_extreme_warm_clip"] = composite(clean, changed, person)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    blend = np.clip(xx / max(width - 1, 1), 0.0, 1.0)[..., None]
    warm = white_balance(expose(clean, -0.25), (1.35, 0.92, 0.60))
    cool = white_balance(expose(clean, -0.85), (0.62, 0.90, 1.42))
    changed = warm * (1.0 - blend) + cool * blend
    changed *= soft_shadow(height, width, 0.58, 0.48, 0.16, 0.30, 0.55)
    candidates["06_extreme_mixed_color"] = composite(clean, changed, person)
    return candidates


def tile(image: np.ndarray, label: str, size: int = 300) -> np.ndarray:
    bgr = cv2.cvtColor(np.uint8(np.clip(image * 255.0 + 0.5, 0, 255)), cv2.COLOR_RGB2BGR)
    result = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (size, 34), (0, 0, 0), thickness=-1)
    cv2.putText(result, label, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_bgr = cv2.imread(args.clean, cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(args.relight_mask, cv2.IMREAD_GRAYSCALE)
    if clean_bgr is None or mask_u8 is None:
        raise FileNotFoundError("Unable to read the clean frame or relight mask.")
    clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    person = mask_u8.astype(np.float32)[..., None] / 255.0
    if person.shape[:2] != clean.shape[:2]:
        person = cv2.resize(person, (clean.shape[1], clean.shape[0]),
                            interpolation=cv2.INTER_LINEAR)[..., None]

    candidates = (make_extreme_candidates(clean, person)
                  if args.severity == "extreme" else make_candidates(clean, person))
    items = [("00_clean_target", clean), *candidates.items()]
    for name, image in items:
        bgr = cv2.cvtColor(np.uint8(np.clip(image * 255.0 + 0.5, 0, 255)),
                           cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{name}.png"), bgr)

    tiles = [tile(image, name) for name, image in items]
    tiles.append(np.zeros_like(tiles[0]))
    montage = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:8])))
    montage_path = output_dir / "degradation_v2_candidates.jpg"
    cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 95])

    foreground = person[..., 0] > 0.5
    for name, image in items:
        y = image @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        print(f"{name}: person_luma={y[foreground].mean():.3f}, "
              f"whole_luma={y.mean():.3f}, clipped={(y >= 0.98).mean() * 100:.2f}%")
    print(f"Montage: {montage_path.resolve()}")


if __name__ == "__main__":
    main()
