"""Build a mask-aware, temporally coherent MEAD dataset for RRNet.

Each clean frame produces one identity row and six newly generated degradation
categories. Original six-digit frame names are preserved in every category.
Randomization is deterministic and hierarchical: person profiles are
stratified, clip parameters are distinct, and frame parameters drift smoothly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tools.generate_mead_degradation_preview import (
        composite,
        directional_gradient,
        expose,
        gamma_srgb,
        linear_to_srgb,
        soft_shadow,
        srgb_to_linear,
        vignette,
        white_balance,
    )
except ModuleNotFoundError:  # Direct execution: python tools/build_....py
    from generate_mead_degradation_preview import (
        composite,
        directional_gradient,
        expose,
        gamma_srgb,
        linear_to_srgb,
        soft_shadow,
        srgb_to_linear,
        vignette,
        white_balance,
    )


FIELDS = [
    "sample_id", "clean_frame", "bad_light_frame", "skin_mask",
    "relight_mask", "person_id", "clip_id", "frame_id", "split",
    "degradation_id", "scenario", "severity", "variant",
]

STANDARD_SCENARIOS = (
    "underexposed_cool", "window_backlight", "warm_side_light",
    "top_light_shadow", "warm_overexposure", "mixed_office",
)
EXTREME_SCENARIOS = (
    "extreme_near_dark", "extreme_backlight", "extreme_split_light",
    "extreme_top_shadow", "extreme_warm_clip", "extreme_mixed_color",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--extreme-rate", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Optional smoke-test limit; zero builds all frames")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def person_profiles(people: list[str], seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(people))
    rng.shuffle(order)
    profiles: dict[str, dict[str, float]] = {}
    for person, rank in zip(people, order):
        phase = 2.0 * math.pi * (rank + 0.5) / max(len(people), 1)
        profiles[person] = {
            "temperature": 0.16 * math.sin(phase),
            "tint": 0.07 * math.cos(1.7 * phase),
            "exposure_bias": -0.16 + 0.32 * (rank + 0.5) / max(len(people), 1),
            "direction": (360.0 * rank / max(len(people), 1) + rng.uniform(-5.0, 5.0)) % 360.0,
            "severity_bias": 0.88 + 0.24 * ((rank * 7) % max(len(people), 1)) / max(len(people) - 1, 1),
        }
    return profiles


def wb_from_profile(profile: dict[str, float], warm: float = 0.0,
                    cool: float = 0.0) -> tuple[float, float, float]:
    temperature = profile["temperature"] + warm - cool
    tint = profile["tint"]
    return (float(np.clip(1.0 + temperature, 0.62, 1.42)),
            float(np.clip(1.0 + tint, 0.78, 1.22)),
            float(np.clip(1.0 - temperature, 0.62, 1.42)))


def choose_scenario_config(person: str, clip: str, category: str,
                           profile: dict[str, float], seed: int,
                           extreme_rate: float) -> dict[str, object]:
    rng = np.random.default_rng(stable_seed(seed, person, clip, category))
    extreme = bool(rng.random() < extreme_rate)
    extreme_map = {
        "underexposed_cool": "extreme_near_dark",
        "window_backlight": "extreme_backlight",
        "warm_side_light": "extreme_split_light",
        "top_light_shadow": "extreme_top_shadow",
        "warm_overexposure": "extreme_warm_clip",
        "mixed_office": "extreme_mixed_color",
    }
    scenario = extreme_map[category] if extreme else category
    return {
        "category": category,
        "scenario": scenario,
        "severity": "extreme" if extreme else "standard",
        "ev_jitter": float(rng.uniform(-0.18, 0.18)),
        "gamma_jitter": float(rng.uniform(-0.08, 0.08)),
        "angle": float((profile["direction"] + rng.uniform(-55.0, 55.0)) % 360.0),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        "cycles": float(rng.uniform(0.20, 0.55)),
        "shadow_x": float(rng.uniform(0.35, 0.68)),
        "shadow_y": float(rng.uniform(0.32, 0.58)),
        "profile": profile,
    }


def apply_scenario(clean: np.ndarray, person: np.ndarray,
                   config: dict[str, object], progress: float) -> np.ndarray:
    height, width = clean.shape[:2]
    scenario = str(config["scenario"])
    profile = config["profile"]
    assert isinstance(profile, dict)
    drift = math.sin(float(config["phase"]) + 2.0 * math.pi
                     * float(config["cycles"]) * progress)
    ev_jitter = float(config["ev_jitter"]) + float(profile["exposure_bias"])
    gamma_jitter = float(config["gamma_jitter"])
    severity = float(profile["severity_bias"])
    angle = float(config["angle"])
    slow_ev = 0.08 * drift

    if scenario == "underexposed_cool":
        changed = gamma_srgb(expose(clean, -1.15 * severity + ev_jitter + slow_ev),
                             1.18 + gamma_jitter)
        changed = white_balance(changed, wb_from_profile(profile, cool=0.10))
        return np.clip(changed * vignette(height, width, 0.60 + 0.05 * drift), 0.0, 1.0)
    if scenario == "window_backlight":
        background = white_balance(expose(clean, 0.45 + 0.12 * drift),
                                   wb_from_profile(profile, cool=0.05))
        foreground = gamma_srgb(expose(clean, -1.05 * severity + ev_jitter + slow_ev),
                                1.12 + gamma_jitter)
        return composite(background, foreground, person)
    if scenario == "warm_side_light":
        gradient = directional_gradient(height, width, 1.28, 0.40, angle)
        changed = linear_to_srgb(srgb_to_linear(clean) * gradient)
        changed = white_balance(changed, wb_from_profile(profile, warm=0.14))
        return composite(clean, changed, person)
    if scenario == "top_light_shadow":
        gradient = directional_gradient(height, width, 1.16, 0.50, 90.0 + 10.0 * drift)
        shadow = soft_shadow(height, width, float(config["shadow_x"]), 0.39,
                             0.32, 0.07, 0.42 * severity)
        changed = linear_to_srgb(srgb_to_linear(clean) * gradient * shadow)
        return composite(clean, changed, person)
    if scenario == "warm_overexposure":
        changed = white_balance(expose(clean, 0.82 * severity + ev_jitter + slow_ev),
                                wb_from_profile(profile, warm=0.12))
        changed = gamma_srgb(changed, 0.90 + gamma_jitter * 0.5)
        return composite(clean, changed, person)
    if scenario == "mixed_office":
        changed = white_balance(expose(clean, -0.48 * severity + ev_jitter + slow_ev),
                                wb_from_profile(profile))
        gradient = directional_gradient(height, width, 0.68, 1.08, angle)
        shadow = soft_shadow(height, width, float(config["shadow_x"]),
                             float(config["shadow_y"]), 0.18, 0.28, 0.33 * severity)
        changed = linear_to_srgb(srgb_to_linear(changed) * gradient * shadow)
        return np.clip(changed * vignette(height, width, 0.77), 0.0, 1.0)
    if scenario == "cool_side_shadow":
        gradient = directional_gradient(height, width, 0.48, 1.18, angle)
        shadow = soft_shadow(height, width, float(config["shadow_x"]),
                             float(config["shadow_y"]), 0.22, 0.25, 0.38 * severity)
        changed = linear_to_srgb(srgb_to_linear(clean) * gradient * shadow)
        changed = white_balance(changed, wb_from_profile(profile, cool=0.14))
        return composite(clean, changed, person)
    if scenario == "vignette_lowlight":
        changed = gamma_srgb(expose(clean, -0.72 * severity + ev_jitter + slow_ev),
                             1.12 + gamma_jitter)
        return np.clip(changed * vignette(height, width, 0.48 + 0.04 * drift), 0.0, 1.0)

    if scenario == "extreme_near_dark":
        changed = gamma_srgb(expose(clean, -2.20 + ev_jitter + slow_ev), 1.30)
        changed = white_balance(changed, wb_from_profile(profile, cool=0.18))
        return np.clip(changed * vignette(height, width, 0.36), 0.0, 1.0)
    if scenario == "extreme_backlight":
        background = white_balance(expose(clean, 1.18 + 0.12 * drift),
                                   wb_from_profile(profile, cool=0.12))
        foreground = gamma_srgb(expose(clean, -1.72 + ev_jitter + slow_ev), 1.24)
        return composite(background, foreground, person)
    if scenario == "extreme_split_light":
        gradient = directional_gradient(height, width, 1.78, 0.16, angle)
        changed = linear_to_srgb(srgb_to_linear(clean) * gradient)
        changed = white_balance(changed, wb_from_profile(profile, warm=0.10))
        return composite(clean, changed, person)
    if scenario == "extreme_top_shadow":
        gradient = directional_gradient(height, width, 1.62, 0.24, 90.0)
        shadow = soft_shadow(height, width, float(config["shadow_x"]), 0.40,
                             0.35, 0.065, 0.76)
        changed = linear_to_srgb(srgb_to_linear(clean) * gradient * shadow)
        return composite(clean, changed, person)
    if scenario == "extreme_warm_clip":
        changed = white_balance(expose(clean, 1.55 + ev_jitter + slow_ev),
                                wb_from_profile(profile, warm=0.22))
        changed = gamma_srgb(changed, 0.82)
        return composite(clean, changed, person)
    if scenario == "extreme_mixed_color":
        _, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        blend = np.clip(xx / max(width - 1, 1), 0.0, 1.0)[..., None]
        warm = white_balance(expose(clean, -0.22), wb_from_profile(profile, warm=0.28))
        cool = white_balance(expose(clean, -0.82), wb_from_profile(profile, cool=0.30))
        changed = warm * (1.0 - blend) + cool * blend
        changed *= soft_shadow(height, width, float(config["shadow_x"]),
                               float(config["shadow_y"]), 0.17, 0.30, 0.55)
        return composite(clean, changed, person)
    raise ValueError(f"Unknown scenario: {scenario}")


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.extreme_rate <= 1.0:
        raise ValueError("--extreme-rate must be within [0, 1]")
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clean_paths = sorted(source.glob("*/*/*/clean_frame/*.png"))
    if not clean_paths:
        raise FileNotFoundError(f"No clean frames found under {source}")
    if args.max_frames > 0:
        clean_paths = clean_paths[:args.max_frames]
    people = sorted({path.parts[-4] for path in clean_paths})
    profiles = person_profiles(people, args.seed)

    clip_frames: dict[tuple[str, str, str], list[Path]] = {}
    for path in clean_paths:
        split, person, clip = path.parts[-5], path.parts[-4], path.parts[-3]
        clip_frames.setdefault((split, person, clip), []).append(path)

    manifest: dict[str, object] = {
        "seed": args.seed,
        "degradation_categories": list(STANDARD_SCENARIOS),
        "extreme_rate": args.extreme_rate,
        "source": str(source),
        "people": profiles,
        "clips": {},
    }
    rows: list[dict[str, str]] = []
    total = len(clean_paths)
    completed = 0
    generated = 0
    skipped = 0
    started = time.time()

    for (split, person, clip), frames in sorted(clip_frames.items()):
        frames.sort()
        configs = [choose_scenario_config(person, clip, category, profiles[person],
                                          args.seed, args.extreme_rate)
                   for category in STANDARD_SCENARIOS]
        manifest["clips"][f"{split}/{person}/{clip}"] = [
            {key: value for key, value in config.items() if key != "profile"}
            for config in configs
        ]
        for frame_position, clean_path in enumerate(frames):
            frame_id = clean_path.stem
            source_clip = clean_path.parent.parent
            skin_source = source_clip / "skin_mask" / clean_path.name
            relight_source = source_clip / "relight_mask" / clean_path.name
            if not skin_source.exists() or not relight_source.exists():
                raise FileNotFoundError(f"Missing mask for {clean_path}")
            target_clip = output / split / person / clip
            clean_target = target_clip / "clean_frame" / clean_path.name
            skin_target = target_clip / "skin_mask" / clean_path.name
            relight_target = target_clip / "relight_mask" / clean_path.name
            link_or_copy(clean_path, clean_target, args.copy_mode)
            link_or_copy(skin_source, skin_target, args.copy_mode)
            link_or_copy(relight_source, relight_target, args.copy_mode)
            common = {
                "clean_frame": relative(clean_target, output),
                "skin_mask": relative(skin_target, output),
                "relight_mask": relative(relight_target, output),
                "person_id": person,
                "clip_id": clip,
                "frame_id": frame_id,
                "split": split,
            }
            rows.append({
                "sample_id": f"{person}_{clip}_{frame_id}_identity",
                "bad_light_frame": relative(clean_target, output),
                "degradation_id": "identity",
                "scenario": "identity",
                "severity": "identity",
                "variant": "identity",
                **common,
            })

            clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
            mask_u8 = cv2.imread(str(relight_source), cv2.IMREAD_GRAYSCALE)
            if clean_bgr is None or mask_u8 is None:
                raise RuntimeError(f"Unable to read {clean_path} or its mask")
            clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            person_mask = mask_u8.astype(np.float32) / 255.0
            if person_mask.shape != clean.shape[:2]:
                person_mask = cv2.resize(person_mask, (clean.shape[1], clean.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            person_mask = person_mask[..., None]
            progress = frame_position / max(len(frames) - 1, 1)
            for config in configs:
                category = str(config["category"])
                degraded_target = target_clip / "degraded_frame" / category / clean_path.name
                if degraded_target.exists():
                    skipped += 1
                else:
                    degraded = apply_scenario(clean, person_mask, config, progress)
                    degraded_bgr = cv2.cvtColor(
                        np.uint8(np.clip(degraded * 255.0 + 0.5, 0, 255)),
                        cv2.COLOR_RGB2BGR,
                    )
                    degraded_target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(degraded_target), degraded_bgr,
                                      [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError(f"Unable to write {degraded_target}")
                    generated += 1
                scenario = str(config["scenario"])
                rows.append({
                    "sample_id": f"{person}_{clip}_{frame_id}_{category}",
                    "bad_light_frame": relative(degraded_target, output),
                    "degradation_id": category,
                    "scenario": scenario,
                    "severity": str(config["severity"]),
                    "variant": category,
                    **common,
                })
            completed += 1
            if args.progress_every and (completed % args.progress_every == 0 or completed == total):
                elapsed = time.time() - started
                rate = completed / max(elapsed, 1e-6)
                eta = (total - completed) / max(rate, 1e-6)
                print(f"Frames {completed}/{total} | generated={generated} skipped={skipped} "
                      f"| {rate:.2f} frame/s | ETA={eta / 60:.1f} min", flush=True)

    metadata_path = output / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest["summary"] = {
        "clean_frames": total,
        "metadata_rows": len(rows),
        "generated_images": generated,
        "skipped_existing_images": skipped,
    }
    with (output / "generation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Completed. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
