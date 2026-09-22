"""Benchmark cached RRNet video inference with independent lighting/depth schedules."""

from __future__ import annotations

import argparse

import torch

from benchmark_common import create_output_dir, load_model, save_results, summarize, system_config


def benchmark_mode(model, image: torch.Tensor, light_every: int, depth_every: int,
                   warmup: int, runs: int) -> dict[str, float]:
    theta, depth, illumination = None, None, None
    with torch.inference_mode():
        for index in range(warmup):
            with torch.autocast("cuda", dtype=torch.float16):
                update_light = theta is None or index % light_every == 0
                update_depth = depth is None or index % depth_every == 0
                if update_light:
                    theta = model.estimate_lighting(image)["theta"]
                if update_depth:
                    depth = model.depth(image)
                if illumination is None or update_light or update_depth:
                    illumination, _ = model.renderer.illumination(depth, theta)
                model.renderer.apply_illumination(image, illumination)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        for index, (start, end) in enumerate(zip(starts, ends)):
            start.record()
            with torch.autocast("cuda", dtype=torch.float16):
                update_light = index % light_every == 0
                update_depth = index % depth_every == 0
                if update_light:
                    theta = model.estimate_lighting(image)["theta"]
                if update_depth:
                    depth = model.depth(image)
                if update_light or update_depth:
                    illumination, _ = model.renderer.illumination(depth, theta)
                model.renderer.apply_illumination(image, illumination)
            end.record()
        torch.cuda.synchronize()
    return summarize([start.elapsed_time(end) for start, end in zip(starts, ends)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    model, _ = load_model(args.config, args.checkpoint)
    image = torch.rand(1, 3, args.height, args.width, device="cuda")
    schedules = {
        "quality_light1_depth1_fp16": (1, 1),
        "safe_light3_depth3_fp16": (3, 3),
        "fast_light10_depth10_fp16": (10, 10),
        "conference_light10_depth3_fp16": (10, 3),
    }
    results = {
        name: benchmark_mode(model, image, light_every, depth_every, args.warmup, args.runs)
        for name, (light_every, depth_every) in schedules.items()
    }
    output = create_output_dir(args.output_dir)
    save_results(output, system_config(args), results)
    print((output / "latency_results.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
