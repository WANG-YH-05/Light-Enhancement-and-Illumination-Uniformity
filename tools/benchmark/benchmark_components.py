"""Benchmark RRNet lighting estimation, depth estimation, and rendering separately."""

from __future__ import annotations

import argparse

import torch

from benchmark_common import benchmark_call, create_output_dir, load_model, save_results, system_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    model, _ = load_model(args.config, args.checkpoint)
    image = torch.rand(1, 3, args.height, args.width, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        theta = model.estimate_lighting(image)["theta"]
        depth = model.depth(image)
    results = {
        "lprm_fp16": benchmark_call(lambda: model.estimate_lighting(image), args.warmup, args.runs),
        "depth_fp16": benchmark_call(lambda: model.depth(image), args.warmup, args.runs),
        "renderer_cached_fp16": benchmark_call(
            lambda: model.renderer(image, depth, theta), args.warmup, args.runs
        ),
    }
    output = create_output_dir(args.output_dir)
    save_results(output, system_config(args), results)
    print((output / "latency_results.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
