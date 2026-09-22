"""Shared helpers for RRNet CUDA latency benchmarks."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

MODEL_ROOT = Path(__file__).resolve().parents[2]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from rrnet.config import load_config, model_kwargs  # noqa: E402
from rrnet.model import RRNet  # noqa: E402


def load_model(config_path: str, checkpoint_path: str) -> tuple[RRNet, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    config = load_config(config_path)
    model = RRNet(**model_kwargs(config, config_path)).cuda().eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model, config


def summarize(milliseconds: list[float]) -> dict[str, float]:
    ordered = sorted(milliseconds)
    p95_index = min(round((len(ordered) - 1) * 0.95), len(ordered) - 1)
    mean_ms = statistics.fmean(milliseconds)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(milliseconds),
        "p95_ms": ordered[p95_index],
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "fps": 1000.0 / mean_ms,
    }


def benchmark_call(call: Callable[[], object], warmup: int, runs: int,
                   use_fp16: bool = True) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                call()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        for start, end in zip(starts, ends):
            start.record()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                call()
            end.record()
        torch.cuda.synchronize()
    return summarize([start.elapsed_time(end) for start, end in zip(starts, ends)])


def create_output_dir(base: str | None) -> Path:
    if base:
        output = Path(base)
    else:
        output = MODEL_ROOT / "outputs" / "benchmarks" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_results(output: Path, config: dict, results: dict) -> None:
    result_path = output / "latency_results.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload.setdefault("benchmark_runs", []).append(config)
        payload.setdefault("results", {}).update(results)
    else:
        payload = {"benchmark_runs": [config], "results": dict(results)}
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["benchmark_runs"], handle, ensure_ascii=False, indent=2)
    lines = []
    for name, metrics in payload["results"].items():
        lines.append(
            f"{name}: mean={metrics['mean_ms']:.3f} ms, median={metrics['median_ms']:.3f} ms, "
            f"p95={metrics['p95_ms']:.3f} ms, fps={metrics['fps']:.2f}"
        )
    (output / "latency_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def system_config(args: object) -> dict:
    return {
        **vars(args),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
