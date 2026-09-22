from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rrnet.config import load_config, model_kwargs
from rrnet.model import RRNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRNet(**model_kwargs(config, args.config)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    image = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)["output"][0].permute(1, 2, 0).cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8)).save(args.output)


if __name__ == "__main__":
    main()
