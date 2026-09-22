"""Write a small contact sheet for reference-conditioned training triplets."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rrnet.reference_data import MEADReferenceTriplets


def tile(tensor, label: str, width: int = 256) -> np.ndarray:
    rgb = np.uint8(np.clip(tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5, 0, 255))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height = round(bgr.shape[0] * width / bgr.shape[1])
    image = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(image, label, (7, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    dataset = MEADReferenceTriplets(args.root, args.split)
    rows = []
    for index in range(args.start, min(args.start + args.samples, len(dataset))):
        sample = dataset[index]
        rows.append(np.hstack((
            tile(sample["input"], f'SOURCE ({sample["source_category"]})'),
            tile(sample["reference"], f'REFERENCE ({sample["target_category"]})'),
            tile(sample["target"], "TARGET: source under reference light"),
        )))
    if not rows:
        raise ValueError("No preview samples selected.")
    sheet = np.vstack(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Unable to write {output}")
    print(output.resolve())


if __name__ == "__main__":
    main()
