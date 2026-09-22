"""Create a separate MEAD CSV with clean-to-clean identity pairs for training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="MEAD_processed directory")
    parser.add_argument("--source", default="metadata.csv")
    parser.add_argument("--output", default="metadata_rrnet_identity_train.csv")
    args = parser.parse_args()
    root = Path(args.root)
    source, output = root / args.source, root / args.output
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"No header found in {source}")
        rows = list(reader)

    identities: list[dict[str, str]] = []
    seen_clean_frames: set[str] = set()
    for row in rows:
        # Identity pairs are training-only: validation remains the original hard task.
        if row["split"] != "train" or row["clean_frame"] in seen_clean_frames:
            continue
        seen_clean_frames.add(row["clean_frame"])
        identity = dict(row)
        identity["sample_id"] = f"{row['sample_id']}_identity"
        identity["bad_light_frame"] = row["clean_frame"]
        identity["degradation_id"] = "identity"
        identities.append(identity)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(identities)
    print(f"Wrote {output}")
    print(f"Original rows: {len(rows)} | training identity rows added: {len(identities)} | total: {len(rows) + len(identities)}")


if __name__ == "__main__":
    main()
