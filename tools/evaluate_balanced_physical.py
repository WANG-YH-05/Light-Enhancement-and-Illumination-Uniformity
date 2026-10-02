"""Evaluate physical checkpoints using deterministic identity-balanced val data."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from rrnet.validation_sampling import balanced_validation_indices
from train import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    c = load_config(args.config)
    d = c['data']
    ds = MEADPhysicalReferencePairs(
        d['root'], 'val', metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=d['sampling_seed'] + 1, variants_per_source=1,
        num_lights=c['model']['num_lights'], dynamic_epoch=False,
        reference_lighting_weights=d['reference_lighting_weights'],
        group_size=d['validation_physical_group_size'],
        reference_sensitivity_fraction=d['validation_reference_sensitivity_fraction'])
    indices = balanced_validation_indices(ds.people, 1,
        c['training']['validation_batches'] * c['training']['batch_size'])
    loader = DataLoader(Subset(ds, indices), batch_size=c['training']['batch_size'],
                        num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceRelativeRRNet(**model_kwargs(c, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state))
    metrics = validate(model, loader, ReferenceRelativeLoss(**c['loss']).to(device),
                       device, c['training']['validation_batches'], True, True, True)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), indices=indices,
                  people={p: sum(ds.people[i] == p for i in indices)
                          for p in sorted(set(ds.people))}, metrics=metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
