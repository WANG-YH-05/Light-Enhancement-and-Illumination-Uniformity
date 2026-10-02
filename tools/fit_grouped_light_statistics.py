"""Train-only calibration using the exact grouped dataset parameter sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config
from rrnet.physical_reference_data import MEADPhysicalReferencePairs


def dataset(config, split='train'):
    d = config['data']
    return MEADPhysicalReferencePairs(
        d['root'], split, metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=d.get('sampling_seed', 20260929),
        variants_per_source=d.get('reference_variants_per_source', 4),
        num_lights=config['model']['num_lights'], dynamic_epoch=True,
        reference_lighting_weights=d['reference_lighting_weights'],
        group_size=d.get('physical_group_size', 4),
        reference_sensitivity_fraction=d.get('reference_sensitivity_fraction', .2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--groups', type=int, default=4000)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.groups < 2:
        raise ValueError('Need at least two groups')
    output = Path(a.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Calibration output already exists; choose a new path')
    ds = dataset(load_config(a.config))
    rng = np.random.default_rng(ds.seed + 1107)
    values = []
    sensitivity = 0
    for i in range(a.groups):
        ds.set_epoch(i // 1000)
        group = ds.sample_parameters(int(rng.integers(len(ds))))
        sensitivity += not group['uniform_group']
        # The joint forward encodes 4 unique inputs + 1 shared reference in
        # uniform groups; sensitivity groups encode 1 input + 4 references.
        source = group['source_theta_target']
        reference = group['reference_theta_target']
        values.extend([source if group['uniform_group'] else source[:1],
                       reference[:1] if group['uniform_group'] else reference])
    theta = torch.cat(values).double().numpy()
    mean, std = theta.mean(0), theta.std(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean.astype('float32'), std=std.astype('float32'))
    report = dict(config=str(Path(a.config).resolve()), split='train', groups=a.groups,
                  encoded_samples=len(theta), sensitivity_groups=sensitivity,
                  training_people=sorted(set(ds.people)), seed=ds.seed,
                  weighting='Unique source/reference encodings per physical group',
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  mean=mean.tolist(), std=std.tolist())
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('mean', 'std', 'training_people')}))


if __name__ == '__main__':
    main()
