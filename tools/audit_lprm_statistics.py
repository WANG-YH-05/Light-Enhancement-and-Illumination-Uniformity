"""Compare fallback denormalization scales with the current synthetic theta sampler."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rrnet.lighting import default_parameter_statistics
from rrnet.physical_reference_data import sample_physical_theta, sample_reference_theta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rng=np.random.default_rng(20261001)
    samples=[]
    for i in range(2000):
        samples.append(sample_physical_theta(rng,9))
        samples.append(sample_reference_theta(rng,9,'dark' if i%2 else 'normal'))
    values=torch.stack(samples)
    empirical_mean=values.mean(0)
    empirical_std=values.std(0,unbiased=False)
    fallback_mean,fallback_std=default_parameter_statistics(9)
    fields={'color':slice(0,3),'direction':slice(3,6),'position':slice(6,9),'attenuation':slice(9,10)}
    rows={}
    for name,sl in fields.items():
        indices=torch.arange(90).reshape(9,10)[:,sl].flatten()
        rows[name]={'empirical_mean_average':float(empirical_mean[indices].mean()),
                    'empirical_std_median':float(empirical_std[indices].median()),
                    'fallback_std_median':float(fallback_std[indices].median()),
                    'fallback_to_empirical_std_median':float((fallback_std[indices]/empirical_std[indices].clamp_min(1e-6)).median()),
                    'normalized_mean_offset_max':float(((empirical_mean[indices]-fallback_mean[indices])/fallback_std[indices]).abs().max())}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'fields':rows,'samples':4000,
        'scope':'Approximate marginal sampler distribution (50% source, 25% dark ref, 25% normal ref); not an exact grouped-training histogram. No statistics/configuration changed.',
        'empirical_mean':empirical_mean.tolist(),'empirical_std':empirical_std.tolist()},indent=2),encoding='utf-8')
    print(json.dumps(rows))


if __name__=='__main__':
    main()
