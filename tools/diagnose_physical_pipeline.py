"""Read-only diagnosis of BN, physical transfer and scalar-exposure baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrnet.config import load_config, model_kwargs
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from train import model_prediction
from rrnet.validation_sampling import balanced_validation_indices
from rrnet.lprm import ResolutionBatchNorm2d


def mae(image, target, mask):
    return float(((image.float() - target.float()).abs() * mask).sum()
                 / (mask.sum().clamp_min(1) * 3))


def span(image, mask):
    from rrnet.losses import luminance
    means = ((luminance(image.float()) * mask).sum((1, 2, 3))
             / mask.sum((1, 2, 3)).clamp_min(1))
    return float(means.max() - means.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split-resolution-bn', action='store_true')
    parser.add_argument('--calibrate-batches', type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = model_kwargs(config, args.config)
    if args.split_resolution_bn:
        kwargs['split_resolution_bn'] = True
    model = ReferenceRelativeRRNet(**kwargs).to(device).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state))
    datasets = {}
    for split in ('train', 'val'):
        datasets[split] = MEADPhysicalReferencePairs(
            data['root'], split, metadata_file=data.get('metadata_file', 'metadata.csv'),
            seed=data['sampling_seed'] + (1 if split == 'val' else 0),
            variants_per_source=1, num_lights=config['model']['num_lights'],
            dynamic_epoch=False, reference_lighting_weights=data['reference_lighting_weights'],
            group_size=4, reference_sensitivity_fraction=0.0)
    bn = [m for m in model.base.lprm.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if args.calibrate_batches:
        # Train split only. No optimizer, gradients or weight updates. Reset
        # stats and estimate each resolution independently using broad coverage.
        train_ds = datasets['train']
        train_ds.reference_sensitivity_fraction = 0.20
        calibration_indices = balanced_validation_indices(
            train_ds.people, 1, args.calibrate_batches)
        momenta = {id(m): m.momentum for m in bn}
        for m in bn:
            m.reset_running_stats()
            m.momentum = None
            if isinstance(m, ResolutionBatchNorm2d):
                m.coarse_running_mean.zero_()
                m.coarse_running_var.fill_(1)
                m.coarse_num_batches_tracked.zero_()
        model.train()
        with torch.inference_mode():
            for n, index in enumerate(calibration_indices):
                with torch.autocast(device.type, dtype=torch.float16,
                                    enabled=device.type == 'cuda'):
                    model_prediction(model, default_collate([train_ds[index]]),
                                     device, True, True, True)
                if (n + 1) % 10 == 0:
                    print(f'Calibration {n + 1}/{len(calibration_indices)}', flush=True)
        for m in bn:
            m.momentum = momenta[id(m)]
        model.eval()
        train_ds.reference_sensitivity_fraction = 0.0
        checkpoint_dir = Path(args.output).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        original_parameters = state.get('model', state)
        assert all(torch.equal(p.detach().cpu(), original_parameters[name])
                   for name, p in model.named_parameters()), 'Calibration modified weights'
        # Save inference-only copy; old optimizer state no longer represents
        # this diagnostic checkpoint and is deliberately excluded.
        torch.save({'model': model.state_dict(), 'config': config,
                    'source_checkpoint': str(Path(args.checkpoint).resolve()),
                    'split_resolution_bn': args.split_resolution_bn,
                    'calibration_batches': len(calibration_indices)},
                   checkpoint_dir / 'bn_calibrated.pt')
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    rows = []
    with torch.inference_mode():
        for split, ds in datasets.items():
            seen = set()
            indices = []
            for i, person in enumerate(ds.people):
                if person not in seen:
                    seen.add(person)
                    indices.append(i)
                if len(indices) == 3:
                    break
            for index in indices:
                item = ds[index]
                batch = default_collate([item])
                predictions = {}
                for mode in ('eval_running_stats', 'batch_stats'):
                    model.eval()
                    if mode == 'batch_stats':
                        for module in bn:
                            module.train()
                    with torch.autocast(device.type, dtype=torch.float16,
                                        enabled=device.type == 'cuda'):
                        source, target, mask, skin, pred, extras = model_prediction(
                            model, batch, device, True, True, True)
                    predictions[mode] = pred
                    for name, value in model.named_buffers():
                        value.copy_(buffers[name])
                model.eval()
                p = predictions['eval_running_stats']
                q = predictions['batch_stats']
                oracle = model.transfer(source, p['depth'], extras['source_theta_target'],
                                        extras['reference_theta_target'], mask)
                # Exhaustive scalar L1 baseline; evaluate the same gain range,
                # clipping/headroom and mask blend used by the model.
                best = []
                for i in range(source.shape[0]):
                    best_error = float('inf')
                    best_image = None
                    for gain in torch.linspace(model.min_transfer_gain,
                                               model.max_transfer_gain, 185, device=device):
                        safe = torch.minimum(gain.expand_as(source[i:i+1]),
                                             model.output_headroom / source[i:i+1].amax(1, keepdim=True).clamp_min(1e-4))
                        image = model.renderer.blend_relight(
                            source[i:i+1] * safe, source[i:i+1], mask[i:i+1])
                        error = mae(image, target[i:i+1], mask[i:i+1])
                        if error < best_error:
                            best_error, best_image = error, image
                    best.append(best_image)
                scalar = torch.cat(best)
                face = extras['source_light_mask']
                row = dict(split=split, index=index, person=item['source_person_id'],
                           reference=item['reference_person_id'],
                           eval_mae=mae(p['loss_output'], target, face),
                           batch_stats_mae=mae(q['loss_output'], target, face),
                           oracle_mae=mae(oracle['loss_output'], target, face),
                           best_scalar_mae=mae(scalar, target, face),
                           eval_vs_batch_mae=mae(p['loss_output'], q['loss_output'], face),
                           eval_output_span=span(p['loss_output'], face),
                           batch_output_span=span(q['loss_output'], face),
                           target_span=span(target, face))
                rows.append(row)
                print(json.dumps(row), flush=True)
    val = datasets['val']
    # Actual validation uses variants_per_source=1, batch_size from config.
    count = min(len(val.people), config['training']['validation_batches']
                * config['training']['batch_size'])
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), rows=rows,
                  split_resolution_bn=args.split_resolution_bn,
                  calibration_batches=args.calibrate_batches,
                  bn_layers=len(bn), validation_people=sorted(set(val.people)),
                  current_validation_prefix_people=sorted(set(val.people[:count])),
                  validation_prefix_source_count=count,
                  scope='Six deterministic groups; train and validation; diagnosis, not final test.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
